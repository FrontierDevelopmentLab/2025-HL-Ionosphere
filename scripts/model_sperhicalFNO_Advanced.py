import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime as _dt

########################################
# 0. Utilities: grids, time helpers, PE
########################################

# Lazy grid cache
_LAT_GRID_CACHED = None
_LON_GRID_CACHED = None

def _load_latlon():
    """Load (H,W) lat/lon numpy grids once."""
    global _LAT_GRID_CACHED, _LON_GRID_CACHED
    if _LAT_GRID_CACHED is None or _LON_GRID_CACHED is None:
        _LAT_GRID_CACHED = np.load('lat_grid.npy')   # (180, 360) float
        _LON_GRID_CACHED = np.load('lon_grid.npy')   # (180, 360) float
    return _LAT_GRID_CACHED, _LON_GRID_CACHED

def _load_qd_for_year(year: int):
    """Quiet-Dipole (QD) lat/lon per year; falls back to geographic lat/lon if not available."""
    try:
        qd_lat = np.load(f'qd_lat_{year}.npy')
        qd_lon = np.load(f'qd_lon_{year}.npy')
        return qd_lat, qd_lon
    except Exception:
        # Fallback to geographic if QD files are missing
        lat, lon = _load_latlon()
        return lat, lon

# Subsolar longitude (Skyfield if present, else simple fallback)
try:
    from skyfield.api import load as _sf_load, wgs84 as _wgs84
    _TS = _sf_load.timescale()
    _EPH = _sf_load('de421.bsp')
    _EARTH = _EPH['earth']
    _SUN = _EPH['sun']
    def _subsolar_lon(dt: _dt.datetime) -> float:
        t = _TS.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        subsolar = _wgs84.subpoint(_EARTH.at(t).observe(_SUN))
        return float(subsolar.longitude.degrees)
except Exception:
    def _subsolar_lon(dt: _dt.datetime) -> float:
        # crude approximation if Skyfield isn't available
        return (dt.hour / 24.0) * 360.0 - 180.0

def _safe_dt(times_obj, b: int, idx: int) -> _dt.datetime:
    """Robustly extract a datetime for sample b at index idx from a batched times container."""
    def _to_dt(x):
        if isinstance(x, _dt.datetime): return x
        if isinstance(x, str): return _dt.datetime.fromisoformat(x)
        return _dt.datetime.fromisoformat(str(x))
    try:
        tb = times_obj[b]
    except Exception:
        tb = times_obj
    try:
        if isinstance(tb, (torch.Tensor, np.ndarray)):
            if tb.ndim == 0: return _to_dt(tb.item())
            if tb.ndim == 1: return _to_dt(tb[min(idx, tb.shape[0]-1)])
            if tb.ndim == 2: return _to_dt(tb[0, min(idx, tb.shape[1]-1)])
    except Exception:
        pass
    if isinstance(tb, (list, tuple)):
        return _to_dt(tb[min(idx, len(tb)-1)])
    return _to_dt(tb)

def _to_batched_grid(grid, B, device):
    """
    Accept grid of shape (H,W) or (B,H,W) or torch tensor; return torch float (B,H,W) on device.
    """
    g = torch.as_tensor(grid, dtype=torch.float32, device=device)
    if g.ndim == 2:
        g = g.unsqueeze(0).expand(B, -1, -1)        # (B,H,W)
    elif g.ndim == 3 and g.shape[0] != B:
        g = g.expand(B, -1, -1)                     # broadcast batch if needed
    return g                                         # (B,H,W)

def add_fourier_positional_encoding(
    x,                       # (B, C, H, W)
    coord_grids,             # list of (H,W) or (B,H,W) arrays/tensors
    n_harmonics: int = 1,
    concat_original_coords: bool = True,
):
    """
    Robust PE builder that supports batched grids.
    For each grid 'g', we map it to angle theta in [-pi, pi], then append:
      - (optional) theta
      - sin(k*theta), cos(k*theta) for k = 1..n_harmonics
    Returns: (B, C + added, H, W)
    """
    B, C, H, W = x.shape
    device = x.device
    feats = [x]

    for grid in coord_grids:
        g = _to_batched_grid(grid, B=B, device=device)          # (B,H,W)
        # normalize each batch sample to [-pi, pi]
        gmin = g.amin(dim=(1, 2), keepdim=True)
        gmax = g.amax(dim=(1, 2), keepdim=True)
        theta = (g - gmin) / (gmax - gmin + 1e-8)
        theta = theta * (2 * np.pi) - np.pi                      # (B,H,W) in [-pi,pi]
        if concat_original_coords:
            feats.append(theta.unsqueeze(1))                     # (B,1,H,W)
        for k in range(1, int(n_harmonics) + 1):
            feats.append(torch.sin(k * theta).unsqueeze(1))      # (B,1,H,W)
            feats.append(torch.cos(k * theta).unsqueeze(1))      # (B,1,H,W)

    return torch.cat(feats, dim=1)                               # (B, C+enc, H, W)

def _build_step_coord_grids(B, H, W, step_times, n_heads: int):
    """
    Build per-step coordinate grids:
      lat, lon, qd_lat, qd_lon, sunlocked_degrees
    and the integer head indices (0..n_heads-1) per-pixel.
    Returns:
      coord_grids (list of np/torch arrays of shape (B,H,W))
      sunlocked_idx (torch.LongTensor, B,H,W)
    """
    lat_np, lon_np = _load_latlon()                              # (H,W)
    sunlocked_deg = []
    qd_lat_list, qd_lon_list = [], []

    for dt in step_times:
        subsolar = _subsolar_lon(dt)
        # degrees in [0,360)
        deg_grid = ((lon_np - subsolar + 360.0) % 360.0).astype(np.float32)  # (H,W)
        sunlocked_deg.append(deg_grid)
        qd_lat_np, qd_lon_np = _load_qd_for_year(dt.year)
        qd_lat_list.append(qd_lat_np.astype(np.float32))
        qd_lon_list.append(qd_lon_np.astype(np.float32))

    # Stack to (B,H,W)
    sunlocked_deg = np.stack(sunlocked_deg, axis=0)              # (B,H,W)
    qd_lat_list   = np.stack(qd_lat_list,   axis=0)              # (B,H,W)
    qd_lon_list   = np.stack(qd_lon_list,   axis=0)              # (B,H,W)

    # Quantize degrees -> head indices 0..n_heads-1 (generalized for arbitrary n_heads)
    # Map [0,360) -> [0,n_heads) then floor and clamp
    mul = float(n_heads) / 360.0
    idx = np.floor(sunlocked_deg * mul).astype(np.int64)
    idx = np.clip(idx, 0, n_heads - 1)
    sunlocked_idx = torch.tensor(idx, dtype=torch.long)

    coord_grids = [
        np.repeat(lat_np[None, ...], B, axis=0),                 # (B,H,W)
        np.repeat(lon_np[None, ...], B, axis=0),                 # (B,H,W)
        qd_lat_list,                                             # (B,H,W)
        qd_lon_list,                                             # (B,H,W)
        sunlocked_deg,                                           # (B,H,W) as a PE channel too
    ]
    return coord_grids, sunlocked_idx


########################################
# 1. SFNO Core  (safer bandlimits + SHT hook)
########################################

class SphericalFourierLayer(nn.Module):
    """
    Spectral mixing layer.
    - Default path uses rFFT2 on equirectangular lat/lon grids.
    - Optional SHT hook: if `use_sht=True` and a backend is available, we could
      dispatch to a real spherical-harmonic transform. We fall back gracefully.
    """
    def __init__(self, in_channels, out_channels, modes_lat, modes_lon, use_sht: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_lat, modes_lon, 2) * 0.01
        )
        self.use_sht = bool(use_sht)
        self._sht_backend = None     # 'torch_harmonics' | 's2fft' | None
        self._sht_setup_done = False # lazily decided at first forward()

    # ---- optional SHT hook (lazy) ----
    def _maybe_setup_sht(self):
        if self._sht_setup_done:
            return
        self._sht_setup_done = True
        if not self.use_sht:
            return
        try:
            import torch_harmonics  # noqa: F401
            self._sht_backend = 'torch_harmonics'
        except Exception:
            try:
                import s2fft  # noqa: F401
                self._sht_backend = 's2fft'
            except Exception:
                self._sht_backend = None
                warnings.warn(
                    "use_sht=True requested but no SHT backend found "
                    "(torch_harmonics / s2fft). Falling back to rFFT2.",
                    RuntimeWarning
                )

    def compl_mul2d(self, x, w):
        w = torch.view_as_complex(w)                     # (..., 2) -> complex
        return torch.einsum("bchw,cohw->bohw", x, w)     # spectral matmul

    def _fourier2_path(self, x):
        """Safe rFFT2 path with runtime bandlimit guards."""
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")                # (B,C,H,W//2+1)
        # Runtime-safe bandlimits (Nyquist guards)
        max_lat = min(self.weight.shape[2], H // 2)            # ≤ H//2
        max_lon = min(self.weight.shape[3], x_ft.shape[-1])    # ≤ W//2+1
        x_ft = x_ft[:, :, :max_lat, :max_lon]
        w = self.weight[:, :, :max_lat, :max_lon, :]           # (C,Co,L,M,2)
        out_ft = self.compl_mul2d(x_ft, w)                     # (B,Co,L,M)
        # Rebuild full rFFT spectrum with correct width
        full_w = x_ft.shape[-1]                                # == W//2+1
        out_ft_full = torch.zeros(
            B, self.weight.shape[1], H, full_w, dtype=torch.cfloat, device=x.device
        )
        out_ft_full[:, :, :max_lat, :max_lon] = out_ft
        return torch.fft.irfft2(out_ft_full, s=(H, W), norm="ortho")

    def _sht_path_or_fallback(self, x):
        """
        Placeholder for a true SHT path.
        If an SHT backend is detected but unsupported grid/ops raise, we fallback.
        """
        try:
            # If needed, one could implement:
            # - Real SHT to (l,m), truncate to (modes_lat, modes_lon),
            # - complex-einsum with learned weights, inverse SHT back to grid.
            # Without strict grid/library guarantees, we fallback safely.
            raise NotImplementedError
        except Exception:
            # Fallback to safe rFFT2
            return self._fourier2_path(x)

    def forward(self, x):
        self._maybe_setup_sht()
        if self.use_sht and (self._sht_backend is not None):
            return self._sht_path_or_fallback(x)
        else:
            return self._fourier2_path(x)


class MCDropout(nn.Dropout):
    def __init__(self, p=0.35):
        super().__init__(p)
        self.mc = False
    def forward(self, x):
        return F.dropout(x, self.p, training=self.training or self.mc)


class SFNOBlock(nn.Module):
    def __init__(self, width, modes_lat, modes_lon, dropout=0.0, mc_dropout=False, use_sht: bool = False):
        super().__init__()
        self.fourier = SphericalFourierLayer(width, width, modes_lat, modes_lon, use_sht=use_sht)
        self.linear = nn.Conv2d(width, width, 1)
        self.dropout = MCDropout(dropout)
        self.dropout.mc = mc_dropout
        self.norm = nn.LayerNorm(width)  # channel LN after HWC permute
        self.act = nn.GELU()
    def set_mc_dropout(self, mc=True):
        self.dropout.mc = mc
    def forward(self, x):
        x1 = self.fourier(x)
        x2 = self.linear(x)
        out = x1 + x2
        out = self.act(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0, 3, 1, 2)
        return out


########################################
# 2. (Fixed) General Fourier Harmonic Positional Encoding
########################################
# Implemented above: add_fourier_positional_encoding(...)

########################################
# 3. SFNO Model (last-frame + PE input; proper AR rollout)
########################################

def _build_area_weights_tensor(device):
    """cos(latitude) area weighting (H,W) -> (1,1,H,W) tensor on device."""
    lat_np, _ = _load_latlon()
    lat_rad = np.deg2rad(lat_np.astype(np.float32))
    w = np.cos(lat_rad)
    w = np.clip(w, 1e-3, None)
    w_t = torch.from_numpy(w).to(device=device, dtype=torch.float32)  # (H,W)
    return w_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

class SphericalFourierNeuralOperatorModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        trunk_width: int = 64,
        trunk_depth: int = 6,
        modes_lat: int = 32,
        modes_lon: int = 64,
        aux_dim: int = 0,
        tasks: tuple = ('vtec',),
        out_shapes: dict = {'vtec': (1, 'grid')},
        probabilistic: bool = True,
        dropout: float = 0.2,
        mc_dropout: bool = False,
        n_sunlocked_heads: int = 360,
        context_window: int = 4,
        prediction_window: int = 1,
        use_sht: bool = False,                # (6) SHT hook flag
        area_weighted_loss: bool = False,     # (5) optional spherical area weighting
    ):
        super().__init__()
        self.trunk_width = trunk_width
        self.tasks = tasks
        self.out_shapes = dict(out_shapes)
        self.aux_dim = aux_dim
        self.probabilistic = probabilistic
        self.n_sunlocked_heads = int(n_sunlocked_heads)
        self.context_window = int(context_window)
        self.prediction_window = int(prediction_window)
        self.use_sht = bool(use_sht)
        self.area_weighted_loss = bool(area_weighted_loss)

        # expose for checkpoint save/load
        self.dropout = float(dropout)
        self.mc_dropout = bool(mc_dropout)

        # Normalize out_shapes to "channels PER STEP"
        self.frame_channels = {}
        for t in self.tasks:
            ch, layout = self.out_shapes.get(t, (1, 'grid'))
            ch = max(1, int(ch))
            if ch > 1 and t == 'vtec':
                ch = 1
            self.frame_channels[t] = ch
            self.out_shapes[t] = (ch, layout)

        self.in_proj = nn.Conv2d(in_channels, trunk_width, 1)
        self.blocks = nn.ModuleList([
            SFNOBlock(trunk_width, modes_lat, modes_lon, dropout=dropout, mc_dropout=mc_dropout, use_sht=self.use_sht)
            for _ in range(trunk_depth)
        ])
        self.aux_proj = nn.Linear(aux_dim, trunk_width) if aux_dim > 0 else None

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            out_dim_per_step = self.frame_channels[task]
            out_channels = out_dim_per_step * (2 if self.probabilistic else 1)
            self.heads[task] = nn.ModuleList([
                nn.Conv2d(trunk_width, out_channels, 1) for _ in range(self.n_sunlocked_heads)
            ])

        # (5) area weights cache (device-specific)
        self._lat_w = None

    def set_mc_dropout(self, mc=True):
        for b in self.blocks:
            b.set_mc_dropout(mc)

    def forward(self, x, sunlocked_lon_grid, aux=None):
        """
        x: (B, in_channels, H, W)  [one frame + PE]
        sunlocked_lon_grid: (B,H,W) Long indices in [0, n_sunlocked_heads-1]
        """
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        if self.aux_proj is not None and aux is not None:
            aux_emb = self.aux_proj(aux).unsqueeze(-1).unsqueeze(-1)
            x = x + aux_emb

        B, _, H, W = x.shape

        if not torch.is_tensor(sunlocked_lon_grid):
            sunlocked_lon_grid = torch.tensor(sunlocked_lon_grid, device=x.device)
        if sunlocked_lon_grid.dim() == 2:
            sunlocked_lon_grid = sunlocked_lon_grid.unsqueeze(0).expand(B, -1, -1)
        sunlocked_lon_grid = sunlocked_lon_grid.to(dtype=torch.long).clamp(0, self.n_sunlocked_heads - 1)

        outs = {}
        for task in self.tasks:
            out_dim = self.frame_channels[task]
            out_channels = out_dim * (2 if self.probabilistic else 1)
            outputs = torch.zeros((B, out_channels, H, W), device=x.device, dtype=x.dtype)

            for head_idx in range(self.n_sunlocked_heads):
                mask = (sunlocked_lon_grid == head_idx)  # (B,H,W)
                if mask.any():
                    out_sub = self.heads[task][head_idx](x)  # (B, out_channels, H, W)
                    # scatter per-batch into masked pixels
                    for b in range(B):
                        mb = mask[b]
                        if mb.any():
                            outputs[b, :, mb] = out_sub[b, :, mb]

            if self.probabilistic:
                mu, logvar = torch.split(outputs, out_dim, dim=1)
                outs[task] = (mu, logvar)
            else:
                outs[task] = outputs
        return outs

    @staticmethod
    def gaussian_nll(mu, log_sigma, x):
        # legacy: expects log_sigma
        return 0.5 * ((x - mu) / log_sigma.exp()).pow(2) + log_sigma + 0.5 * np.log(2 * np.pi)

    @staticmethod
    def gaussian_nll_from_logvar(mu, logvar, x):
        # (4) correct NLL when head produces log-variance
        return 0.5 * ((x - mu) ** 2 * torch.exp(-logvar) + logvar + np.log(2 * np.pi))

    # -----------------------------
    # Proper multi-step AR loss (matches run.py semantics)
    # -----------------------------
    def loss(
        self,
        batch,                       # (B, T, C, H, W) with ch0 = VTEC
        times=None,                  # OPTIONAL time container to derive sun-locked/QD per step
        sunlocked_lon_grid=None,     # OPTIONAL (B,H,W) or (B,P,H,W) indices; used if times is None
        n_harmonics: int = 1,
        reduction: str = "mean",
    ):
        """
        Uses the last `context_window` frame as input (plus PE) and rolls out
        `prediction_window` steps autoregressively. Per step:
          - recompute PE (lat, lon, QD, sun-locked) for that timestep,
          - feed ONE frame + PE (constant in_channels),
          - replace ONLY VTEC channel in the next frame with the prediction (true AR),
          - exogenous channels taken from ground truth at that step.
        Returns a scalar MSE averaged across the P steps.
        """
        assert batch.dim() == 5, "Expected (B,T,C,H,W)"
        B, T, Ctot, H, W = batch.shape
        assert T >= self.context_window + self.prediction_window, \
            f"Need at least context+prediction frames, got {T}"

        device = batch.device
        P = int(self.prediction_window)
        cur = batch[:, self.context_window - 1].clone()          # (B,Ctot,H,W)
        total = 0.0

        # (5) prepare area weights lazily
        if self.area_weighted_loss and (self._lat_w is None or self._lat_w.device != device):
            self._lat_w = _build_area_weights_tensor(device)     # (1,1,H,W) fixed to your grid

        for step in range(P):
            step_idx = self.context_window - 1 + step

            if times is not None:
                # Build per-batch coordinate grids and head indices from times
                step_times = [_safe_dt(times, b, step_idx) for b in range(B)]
                coord_grids, sl_idx = _build_step_coord_grids(B, H, W, step_times, self.n_sunlocked_heads)
                sl_idx = sl_idx.to(device=device)
            else:
                # Use provided sun-locked indices (static or per-step)
                if sunlocked_lon_grid is None:
                    raise ValueError("Either `times` or `sunlocked_lon_grid` must be provided.")
                if torch.is_tensor(sunlocked_lon_grid):
                    if sunlocked_lon_grid.dim() == 4:        # (B,P,H,W)
                        sl_idx = sunlocked_lon_grid[:, step].to(dtype=torch.long, device=device)
                    else:                                    # (B,H,W) or (H,W)
                        s = sunlocked_lon_grid
                        if s.dim() == 2:
                            s = s.unsqueeze(0).expand(B, -1, -1)
                        sl_idx = s.to(dtype=torch.long, device=device)
                else:
                    # assume numpy arrays
                    s = sunlocked_lon_grid
                    if s.ndim == 3 and s.shape[0] == P:
                        s = s[step]
                    if s.ndim == 2:
                        s = np.repeat(s[None, ...], B, axis=0)
                    sl_idx = torch.tensor(s, dtype=torch.long, device=device)
                # For PE we still want a degrees grid that matches these indices
                lat_np, lon_np = _load_latlon()
                deg_from_idx = (sl_idx.detach().cpu().numpy().astype(np.float32) * (360.0 / float(self.n_sunlocked_heads)))

                qd_lat_np, qd_lon_np = _load_qd_for_year(2015)
                qd_lat_b = np.repeat(qd_lat_np[None, ...].astype(np.float32), B, axis=0)  # (B,H,W)
                qd_lon_b = np.repeat(qd_lon_np[None, ...].astype(np.float32), B, axis=0)  # (B,H,W)

                coord_grids = [
                    np.repeat(lat_np[None, ...], B, axis=0),
                    np.repeat(lon_np[None, ...], B, axis=0),
                    qd_lat_b,
                    qd_lon_b,
                    deg_from_idx,
                ]

            # Build one-frame input + PE
            x_in = add_fourier_positional_encoding(cur, coord_grids, n_harmonics=n_harmonics, concat_original_coords=True)  # (B,in_channels,H,W)

            # Forward pass with per-pixel head selection
            out = self.forward(x_in, sl_idx)
            task = self.tasks[0]
            if self.probabilistic:
                pred_mu, _pred_logv = out[task]     # (B,1,H,W) each
                yhat = pred_mu
            else:
                pred_mu = out[task]
                yhat = pred_mu

            # Supervise JPLD/VTEC (channel 0) at t = context + step
            y_t = batch[:, self.context_window + step, 0:1, :, :]   # (B,1,H,W)

            if self.area_weighted_loss:
                # Weighted spatial average per-sample, then batch mean
                w = self._lat_w.to(device=device)                   # (1,1,H,W)
                spatial = ((pred_mu - y_t) ** 2 * w).sum(dim=(2,3)) / w.sum()
                step_loss = spatial.mean()
            else:
                step_loss = torch.mean((pred_mu - y_t) ** 2)

            total = total + step_loss

            # AR update: next input frame = ground-truth exogenous + our predicted VTEC
            next_frame = batch[:, self.context_window + step].clone()  # (B,Ctot,H,W)
            next_frame[:, 0:1] = yhat
            cur = next_frame

        loss = total / float(P)
        if reduction == "sum":
            loss = loss * (B * P)
        return loss

    # -----------------------------
    # Proper multi-step AR predict (matches run.py)
    # -----------------------------
    def predict(
        self,
        data_context,                 # (B, Tc, C, H, W)
        sunlocked_lon_grid=None,      # (B,H,W) or (B,P,H,W) long indices  OR None if using `times`
        prediction_window: int = None,
        aux=None,
        times=None,                   # OPTIONAL: will be used to derive sun-locked/QD if provided
        n_harmonics: int = 1,
    ):
        """
        Returns predictions for VTEC channel: (B, P, 1, H, W)
        - Starts from the LAST context frame
        - Each step re-adds PE and uses per-pixel sun-locked head selection
        - At inference, exogenous channels are persisted (unless external forecasts are fed)
        """
        assert data_context.dim() == 5, "Expected (B,Tc,C,H,W)"
        B, Tc, Ctot, H, W = data_context.shape
        assert Tc >= self.context_window, "data_context too short for configured context_window"

        P = int(prediction_window) if prediction_window is not None else int(self.prediction_window)
        preds = []
        device = data_context.device

        # Seed with the last available context frame
        cur = data_context[:, Tc - 1].clone()                   # (B,Ctot,H,W)

        for step in range(P):
            step_idx = (Tc - 1) + step  # used for optional QD-year lookup if times is provided

            # If we have times, compute per-step grids and head indices; else use provided indices
            if times is not None:
                step_times = [_safe_dt(times, b, step_idx) for b in range(B)]
                coord_grids, sl_idx = _build_step_coord_grids(B, H, W, step_times, self.n_sunlocked_heads)
                sl_idx = sl_idx.to(device=device)
            else:
                if sunlocked_lon_grid is None:
                    raise ValueError("Either `times` or `sunlocked_lon_grid` must be provided.")
                if torch.is_tensor(sunlocked_lon_grid):
                    if sunlocked_lon_grid.dim() == 4:           # (B,P,H,W)
                        sl_idx = sunlocked_lon_grid[:, step].to(dtype=torch.long, device=device)
                    else:                                       # (B,H,W) or (H,W)
                        s = sunlocked_lon_grid
                        if s.dim() == 2:
                            s = s.unsqueeze(0).expand(B, -1, -1)
                        sl_idx = s.to(dtype=torch.long, device=device)
                else:
                    s = sunlocked_lon_grid
                    if s.ndim == 3 and s.shape[0] == P:
                        s = s[step]
                    if s.ndim == 2:
                        s = np.repeat(s[None, ...], B, axis=0)
                    sl_idx = torch.tensor(s, dtype=torch.long, device=device)

                # For PE, also include a degrees grid matching these indices
                lat_np, lon_np = _load_latlon()
                deg_from_idx = (sl_idx.cpu().numpy().astype(np.float32) * (360.0 / float(self.n_sunlocked_heads)))
                coord_grids = [
                    np.repeat(lat_np[None, ...], B, axis=0),
                    np.repeat(lon_np[None, ...], B, axis=0),
                    *( _load_qd_for_year(_safe_dt(times, 0, step_idx).year) if times is not None
                       else _load_qd_for_year(2015) ),  # (3) precedence-safe starred unpack
                    deg_from_idx,
                ]

            # One-frame + PE, constant channel count
            x_in = add_fourier_positional_encoding(cur, coord_grids, n_harmonics=n_harmonics, concat_original_coords=True)

            out = self.forward(x_in, sl_idx, aux=aux)
            task = self.tasks[0]
            if self.probabilistic:
                yhat = out[task][0]                           # (B,1,H,W)
            else:
                yhat = out[task]                              # (B,1,H,W)

            preds.append(yhat.unsqueeze(1))                   # (B,1,1,H,W)

            # Advance AR state: persist exogenous; replace VTEC with our prediction
            next_frame = cur.clone()
            next_frame[:, 0:1] = yhat
            cur = next_frame

        return torch.cat(preds, dim=1)                        # (B,P,1,H,W)

########################################
# 4. Latitude Band Ensemble (smooth blend)
########################################

class LatBandSFNOEnsemble(nn.Module):
    """
    Smoothly blends per-latitude-band SFNO models.
    Expects x = one-frame + PE, (B,C,H,W), and returns the same dict structure as the base model.
    """
    def __init__(self, band_edges, base_model_args):
        super().__init__()
        self.bands = band_edges
        self.models = nn.ModuleList([
            SphericalFourierNeuralOperatorModel(**base_model_args) for _ in self.bands
        ])
        self.band_centers = [(b[0] + b[1]) / 2. for b in band_edges]

    def forward(self, x, lat_grid, sunlocked_lon_grid, aux=None):
        """
        x: (B,C,H,W) one-frame + PE
        lat_grid: (H,W) or (B,H,W) degrees
        sunlocked_lon_grid: (B,H,W) long indices
        """
        B, _, H, W = x.shape
        device = x.device
        lat = torch.as_tensor(lat_grid, dtype=torch.float32, device=device)
        if lat.ndim == 2:
            lat = lat.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)

        # Prepare accumulators based on the first model's config
        m0 = self.models[0]
        prob = bool(m0.probabilistic)
        tasks = m0.tasks
        frame_ch = m0.frame_channels

        acc = {}
        for t in tasks:
            ch = frame_ch[t]
            if prob:
                acc[t] = (
                    torch.zeros(B, ch, H, W, device=device, dtype=x.dtype),  # mu
                    torch.zeros(B, ch, H, W, device=device, dtype=x.dtype),  # logvar
                )
            else:
                acc[t] = torch.zeros(B, ch, H, W, device=device, dtype=x.dtype)
        weight_sum = torch.zeros(B, H, W, device=device, dtype=x.dtype)

        # Blend
        for i, (lat_min, lat_max) in enumerate(self.bands):
            center = (lat_min + lat_max) / 2.0
            width = (lat_max - lat_min)
            band_weight = 1.0 - torch.abs(lat - center) / (max(width, 1e-6) / 2.0)  # (B,H,W)
            band_weight = torch.clamp(band_weight, min=0.0)
            mask = (lat >= lat_min) & (lat < lat_max)
            band_weight = band_weight * mask

            if band_weight.sum() == 0:
                continue

            # Optionally weight inputs; also weight outputs
            x_in = x * band_weight.unsqueeze(1)  # (B,C,H,W)
            out_i = self.models[i](x_in, sunlocked_lon_grid, aux=aux)

            for t in tasks:
                bw = band_weight.unsqueeze(1)  # (B,1,H,W)
                if prob:
                    mu_i, logv_i = out_i[t]
                    acc_mu, acc_lv = acc[t]
                    acc[t] = (acc_mu + mu_i * bw, acc_lv + logv_i * bw)
                else:
                    acc[t] = acc[t] + out_i[t] * bw

            weight_sum = weight_sum + band_weight

        # Normalize by total weights
        eps = 1e-6
        for t in tasks:
            if prob:
                mu, lv = acc[t]
                mu = mu / (weight_sum.unsqueeze(1) + eps)
                lv = lv / (weight_sum.unsqueeze(1) + eps)
                acc[t] = (mu, lv)
            else:
                acc[t] = acc[t] / (weight_sum.unsqueeze(1) + eps)

        return acc

########################################
# 5. Deep Ensemble Wrapper
########################################

class DeepEnsemble(nn.Module):
    def __init__(self, base_model_class, n_ensemble, base_model_args):
        super().__init__()
        self.models = nn.ModuleList([
            base_model_class(**base_model_args) for _ in range(n_ensemble)
        ])
    def set_mc_dropout(self, mc=True):
        for model in self.models:
            if hasattr(model, "set_mc_dropout"):
                model.set_mc_dropout(mc)
    def forward(self, *args, **kwargs):
        return [model(*args, **kwargs) for model in self.models]
    def predict(self, *args, **kwargs):
        return torch.stack([model.predict(*args, **kwargs) for model in self.models], dim=0)
