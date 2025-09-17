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
    Spectral mixing layer with two backends:
      - rFFT2 path (Cartesian equirectangular)
      - SHT path (spherical harmonic transform via torch_harmonics)

    For SHT, we enforce DH bandlimit on an equiangular grid (W = 2*H).
    Coefficients are truncated to (Ls, Ms) with triangular (ℓ >= m) mask.
    """
    def __init__(self, in_channels, out_channels, modes_lat, modes_lon, use_sht: bool = False):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_lat    = int(modes_lat)
        self.modes_lon    = int(modes_lon)
        self.use_sht      = bool(use_sht)

        # --- Discover grid now and enforce DH shape W = 2H ---
        lat_np, lon_np = _load_latlon()
        H, W = lat_np.shape
        if W != 2 * H:
            raise ValueError(f"SHT requires equiangular N×2N grid, got {H}×{W}")
        self.H, self.W = int(H), int(W)

        # --- rFFT weights (kept for rFFT backend) ---
        self.weight_fft = nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, self.modes_lat, self.modes_lon, 2) * 0.01
        )

        # --- SHT setup ---
        self._sht = None
        self._isht = None
        if self.use_sht:
            try:
                import torch_harmonics as th
            except Exception as e:
                raise RuntimeError(
                    "use_sht=True but torch_harmonics is not available. "
                    "Install with `pip install torch-harmonics` or `conda install -c conda-forge torch-harmonics`."
                ) from e

            # Modules (dtype/device set at forward)
            self._sht  = th.RealSHT(self.H, self.W, grid="equiangular")
            self._isht = th.InverseRealSHT(self.H, self.W, grid="equiangular")

            # DH exact bandlimit for N×2N grid:
            dh_Lmax = self.H // 2 - 1
            Lmax = max(0, min(dh_Lmax, self.modes_lat - 1))
            Mmax = max(0, min(Lmax,    self.modes_lon - 1))
            self._Ldim = self.H
            self._Mdim = self.H + 1
            self._Ls   = Lmax + 1
            self._Ms   = Mmax + 1

            # Triangular mask (ℓ >= m) over kept sub-block
            l_idx = torch.arange(self._Ls).view(self._Ls, 1)
            m_idx = torch.arange(self._Ms).view(1, self._Ms)
            tri_mask = (l_idx >= m_idx)  # [Ls, Ms] bool
            self.register_buffer("_mask_sub_tri", tri_mask, persistent=False)

            # Complex weights for (ℓ,m) mixing, stored as real-imag pairs (complex64)
            w = torch.zeros(self.in_channels, self.out_channels, self._Ls, self._Ms, 2, dtype=torch.float32)
            w[..., 0].normal_(mean=0.0, std=0.01)
            w[..., 1].normal_(mean=0.0, std=0.01)
            tri = self._mask_sub_tri.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,Ls,Ms,1]
            w = w * tri                                                        # zero invalid m > ℓ
            self.weight_sht = nn.Parameter(w)

        # Disable AMP/TF32 only around SHT transforms (safe elsewhere)
        self._disable_amp = dict(device_type="cuda", enabled=False)

    # -------- rFFT path --------
    def _fourier2_path(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        max_lat = min(self.weight_fft.shape[2], H // 2)
        max_lon = min(self.weight_fft.shape[3], x_ft.shape[-1])
        x_ft = x_ft[:, :, :max_lat, :max_lon]
        w    = torch.view_as_complex(self.weight_fft[:, :, :max_lat, :max_lon, :])
        # (B,Cin,L,M) @ (Cin,Cout,L,M) -> (B,Cout,L,M)
        out_ft = torch.einsum("bclm,colm->bolm", x_ft, w)
        full_w = x_ft.shape[-1]
        out_ft_full = torch.zeros(B, self.out_channels, H, full_w, dtype=torch.cfloat, device=x.device)
        out_ft_full[:, :, :max_lat, :max_lon] = out_ft
        return torch.fft.irfft2(out_ft_full, s=(H, W), norm="ortho")

    # -------- SHT path (no in-place on views) --------
    def _sht_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Cin, H, W] (real). Returns [B, Cout, H, W] (real).
        SHT/ISHT in float64; mixing in complex64 to save memory.
        """
        B, Cin, H, W = x.shape
        assert H == self.H and W == self.W, f"Expected grid {self.H}x{self.W}, got {H}x{W}"
        dev = x.device

        # Move modules & mask
        self._sht  = self._sht.to(device=dev,  dtype=torch.float64)
        self._isht = self._isht.to(device=dev, dtype=torch.float64)
        tri_mask   = self._mask_sub_tri.to(device=dev)

        # SHT prefers float64; disable AMP/TF32 just for SHT/ISHT
        from contextlib import nullcontext
        autocast_ctx = torch.autocast(**self._disable_amp) if dev.type == "cuda" else nullcontext()

        # --- forward SHT ---
        with autocast_ctx:
            x64 = x.to(torch.float64)
            x_flat = x64.reshape(B * Cin, H, W).contiguous()
            flm_all = self._sht(x_flat)  # [B*Cin, Ldim(=H), Mdim(=H+1)] complex128

        flm_all = flm_all.reshape(B, Cin, self._Ldim, self._Mdim)
        flm_sub = flm_all[:, :, :self._Ls, :self._Ms]                         # [B,Cin,Ls,Ms] c128
        flm_sub = flm_sub * tri_mask.view(1, 1, self._Ls, self._Ms)           # mask invalid m>ℓ
        flm_c   = flm_sub.to(torch.complex64)

        # --- spectral mixing (complex) ---
        w_cplx  = torch.view_as_complex(self.weight_sht)                      # [Cin,Cout,Ls,Ms] c64
        out_sub = torch.einsum("bclm,colm->bolm", flm_c, w_cplx)              # [B,Cout,Ls,Ms] c64

        # Enforce m=0 purely real WITHOUT in-place on a view:
        m0_real = out_sub[..., 0].real                                        # [B,Cout,Ls]
        m0      = torch.complex(m0_real, torch.zeros_like(m0_real))           # [B,Cout,Ls] c64
        if self._Ms > 1:
            out_sub = torch.cat([m0.unsqueeze(-1), out_sub[..., 1:]], dim=-1) # new tensor
        else:
            out_sub = m0.unsqueeze(-1)

        # Embed into full coefficient grid (no in-place on a view of a leaf tensor)
        out_full = torch.zeros(B, self.out_channels, self._Ldim, self._Mdim,
                               dtype=out_sub.dtype, device=dev)
        out_full = out_full.index_put(
            (torch.arange(B, device=dev).view(-1,1,1,1),
             torch.arange(self.out_channels, device=dev).view(1,-1,1,1),
             torch.arange(self._Ls, device=dev).view(1,1,-1,1),
             torch.arange(self._Ms, device=dev).view(1,1,1,-1)),
            out_sub * tri_mask.view(1,1,self._Ls,self._Ms),
            accumulate=False
        )

        # --- inverse SHT ---
        with autocast_ctx:
            out_full_c128 = out_full.to(torch.complex128)
            y64_flat = self._isht(out_full_c128.reshape(B * self.out_channels, self._Ldim, self._Mdim))
        y = y64_flat.reshape(B, self.out_channels, H, W).to(dtype=x.dtype)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sht_path(x) if self.use_sht else self._fourier2_path(x)



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
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out



########################################
# 2. SFNO Model (last-frame + PE input; proper AR rollout)
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
        sunlocked_chunk_size: int = 64,       # sunlcoked chunks
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
        self.sunlocked_chunk_size = int(max(1, min(n_sunlocked_heads, sunlocked_chunk_size)))

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
        sunlocked_lon_grid: (B,H,W) long indices in [0, n_sunlocked_heads-1]
        Vectorized sun-locked head selection:
        - compute 1×1 conv for many heads at once (chunked),
        - gather the correct head per pixel,
        - no Python loops over heads or batch.
        """
        import torch.nn.functional as F

        # trunk
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)

        # optional aux
        if self.aux_proj is not None and aux is not None:
            aux_emb = self.aux_proj(aux).unsqueeze(-1).unsqueeze(-1)  # (B,Wc,1,1)
            x = x + aux_emb

        B, Wc, H, W = x.shape
        device, dtype = x.device, x.dtype

        # normalize/expand indices
        if not torch.is_tensor(sunlocked_lon_grid):
            sunlocked_lon_grid = torch.tensor(sunlocked_lon_grid, device=device)
        if sunlocked_lon_grid.dim() == 2:
            sunlocked_lon_grid = sunlocked_lon_grid.unsqueeze(0).expand(B, -1, -1)
        sl_idx = sunlocked_lon_grid.to(dtype=torch.long).clamp_(0, self.n_sunlocked_heads - 1)  # (B,H,W)

        outs = {}
        for task in self.tasks:
            out_dim_per_step = self.frame_channels[task]
            out_channels = out_dim_per_step * (2 if self.probabilistic else 1)

            # ---- stack all head params for this task (keeps grads) ----
            heads = self.heads[task]  # ModuleList of Conv2d (1×1)
            # weights: (n_heads, out_ch, in_ch, 1, 1) -> (n_heads*out_ch, in_ch, 1,1)
            w_full = torch.stack([h.weight for h in heads], dim=0).to(device=device, dtype=dtype)
            w_full = w_full.view(self.n_sunlocked_heads * out_channels, Wc, 1, 1)
            # bias: (n_heads*out_ch,)
            if heads[0].bias is None:
                b_full = torch.zeros(self.n_sunlocked_heads * out_channels, device=device, dtype=dtype)
            else:
                b_full = torch.cat([h.bias.to(device=device, dtype=dtype) for h in heads], dim=0)

            # output buffer
            selected = torch.zeros(B, out_channels, H, W, device=device, dtype=dtype)

            # chunk over heads to cap memory
            csz = int(self.sunlocked_chunk_size)
            for h0 in range(0, self.n_sunlocked_heads, csz):
                h1 = min(self.n_sunlocked_heads, h0 + csz)
                c = h1 - h0  # heads in this chunk

                # slice weights/bias for this chunk and convolve once
                w_chunk = w_full[h0 * out_channels : h1 * out_channels]                 # (c*out_ch, Wc, 1, 1)
                b_chunk = b_full[h0 * out_channels : h1 * out_channels]                 # (c*out_ch,)
                y_chunk = F.conv2d(x, w_chunk, b_chunk)                                 # (B, c*out_ch, H, W)
                y_chunk = y_chunk.view(B, c, out_channels, H, W)                        # (B, c, out_ch, H, W)

                # for this chunk, where do indices land?
                idx_rel = sl_idx - h0                                                   # (B,H,W), in [-, c-1]
                mask = (idx_rel >= 0) & (idx_rel < c)                                   # (B,H,W) for positions owned by this chunk
                if not mask.any().item():
                    continue

                # gather the chosen head within this chunk
                # index shape must match input except for dim=1 (head dim)
                idx_rel_clamped = idx_rel.clamp_(0, c - 1)
                idx_g = idx_rel_clamped.unsqueeze(1).unsqueeze(2).expand(B, 1, out_channels, H, W)
                picked = y_chunk.gather(dim=1, index=idx_g).squeeze(1)                  # (B, out_ch, H, W)

                # write only where this chunk owns the pixel
                m = mask.unsqueeze(1)                                                   # (B,1,H,W) -> broadcast over channels
                selected = torch.where(m, picked, selected)

            # package result
            if self.probabilistic:
                mu, logvar = torch.split(selected, out_dim_per_step, dim=1)
                outs[task] = (mu, logvar)
            else:
                outs[task] = selected

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
# 3. Latitude Band Ensemble (smooth blend)
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

            if band_weight.sum().item() == 0:
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
# 4. Deep Ensemble Wrapper
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
