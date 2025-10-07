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
    x,
    coord_grids,
    n_harmonics: int = 1,
    concat_original_coords: bool = False,
    concat_flags=None,
):
    """
    Seam-safe PE:
      - Use sin/cos for periodic angles (lon, QD lon, sun-locked deg) -> NO raw channel
      - Optionally keep raw for non-periodic coords (lat, QD lat)
    Args:
      x: (B,C,H,W)
      coord_grids: list of (B,H,W) or (H,W) arrays in *degrees*
      concat_flags: list[bool] same length as coord_grids; True -> also append raw angle (radians)
    """
    import numpy as _np
    B, C, H, W = x.shape
    dev = x.device
    feats = [x]

    if concat_flags is None:
        concat_flags = [concat_original_coords] * len(coord_grids)

    for grid, use_raw in zip(coord_grids, concat_flags):
        g = torch.as_tensor(grid, dtype=torch.float32, device=dev)
        if g.ndim == 2: g = g.unsqueeze(0).expand(B, -1, -1)
        elif g.ndim == 3 and g.shape[0] != B: g = g.expand(B, -1, -1)
        theta = g * (_np.pi / 180.0)  # degrees -> radians

        # Append raw only if requested (avoid seams for periodic coords!)
        if use_raw:
            feats.append(theta.unsqueeze(1))

        # Always append harmonics
        for k in range(1, int(n_harmonics) + 1):
            feats.append(torch.sin(k * theta).unsqueeze(1))
            feats.append(torch.cos(k * theta).unsqueeze(1))

    return torch.cat(feats, dim=1)


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

    If force_real_coeffs=True, we explicitly zero imaginary components
    at locations that must be real (DC/Nyquist for rFFT, m=0 for SHT) and
    optionally clamp tiny imaginary noise everywhere.
    """
    def __init__(self, in_channels, out_channels, modes_lat, modes_lon,
                 use_sht: bool = True, force_real_coeffs: bool = True):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_lat    = int(modes_lat)
        self.modes_lon    = int(modes_lon)
        self.use_sht      = bool(use_sht)
        self.force_real   = bool(force_real_coeffs)

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
            # MW equiangular N×2N bandlimit:
            Lmax_grid = self.H - 1
            Lmax = min(self.modes_lat - 1, Lmax_grid)
            Mmax = min(self.modes_lon - 1, Lmax)  # m ≤ l

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
    # -------- rFFT path --------
    def _fourier2_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correct realness handling for 2D rFFT:
        - Only the four special bins are strictly real: (0,0), (0, W/2), (H/2, 0), (H/2, W/2)
        (Nyquist bins only when dimension is even).
        - Zero-pad to full rFFT size (H, W//2+1) before irfft2.
        - Optionally clamp tiny imaginary noise everywhere.
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1)

        max_lat = min(self.weight_fft.shape[2], H)
        max_lon = min(self.weight_fft.shape[3], x_ft.shape[-1])

        x_ft_crop = x_ft[:, :, :max_lat, :max_lon]
        w = torch.view_as_complex(self.weight_fft[:, :, :max_lat, :max_lon, :])
        out_ft_crop = torch.einsum("bclm,colm->bolm", x_ft_crop, w)  # (B, Cout, max_lat, max_lon)

        # Zero-pad back to full rFFT grid size along last dim = W//2+1
        full_w = W // 2 + 1
        out_ft_full = torch.zeros(B, self.out_channels, H, full_w, dtype=torch.cfloat, device=x.device)
        out_ft_full[:, :, :max_lat, :max_lon] = out_ft_crop

        # Enforce realness ONLY at the provably real bins
        def _make_real(ix, iy):
            r = out_ft_full[:, :, ix, iy].real
            out_ft_full[:, :, ix, iy] = torch.complex(r, torch.zeros_like(r))

        _make_real(0, 0)
        if W % 2 == 0:
            _make_real(0, W // 2)
        if H % 2 == 0:
            _make_real(H // 2, 0)
            if W % 2 == 0:
                _make_real(H // 2, W // 2)

        # Optionally clamp tiny imaginary noise elsewhere (numerical stability)
        if self.force_real:
            imag = out_ft_full.imag
            out_ft_full = torch.complex(out_ft_full.real, imag.masked_fill(imag.abs() < 1e-7, 0.0))

        return torch.fft.irfft2(out_ft_full, s=(H, W), norm="ortho")


    # -------- SHT path (no in-place on views) --------
    
    def _sht_path(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        assert H == self.H and W == self.W
        dev = x.device

        # Modules & mask to device/dtype
        self._sht  = self._sht.to(device=dev,  dtype=torch.float64)
        self._isht = self._isht.to(device=dev, dtype=torch.float64)
        tri_mask   = self._mask_sub_tri.to(device=dev)
        from contextlib import nullcontext
        autocast_ctx = torch.autocast(**self._disable_amp) if dev.type == "cuda" else nullcontext()

        # --- forward SHT (float64, complex128) ---
        with autocast_ctx:
            x64 = x.to(torch.float64)
            flm_all = self._sht(x64.reshape(B * Cin, H, W))  # [B*Cin, Ldim, Mdim] c128
        flm_all = flm_all.reshape(B, Cin, self._Ldim, self._Mdim)
        flm_sub = flm_all[:, :, :self._Ls, :self._Ms]                      # [B,Cin,Ls,Ms] c128
        flm_sub = flm_sub * tri_mask.view(1,1,self._Ls,self._Ms)

        # Cast to c64; optional tiny-imag clamp (hygiene only)
        flm_c = flm_sub.to(torch.complex64)
        if self.force_real:  # interpret as: "clamp tiny imag," NOT "make all real"
            imag = flm_c.imag
            flm_c = torch.complex(flm_c.real, imag.masked_fill(imag.abs() < 1e-7, 0.0))

        # --- complex spectral mixing ---
        w_cplx = torch.view_as_complex(self.weight_sht)  # [Cin,Cout,Ls,Ms] c64
        # keep weights complex; if you *must* be conservative, you can clamp tiny imag only:
        if self.force_real:
            w_im = w_cplx.imag
            w_cplx = torch.complex(w_cplx.real, w_im.masked_fill(w_im.abs() < 1e-7, 0.0))

        out_sub = torch.einsum("bclm,colm->bolm", flm_c, w_cplx)           # [B,Cout,Ls,Ms] c64

        # Enforce m=0 purely real (this is the *only* strict realness rule)
        m0_real = out_sub[..., 0].real
        m0      = torch.complex(m0_real, torch.zeros_like(m0_real))
        if self._Ms > 1:
            out_sub = torch.cat([m0.unsqueeze(-1), out_sub[..., 1:]], dim=-1)
        else:
            out_sub = m0.unsqueeze(-1)

        # Embed kept block via slicing (faster & safer than index_put)
        out_full = torch.zeros(B, self.out_channels, self._Ldim, self._Mdim,
                            dtype=out_sub.dtype, device=dev)
        out_full[:, :, :self._Ls, :self._Ms] = out_sub * tri_mask

        # --- inverse SHT ---
        with autocast_ctx:
            y64_flat = self._isht(out_full.to(torch.complex128).reshape(B * self.out_channels,
                                                                        self._Ldim, self._Mdim))
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
    def __init__(self, width, modes_lat, modes_lon, dropout=0.0, mc_dropout=False,
                 use_sht: bool = True, force_real_coeffs: bool = True):
        super().__init__()
        self.fourier = SphericalFourierLayer(width, width, modes_lat, modes_lon,
                                             use_sht=use_sht, force_real_coeffs=force_real_coeffs)
        self.linear = nn.Conv2d(width, width, 1)
        self.dropout = MCDropout(dropout)
        self.dropout.mc = mc_dropout
        self.norm = nn.LayerNorm(width)
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
        use_sht: bool = True,                # (6) SHT hook flag
        area_weighted_loss: bool = False,     # (5) optional spherical area weighting
        sunlocked_chunk_size: int = 64,       # sunlcoked chunks
        force_real_coeffs: bool = True,
        
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
        self.force_real_coeffs = bool(force_real_coeffs)
        self.output_blur_sigma = 0.0

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
            SFNOBlock(trunk_width, modes_lat, modes_lon, dropout=dropout, mc_dropout=mc_dropout, use_sht=self.use_sht, force_real_coeffs=self.force_real_coeffs)
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

    def _circular_delta(self, a, b, n):
        # minimal signed distance on circle of size n
        d = (a - b + n / 2) % n - n / 2
        return d

    def _blur_mu(self, t):
        # depthwise 3x3 Gaussian (sigma≈0.85) on μ only; disabled if sigma==0
        if not (self.output_blur_sigma and self.output_blur_sigma > 0):
            return t
        import math, torch.nn.functional as F
        s = float(self.output_blur_sigma)
        # isotropic 3x3 gaussian kernel parametrized by sigma s
        xs = torch.tensor([-1.0, 0.0, 1.0], device=t.device, dtype=t.dtype)
        g1d = torch.exp(-0.5 * (xs / s)**2)
        g1d = g1d / g1d.sum()
        k2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        ch = t.shape[1]
        k = k2d.repeat(ch, 1, 1, 1)
        return F.conv2d(t, k, padding=1, groups=ch)


    def forward(self, x: torch.Tensor, sunlocked_lon_grid, aux=None, sunlocked_deg=None, head_blend_sigma: float = 0.5):
        """
        sunlocked_lon_grid: (B,H,W) integer indices [0..n_heads-1]
        sunlocked_deg: (B,H,W) degrees in [0,360); used for soft circular blending.
                    If None, integer indices are used as centers.
        head_blend_sigma: gaussian sigma in *head-index* units (0.5 ~ blend nearest 2-3 heads)
        """
        import torch.nn.functional as F

        # trunk
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)

        if self.aux_proj is not None and aux is not None:
            aux_emb = self.aux_proj(aux).unsqueeze(-1).unsqueeze(-1)
            x = x + aux_emb

        B, Wc, H, W = x.shape
        device, dtype = x.device, x.dtype

        # indices / centers
        if not torch.is_tensor(sunlocked_lon_grid):
            sunlocked_lon_grid = torch.tensor(sunlocked_lon_grid, device=device)
        if sunlocked_lon_grid.dim() == 2:
            sunlocked_lon_grid = sunlocked_lon_grid.unsqueeze(0).expand(B, -1, -1)
        sl_idx = sunlocked_lon_grid.to(dtype=torch.long).clamp_(0, self.n_sunlocked_heads - 1)  # (B,H,W)

        if sunlocked_deg is not None:
            if not torch.is_tensor(sunlocked_deg):
                sunlocked_deg = torch.tensor(sunlocked_deg, dtype=torch.float32, device=device)
            if sunlocked_deg.dim() == 2:
                sunlocked_deg = sunlocked_deg.unsqueeze(0).expand(B, -1, -1)
            head_center = (sunlocked_deg * (self.n_sunlocked_heads / 360.0)).to(dtype=torch.float32)  # (B,H,W)
        else:
            head_center = sl_idx.to(torch.float32)

        outs = {}
        eps = 1e-12

        for task in self.tasks:
            out_dim_per_step = self.frame_channels[task]
            out_channels = out_dim_per_step * (2 if self.probabilistic else 1)

            heads = self.heads[task]
            w_full = torch.stack([h.weight for h in heads], dim=0).to(device=device, dtype=dtype)
            w_full = w_full.view(self.n_sunlocked_heads * out_channels, Wc, 1, 1)
            b_full = (torch.cat([h.bias.to(device=device, dtype=dtype) for h in heads], dim=0)
                    if heads[0].bias is not None else torch.zeros(self.n_sunlocked_heads * out_channels, device=device, dtype=dtype))

            # accumulate weighted sum across chunks
            selected = torch.zeros(B, out_channels, H, W, device=device, dtype=dtype)
            wsum     = torch.zeros(B, 1,          H, W, device=device, dtype=dtype)

            csz = int(self.sunlocked_chunk_size)
            for h0 in range(0, self.n_sunlocked_heads, csz):
                h1 = min(self.n_sunlocked_heads, h0 + csz)
                c  = h1 - h0

                # conv for this chunk of heads -> (B, c*out_ch, H, W) -> (B, c, out_ch, H, W)
                w_chunk = w_full[h0 * out_channels : h1 * out_channels]
                b_chunk = b_full[h0 * out_channels : h1 * out_channels]
                y_chunk = F.conv2d(x, w_chunk, b_chunk).view(B, c, out_channels, H, W)

                # circular gaussian weights per-pixel for these head ids
                head_ids = torch.arange(h0, h1, device=device, dtype=torch.float32).view(1, c, 1, 1)
                d = self._circular_delta(head_ids, head_center.unsqueeze(1), self.n_sunlocked_heads)  # (B,c,H,W)
                sigma = float(max(1e-6, head_blend_sigma))
                w_chunk_pix = torch.exp(-0.5 * (d / sigma) ** 2).unsqueeze(2)  # (B,c,1,H,W)

                # weighted accumulation
                selected = selected + (y_chunk * w_chunk_pix).sum(dim=1)        # sum over heads in chunk
                wsum     = wsum     + w_chunk_pix.sum(dim=1)                    # (B,1,H,W)

            selected = selected / (wsum + eps)

            if self.probabilistic:
                mu, logvar = torch.split(selected, out_dim_per_step, dim=1)
                mu = self._blur_mu(mu)  # optional tiny smooth on μ only
                outs[task] = (mu, logvar)
            else:
                outs[task] = self._blur_mu(selected)

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
            # coord_grids = [lat, lon, qd_lat, qd_lon, sunlocked_deg]
            concat_flags = [True,  False,  True,    False,  False]  # raw only for non-periodic
            x_in = add_fourier_positional_encoding(
                cur, coord_grids, n_harmonics=n_harmonics, concat_original_coords=False, concat_flags=concat_flags
)


            # Forward pass with per-pixel head selection
            out = self.forward(x_in, sl_idx, sunlocked_deg=torch.as_tensor(coord_grids[-1], dtype=torch.float32, device=x_in.device), head_blend_sigma=0.5)
            task = self.tasks[0]

            # Supervise JPLD/VTEC (channel 0) at t = context + step
            y_t = batch[:, self.context_window + step, 0:1, :, :]   # (B,1,H,W)

            if self.probabilistic:
                # Heads output (mu, logvar)
                pred_mu, pred_logvar = out[task]                    # (B,1,H,W) each
                yhat = pred_mu                                      # use mean for AR update

                # Clamp log-variance for numerical stability
                pred_logvar = pred_logvar.clamp(
                    min=float(np.log(1e-6)),
                    max=float(np.log(1e2)),
                )

                # Per-pixel Gaussian NLL
                perpix_nll = self.gaussian_nll_from_logvar(pred_mu, pred_logvar, y_t)  # (B,1,H,W)

                if self.area_weighted_loss:
                    # Weighted spatial average per-sample, then batch mean
                    w = self._lat_w.to(device=device)                                   # (1,1,H,W)
                    spatial = (perpix_nll * w).sum(dim=(2, 3)) / w.sum()
                    step_loss = spatial.mean()
                else:
                    step_loss = perpix_nll.mean()

            else:
                # Deterministic path (fallback): MSE
                pred_mu = out[task]
                yhat = pred_mu

                if self.area_weighted_loss:
                    w = self._lat_w.to(device=device)                                   # (1,1,H,W)
                    spatial = ((pred_mu - y_t) ** 2 * w).sum(dim=(2, 3)) / w.sum()
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
            # coord_grids = [lat, lon, qd_lat, qd_lon, sunlocked_deg]
            concat_flags = [True,  False,  True,    False,  False]  # raw only for non-periodic
            x_in = add_fourier_positional_encoding(
                cur, coord_grids, n_harmonics=n_harmonics, concat_original_coords=False, concat_flags=concat_flags
            )


            out = self.forward(x_in, sl_idx, aux=aux, sunlocked_deg=torch.as_tensor(coord_grids[-1], dtype=torch.float32, device=x_in.device), head_blend_sigma=0.5)
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
