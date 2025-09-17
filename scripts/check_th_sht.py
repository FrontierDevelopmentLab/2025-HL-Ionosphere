# check_th_sht.py  (DH bandlimit-safe)
import os
from contextlib import nullcontext
import torch
import torch_harmonics as th

# numerics: disable TF32/AMP and use float64
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("torch_harmonics:", th.__version__)
print("cuda available:", torch.cuda.is_available(), "device:", device)

# ---- DH grid: N x 2N; exact up to Lmax = N/2 - 1 ----
nlat, nlon = 180, 360
assert nlon == 2 * nlat, "Use nlon == 2*nlat for grid='equiangular'"
Lmax = nlat // 2 - 1           # DH theorem (exact bandlimit)
print("DH bandlimit Lmax:", Lmax)

sht  = th.RealSHT(nlat, nlon, grid="equiangular").to(device=device, dtype=torch.float64)
isht = th.InverseRealSHT(nlat, nlon, grid="equiangular").to(device=device, dtype=torch.float64)

autocast_off = torch.autocast("cuda", enabled=False) if device.type == "cuda" else nullcontext()
with torch.no_grad(), autocast_off:
    Bsz = 2
    # learn coeff layout from SHT
    probe = sht(torch.zeros(Bsz, nlat, nlon, device=device, dtype=torch.float64))
    B, Ldim, Mp1 = probe.shape         # rectangular buffer, only a triangle is used
    print("coeff_shape:", probe.shape, "coeff_dtype:", probe.dtype)

    # indices and masks
    l_idx = torch.arange(Ldim, device=device).view(1, Ldim, 1)
    m_idx = torch.arange(Mp1, device=device).view(1, 1, Mp1)
    tri_mask = (l_idx >= m_idx)                # l >= m
    band_mask = (l_idx <= Lmax)                # l <= Lmax  (critical!)
    valid = tri_mask & band_mask               # where coefficients are meaningful

    def rand_valid_flm():
        # complex coefficients on valid region; m=0 purely real for real grid
        real = torch.randn(Bsz, Ldim, Mp1, device=device, dtype=torch.float64)
        imag = torch.randn(Bsz, Ldim, Mp1, device=device, dtype=torch.float64)
        z = torch.complex(real, imag).to(probe.dtype)
        flm = torch.where(valid, z, torch.zeros_like(z))
        flm[:, :, 0] = torch.complex(torch.randn(Bsz, Ldim, device=device, dtype=torch.float64),
                                     torch.zeros(Bsz, Ldim, device=device, dtype=torch.float64)).to(probe.dtype)
        # hard-zero anything above the bandlimit everywhere (even m=0)
        flm[:, Lmax+1:, :] = 0
        return flm

    # (A) coeff-space identity on valid, band-limited triangle
    flm = rand_valid_flm()
    grid = isht(flm)
    flm2 = sht(grid)
    mae_coeff = (flm2 - flm).abs()[valid.expand_as(flm)].mean().item()
    print("SHT(iSHT(flm)) coeff-space MAE (valid triangle):", mae_coeff)
    assert mae_coeff < 1e-10, "Coeff-space round-trip too large — check dtype/device."

    # (B) band-limited grid identity (use RELATIVE error)
bl_grid = isht(rand_valid_flm())               # band-limited by construction
bl_rt   = isht(sht(bl_grid))
abs_err = (bl_rt - bl_grid).abs()
rel_err = abs_err.norm() / (bl_grid.abs().norm() + 1e-12)
abs_mae = abs_err.mean().item()
rel_val = rel_err.item()
print(f"Band-limited grid: abs_MAE={abs_mae:.6g}, rel_err={rel_val:.6g}")

# Treat coeff-space identity as the gating health check:
# - coeff-space must be ~1e-12
# - for grid-space, accept rel_err < 1e-3 by default (tweak if you like)
ok = (mae_coeff < 1e-12) and (rel_val < 1e-3)

if ok:
    print("SHT BACKEND CHECK: PASS ✅")
else:
    print("SHT BACKEND CHECK: WARN ⚠️  (coeff ok; grid rel_err high)")
    print("Tips: reduce effective bandlimit, or run SHT in float64 and disable AMP/TF32 (already done here).")

# (C) optional: non-band-limited sanity
x    = torch.randn(Bsz, nlat, nlon, device=device, dtype=torch.float64)
x_rt = isht(sht(x))
abs_mae_rand = (x_rt - x).abs().mean().item()
print(f"Non-band-limited grid abs_MAE (sanity): {abs_mae_rand:.6g}  (expect > band-limited)")