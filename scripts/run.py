import argparse, datetime, pprint, os, sys, glob, shutil, random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import contextlib
from torch import amp as torch_amp

# --- GPU fast-math + autotune (A100) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')            # set backend BEFORE importing pyplot
import matplotlib.pyplot as plt


# --- extra deps for SFNO & helpers ---

# Optional deps used by SFNO helpers (safe to miss)
try:
    from skyfield.api import load as sf_load, wgs84
    _SF_TS = sf_load.timescale()
    _SF_EPH = sf_load('de421.bsp')
    _SF_EARTH = _SF_EPH['earth']
    _SF_SUN = _SF_EPH['sun']
    def _get_subsolar_longitude(dt):
        t = _SF_TS.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        subsolar = wgs84.subpoint(_SF_EARTH.at(t).observe(_SF_SUN))
        return subsolar.longitude.degrees
except Exception:
    def _get_subsolar_longitude(dt):
        # Fallback approximation if skyfield isn't available
        return (dt.hour / 24.0) * 360.0 - 180.0

# Optional advanced model + physics-informed loss
try:
    # (spelling is intentional in repo) -> class name is SphericalFourierNeuralOperatorModel
    from model_sperhicalFNO_Advanced import SphericalFourierNeuralOperatorModel
except Exception:
    SphericalFourierNeuralOperatorModel = None

try:
    from physics_losses import geophysics_informed_loss
except Exception:
    geophysics_informed_loss = None


from util import Tee
try:
    import wandb
except ImportError:
    wandb = None    

from util import set_random_seed
from util import md5_hash_str


# --- Sweep helpers ------------------------------------------------------------
SWEEP_OVERRIDABLE = {
    # training knobs
    "learning_rate": float,
    "weight_decay": float,
    "batch_size": int,
    "epochs": int,
    "dropout": float,
    "context_window": int,
    "prediction_window": int,
    "date_dilation": int,

    # SFNO knobs (already read with getattr(...) later)
    "sfno_width": int,
    "sfno_depth": int,
    "sfno_modes_lat": int,
    "sfno_modes_lon": int,
    "n_sunlocked_heads": int,
    "n_harmonics": int,
    "spectral_backend": str,
    "area_weighted_loss": bool,

    # W&B quality-of-life
    "wandb_mode": str,            # enforce "online" in sweeps
    "wandb_project": str,         # keep consistent
    "aux_dim": int,
    "mc_dropout": bool,
}

def _apply_wandb_config_overrides(args):
    """If running under a W&B Sweep, copy wandb.config values into argparse args."""
    try:
        import wandb as _wandb  # use same module
    except Exception:
        return args

    if getattr(_wandb, "run", None) is None:
        return args

    cfg = dict(_wandb.config)  # FrozenOrderedDict -> plain dict
    for k, caster in SWEEP_OVERRIDABLE.items():
        if k in cfg and hasattr(args, k):
            try:
                setattr(args, k, caster(cfg[k]))
            except Exception:
                setattr(args, k, cfg[k])
    return args


def _materialize_target_dir_placeholders(target_dir: str):
    """
    Allow --target_dir like './sweeps/{wandb_run_id}' to fan out per run.
    Safe when not in W&B.
    """
    try:
        import wandb as _wandb
        run_id  = getattr(getattr(_wandb, "run", None), "id", None)
        run_name = getattr(getattr(_wandb, "run", None), "name", None)
    except Exception:
        run_id = run_name = None
    if run_id:
        return target_dir.format(wandb_run_id=run_id, wandb_run_name=(run_name or "unnamed"))
    return target_dir
# ------------------------------------------------------------------------------



# from model_vae import VAE1
from model_convlstm import IonCastConvLSTM
from model_lstm import IonCastLSTM
from model_linear import IonCastLinear
from model_persistence import IonCastPersistence
from model_lstmsdo import IonCastLSTMSDO
from dataset_jpld import JPLD
from dataset_sequences import Sequences
from dataset_union import Union
from dataset_sunmoongeometry import SunMoonGeometry
from dataset_celestrak import CelesTrak
from dataset_omniweb import OMNIWeb
from dataset_set import SET
from dataset_sdocore import SDOCore
from dataloader_cached import CachedDataLoader
from events import EventCatalog, validation_events_1, validation_events_2, validation_events_3, validation_events_4
from eval import eval_forecast_long_horizon, save_metrics, eval_forecast_fixed_lead_time, aggregate_and_plot_fixed_lead_time_metrics

# --- Optional Spherical FNO (SFNO) support -----------------------------------
try:
    # file name is intentionally "model_sperhicalFNO_Advanced.py" in this repo
    from model_sperhicalFNO_Advanced import SphericalFourierNeuralOperatorModel as _SFNO
except Exception:
    _SFNO = None

import numpy as _np
import torch as _torch

# Lazy caches for lat/lon grids (npz or npy files expected in CWD)
_lat_grid_np = None
_lon_grid_np = None

def _load_latlon():
    global _lat_grid_np, _lon_grid_np
    if _lat_grid_np is None or _lon_grid_np is None:
        _lat_grid_np = _np.load('lat_grid.npy')   # shape (180, 360)
        _lon_grid_np = _np.load('lon_grid.npy')   # shape (180, 360)
    return _lat_grid_np, _lon_grid_np

def _add_fourier_pe(x_last, lat_np, lon_np, n_harmonics: int):
    """
    x_last: (B, C, H, W) float tensor on device
    lat_np/lon_np: numpy arrays (H, W) with degrees
    returns: (B, C + pe_channels, H, W)
    """
    B, C, H, W = x_last.shape
    device = x_last.device
    lat = _torch.tensor(lat_np, dtype=_torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
    lon = _torch.tensor(lon_np, dtype=_torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
    # normalize each grid to [-pi, pi]
    def _to_angle(g):
        gmin = g.amin(dim=(1, 2), keepdim=True)
        gmax = g.amax(dim=(1, 2), keepdim=True)
        g = (g - gmin) / (gmax - gmin + 1e-8)
        return g * (2 * _np.pi) - _np.pi
    lat_a = _to_angle(lat)
    lon_a = _to_angle(lon)

    feats = [x_last]
    # optionally include the raw angles
    feats.append(lat_a.unsqueeze(1))
    feats.append(lon_a.unsqueeze(1))
    # harmonics
    for k in range(1, max(1, int(n_harmonics)) + 1):
        feats.append(_torch.sin(k * lat_a).unsqueeze(1))
        feats.append(_torch.cos(k * lat_a).unsqueeze(1))
        feats.append(_torch.sin(k * lon_a).unsqueeze(1))
        feats.append(_torch.cos(k * lon_a).unsqueeze(1))
    return _torch.cat(feats, dim=1)

class _SFNOAdapter(_torch.nn.Module):
    """
    Thin wrapper so SFNO plugs into run.py with the same .loss() API as IonCast*.
    - Uses the last context frame (+ simple Fourier positional encodings) as input
    - Predicts the next JPLD frame (1-step) and trains with MSE
    - Keeps the rest of the main-branch training/validation/eval code unchanged
    """
    def __init__(self, base_cfg: dict, context_window: int, prediction_window: int, n_harmonics: int):
        super().__init__()
        if _SFNO is None:
            raise ImportError("SphericalFourierNeuralOperatorModel not available in this environment.")
        self.base_cfg = dict(base_cfg)
        self.sfno = _SFNO(**self.base_cfg)
        self.context_window = int(context_window)
        self.prediction_window = int(prediction_window)
        self.n_harmonics = int(n_harmonics)
        self.name = 'SphericalFourierNeuralOperatorModel'  # run.py uses this

    def forward(self, x_input):
        # Some SFNO variants accept a second arg (e.g., sun-locked grid).
        try:
            B, C, H, W = x_input.shape
            sunlocked_zero = _torch.zeros((B, H, W), dtype=_torch.long, device=x_input.device)
            return self.sfno(x_input, sunlocked_zero)
        except TypeError:
            return self.sfno(x_input)

    def loss(self, combined_seq, jpld_weight: float = 1.0):
        """
        combined_seq: (B, T, C_total, H, W) where channel 0 is JPLD target
        Trains 1-step ahead: target = t=context_window (next frame).
        Returns: (loss_tensor, rmse_tensor, jpld_rmse_tensor)
        """
        assert combined_seq.dim() == 5, "Expected (B,T,C,H,W)"
        B, T, Ctot, H, W = combined_seq.shape
        assert T >= self.context_window + 1, "Need at least context_window + 1 frames"
        # Inputs
        x_last = combined_seq[:, self.context_window - 1, :, :, :]           # (B, C, H, W)
        lat_np, lon_np = _load_latlon()
        x_in = _add_fourier_pe(x_last, lat_np, lon_np, self.n_harmonics)      # (B, C+PE, H, W)
        # Prediction
        out = self.forward(x_in)
        if isinstance(out, dict) and 'vtec' in out:
            pred = out['vtec'][0] if isinstance(out['vtec'], (tuple, list)) else out['vtec']
        else:
            pred = out
        # Target = next JPLD frame (assumes JPLD is channel 0 in combined tensor)
        y = combined_seq[:, self.context_window, 0:1, :, :]
        mse = _torch.mean((pred - y) ** 2)
        rmse = mse.sqrt()
        return mse, rmse, rmse  # (loss, rmse (all), jpld_rmse)
# ---------------------------------------------------------------------------


event_catalog = EventCatalog(events_csv_file_name='../data/events.csv')



def save_model(model, optimizer, scheduler, epoch, iteration, train_losses, valid_losses,
               train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses,
               best_valid_rmse, file_name):
    print('Saving model to {}'.format(file_name))
    # Normalize model.name for downstream checks
    if not hasattr(model, 'name'):
        try:
            model.name = model.__class__.__name__
        except Exception:
            model.name = 'Model'

    if isinstance(model, IonCastConvLSTM):
        checkpoint = {
            'model': 'IonCastConvLSTM',
            'epoch': epoch, 'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses, 'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses, 'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels, 'model_output_channels': model.output_channels,
            'model_hidden_dim': model.hidden_dim, 'model_num_layers': model.num_layers,
            'model_context_window': model.context_window, 'model_prediction_window': model.prediction_window,
            'model_dropout': model.dropout, 'model_name': model.name
        }
    elif isinstance(model, IonCastLSTM):
        checkpoint = {
            'model': 'IonCastLSTM',
            'epoch': epoch, 'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses, 'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses, 'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels, 'model_output_channels': model.output_channels,
            'model_base_channels': model.base_channels, 'model_lstm_dim': model.lstm_dim,
            'model_num_layers': model.num_layers, 'model_context_window': model.context_window,
            'model_dropout': model.dropout, 'model_name': model.name
        }
    elif isinstance(model, IonCastLinear):
        checkpoint = {
            'model': 'IonCastLinear',
            'epoch': epoch, 'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses, 'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses, 'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels, 'model_output_channels': model.output_channels,
            'model_context_window': model.context_window, 'model_name': model.name
        }
    elif isinstance(model, IonCastPersistence):
        checkpoint = {
            'model': 'IonCastPersistence',
            'epoch': epoch, 'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses, 'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses, 'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels, 'model_output_channels': model.output_channels,
            'model_context_window': model.context_window, 'model_name': model.name
        }
    elif isinstance(model, IonCastLSTMSDO):
        checkpoint = {
            'model': 'IonCastLSTMSDO',
            'epoch': epoch, 'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses, 'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses, 'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels, 'model_output_channels': model.output_channels,
            'model_base_channels': model.base_channels, 'model_lstm_dim': model.lstm_dim,
            'model_num_layers': model.num_layers, 'model_context_window': model.context_window,
            'model_dropout': model.dropout, 'model_sdo_dim': model.sdo_dim,
            'model_sdo_num_layers': model.sdo_num_layers, 'model_name': model.name
        }
    elif (SphericalFourierNeuralOperatorModel is not None) and isinstance(model, SphericalFourierNeuralOperatorModel):
        # Be defensive about attributes; fall back if the SFNO implementation differs
        in_proj = getattr(model, 'in_proj', None)
        trunk_width = getattr(in_proj, 'out_channels', None) if in_proj is not None else None
        in_channels = getattr(in_proj, 'in_channels', None) if in_proj is not None else None
        blocks = getattr(model, 'blocks', [])
        modes_lat = modes_lon = None
        if blocks:
            fourier = getattr(blocks[0], 'fourier', None)
            modes_lat = modes_lon = None
            if fourier is not None:
                wf = getattr(fourier, 'weight_fft', None)
                ws = getattr(fourier, 'weight_sht', None)
                if wf is not None: modes_lat, modes_lon = int(wf.shape[2]), int(wf.shape[3])
                elif ws is not None: modes_lat, modes_lon = int(ws.shape[2]), int(ws.shape[3])

        checkpoint = {
            'model': 'SphericalFourierNeuralOperatorModel',
            'epoch': epoch, 'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses, 'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses, 'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses, 'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_in_channels': in_channels,
            'model_trunk_width': trunk_width,
            'model_trunk_depth': len(blocks),
            'model_modes_lat': modes_lat,
            'model_modes_lon': modes_lon,
            'model_aux_dim': getattr(model, 'aux_dim', 0),
            'model_tasks': getattr(model, 'tasks', ("vtec",)),
            'model_out_shapes': getattr(model, 'out_shapes', {"vtec": (1, "grid")}),
            'model_probabilistic': getattr(model, 'probabilistic', False),
            'model_dropout': getattr(model, 'dropout', 0.0),
            'model_mc_dropout': getattr(model, 'mc_dropout', False),
            'model_context_window': getattr(model, 'context_window', None),
            'model_prediction_window': getattr(model, 'prediction_window', None),
            'model_use_sht': getattr(model, 'use_sht', True),
            'model_force_real_coeffs': getattr(model, 'force_real_coeffs', True),
            'model_n_sunlocked_heads': getattr(model, 'n_sunlocked_heads', 360),
            'model_area_weighted_loss': getattr(model, 'area_weighted_loss', False),
            'model_head_smooth_reg': getattr(model, 'head_smooth_reg', 0.0),
            'model_lon_tv_reg': getattr(model, 'lon_tv_reg', 0.0),
            'model_lon_highfreq_reg': getattr(model, 'lon_highfreq_reg', 0.0),
            'model_lon_highfreq_kmin': getattr(model, 'lon_highfreq_kmin', 72),
            'model_head_blend_sigma': getattr(model, 'head_blend_sigma', None),
            'model_lon_blur_sigma_deg': getattr(model, "lon_blur_sigma_deg", 5.0),


            'model_name': 'SphericalFourierNeuralOperatorModel'
        }
    else:
        raise ValueError('Unknown model type: {}'.format(model))
    torch.save(checkpoint, file_name)

def _move_batch_to_device(batch, device, channels_last=False):
    def _cast(t):
        if not torch.is_tensor(t):
            return t
        t = t.to(device, non_blocking=True)
        if channels_last and t.dim() == 4:
            t = t.contiguous(memory_format=torch.channels_last)
        return t
    if isinstance(batch, (list, tuple)):
        return type(batch)(_cast(x) for x in batch)
    if isinstance(batch, dict):
        return {k: _cast(v) for k, v in batch.items()}
    return _cast(batch)


# ================= SFNO helper utilities =================

def _safe_time_at(times_obj, b, idx):
    """
    Robustly fetch a datetime for sample b at sequence idx from a batch 'times' container.
    Handles shapes like [B][T], [T][B], [B], [T], tensors/ndarrays, and even stringified lists.
    """
    import datetime as _dt
    try:
        import numpy as _np, torch as _torch
    except Exception:
        _np = None; _torch = None

    def _to_dt(x):
        # already a datetime
        if isinstance(x, _dt.datetime):
            return x

        # bytes -> str
        if isinstance(x, (bytes, bytearray)):
            x = x.decode("utf-8")

        # tensors / ndarrays
        if (_torch is not None and isinstance(x, _torch.Tensor)) or (_np is not None and isinstance(x, _np.ndarray)):
            # scalar
            if getattr(x, "ndim", 0) == 0:
                return _to_dt(x.item())
            # fall back to first element
            return _to_dt(x.tolist())

        # lists / tuples -> prefer idx if available, else first
        if isinstance(x, (list, tuple)):
            if not x:
                raise ValueError("empty time container")
            j = 0 if not isinstance(idx, int) else min(idx, len(x) - 1)
            return _to_dt(x[j])

        # strings (possibly a stringified list)
        s = str(x)
        if s.startswith("[") and s.endswith("]"):
            # pick the first item inside the list representation
            inner = s[1:-1].strip()
            if inner:
                first = inner.split(",")[0].strip().strip("'").strip('"')
                return _dt.datetime.fromisoformat(first)
        return _dt.datetime.fromisoformat(s)

    # Try common container layouts in order of likelihood
    try:
        return _to_dt(times_obj[b][min(idx, len(times_obj[b]) - 1)])
    except Exception:
        pass
    try:
        return _to_dt(times_obj[min(idx, len(times_obj) - 1)][b])
    except Exception:
        pass
    try:
        return _to_dt(times_obj[b])
    except Exception:
        pass
    return _to_dt(times_obj)


def ensure_grid(tensor, target_channels, H=180, W=360):
    while tensor.dim() > 3 and tensor.shape[-1] == 1 and tensor.shape[-2] == 1:
        tensor = tensor.squeeze(-1).squeeze(-1)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(-1).unsqueeze(-1)

    C = tensor.shape[2]
    if C < target_channels:
        pad = torch.zeros(tensor.shape[0], tensor.shape[1],
                          target_channels - C, tensor.shape[-2], tensor.shape[-1],
                          dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=2)
    elif C > target_channels:
        tensor = tensor[:, :, :target_channels, :, :]

    # expand H/W if they’re singleton
    if tensor.shape[-2] == 1 and H != 1: tensor = tensor.expand(-1, -1, -1, H, tensor.shape[-1])
    if tensor.shape[-1] == 1 and W != 1: tensor = tensor.expand(-1, -1, -1, -1, W)
    return tensor

# Grids and positional encodings
_lat_grid = _lon_grid = None

def _get_latlon_grids():
    global _lat_grid, _lon_grid
    if _lat_grid is None or _lon_grid is None:
        _lat_grid = np.load('lat_grid.npy')   # (180, 360)
        _lon_grid = np.load('lon_grid.npy')
    return _lat_grid, _lon_grid

def _load_qd_grid_for_year(year):
    """Return QD grids if available, else gracefully fall back to geographic lat/lon."""
    try:
        qd_lat = _np.load(f'qd_lat_{year}.npy').astype(_np.float32)
        qd_lon = _np.load(f'qd_lon_{year}.npy').astype(_np.float32)
        return qd_lat, qd_lon
    except Exception:
        # Fallback to geographic if QD files are missing
        return _lat_grid.astype(_np.float32), _lon_grid.astype(_np.float32)
    
def _quantize_degrees_to_heads(deg_grids: _np.ndarray, n_heads: int) -> _np.ndarray:
    """
    Map sun-locked longitude degrees in [0,360) to head indices [0, n_heads-1].
    Works for any n_heads (not only 360).
    """
    mul = float(n_heads) / 360.0
    idx = _np.floor(deg_grids * mul).astype(_np.int64)
    return _np.clip(idx, 0, n_heads - 1)



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



def _rmse_tecu(pred, target):
    # Unnormalize via JPLD static method if available; else raw RMSE
    try:
        pred_u = JPLD.unnormalize(pred)
        targ_u = JPLD.unnormalize(target)
        return torch.sqrt(torch.mean((pred_u - targ_u) ** 2))
    except Exception:
        return torch.sqrt(torch.mean((pred - target) ** 2))

# Physics-informed loss bridge (works whether physics_losses is present or not)
def _sfno_weighted_loss(pred, logvar, y_jpld, iteration,
                        jpld_weight=1.0, aux_weight=1.0, aux_target=None,
                        times=None, slots_per_day=None, delta_minutes=15, tol_slots=1):
    if geophysics_informed_loss is None:
        base = torch.mean((pred[:, 0:1] - y_jpld) ** 2)
        return base, base, torch.zeros((), device=pred.device), {'data': float(base.detach())}

    if pred.dim() == 3: pred = pred.unsqueeze(1)
    if logvar is not None and logvar.dim() == 3: logvar = logvar.unsqueeze(1)

    C = pred.shape[1]
    jpld_pred = pred[:, 0:1]
    jpld_logv = logvar[:, 0:1] if logvar is not None else None
    aux_pred  = pred[:, 1:] if C > 1 else None
    aux_logv  = logvar[:, 1:] if (logvar is not None and C > 1) else None

    if slots_per_day is None:
        slots_per_day = max(1, int(round(1440.0 / float(delta_minutes))))

    def _call_loss(p_, lv_, t_):
        try:
            return geophysics_informed_loss(p_, lv_, t_, iteration,
                                            times=times, slots_per_day=slots_per_day, tol_slots=tol_slots)
        except TypeError:
            return geophysics_informed_loss(p_, lv_, t_, iteration)

    jpld_loss = _call_loss(jpld_pred, jpld_logv, y_jpld)
    aux_loss = torch.zeros((), device=pred.device)
    if aux_pred is not None and aux_pred.shape[1] > 0 and aux_target is not None:
        aux_loss = _call_loss(aux_pred, aux_logv, aux_target)

    total = jpld_weight * jpld_loss + aux_weight * aux_loss
    return total, jpld_loss, aux_loss, {'data': float(jpld_loss.detach()), 'aux': float(aux_loss.detach())}


def load_model(file_name, device):
    checkpoint = torch.load(file_name, map_location=device)
    if checkpoint['model'] == 'IonCastConvLSTM':
        model = IonCastConvLSTM(
            input_channels=checkpoint['model_input_channels'],
            output_channels=checkpoint['model_output_channels'],
            hidden_dim=checkpoint.get('model_hidden_dim', 64),
            num_layers=checkpoint.get('model_num_layers', 1),
            context_window=checkpoint['model_context_window'],
            prediction_window=checkpoint['model_prediction_window'],
            dropout=checkpoint.get('model_dropout', 0.0),
            name=checkpoint.get('model_name', 'IonCastConvLSTM')
        )
    elif checkpoint['model'] == 'IonCastLSTM':
        model = IonCastLSTM(
            input_channels=checkpoint['model_input_channels'],
            output_channels=checkpoint['model_output_channels'],
            base_channels=checkpoint['model_base_channels'],
            lstm_dim=checkpoint['model_lstm_dim'],
            num_layers=checkpoint['model_num_layers'],
            context_window=checkpoint['model_context_window'],
            dropout=checkpoint['model_dropout'],
            name=checkpoint.get('model_name', 'IonCastLSTM')
        )
    elif checkpoint['model'] == 'IonCastLinear':
        model = IonCastLinear(
            input_channels=checkpoint['model_input_channels'],
            output_channels=checkpoint['model_output_channels'],
            context_window=checkpoint['model_context_window'],
            name=checkpoint.get('model_name', 'IonCastLinear')
        )
    elif checkpoint['model'] == 'IonCastPersistence':
        model = IonCastPersistence(
            input_channels=checkpoint['model_input_channels'],
            output_channels=checkpoint['model_output_channels'],
            context_window=checkpoint['model_context_window'],
            name=checkpoint.get('model_name', 'IonCastPersistence')
        )
    elif checkpoint['model'] == 'IonCastLSTMSDO':
        model = IonCastLSTMSDO(
            input_channels=checkpoint['model_input_channels'],
            output_channels=checkpoint['model_output_channels'],
            base_channels=checkpoint['model_base_channels'],
            lstm_dim=checkpoint['model_lstm_dim'],
            num_layers=checkpoint['model_num_layers'],
            context_window=checkpoint['model_context_window'],
            dropout=checkpoint['model_dropout'],
            sdo_dim=checkpoint['model_sdo_dim'],
            sdo_num_layers=checkpoint['model_sdo_num_layers'],
            name=checkpoint.get('model_name', 'IonCastLSTMSDO')
        )
    elif (SphericalFourierNeuralOperatorModel is not None) and checkpoint['model'] == 'SphericalFourierNeuralOperatorModel':
        model = SphericalFourierNeuralOperatorModel(
            in_channels=checkpoint.get('model_in_channels', 64),
            trunk_width=checkpoint.get('model_trunk_width', 64),
            trunk_depth=checkpoint.get('model_trunk_depth', 8),
            modes_lat=checkpoint.get('model_modes_lat', 32),
            modes_lon=checkpoint.get('model_modes_lon', 64),
            aux_dim=checkpoint.get('model_aux_dim', 0),
            tasks=tuple(checkpoint.get('model_tasks', ("vtec",))),
            out_shapes=checkpoint.get('model_out_shapes', {"vtec": (1, "grid")}),
            probabilistic=checkpoint.get('model_probabilistic', False),
            dropout=checkpoint.get('model_dropout', 0.0),
            mc_dropout=checkpoint.get('model_mc_dropout', False),
            # windows (fallbacks keep old checkpoints working)
            context_window=checkpoint.get('model_context_window', 4),
            prediction_window=checkpoint.get('model_prediction_window', 1),
            use_sht=checkpoint.get('model_use_sht', True),
            force_real_coeffs=checkpoint.get('model_force_real_coeffs', True),
            n_sunlocked_heads=checkpoint.get('model_n_sunlocked_heads', 360),
            area_weighted_loss=checkpoint.get('model_area_weighted_loss', False),
            head_smooth_reg=checkpoint.get('model_head_smooth_reg', 0.0),
            lon_tv_reg=checkpoint.get('model_lon_tv_reg', 0.0),
            lon_highfreq_reg=checkpoint.get('model_lon_highfreq_reg', 0.0),
            lon_highfreq_kmin=checkpoint.get('model_lon_highfreq_kmin', 72), 
            lon_blur_sigma_deg=checkpoint.get("model_lon_blur_sigma_deg", 0.0),

        )
        if 'model_head_blend_sigma' in checkpoint and checkpoint['model_head_blend_sigma'] is not None:
            model.head_blend_sigma = checkpoint['model_head_blend_sigma']
        model.name = 'SphericalFourierNeuralOperatorModel'

    else:
        raise ValueError('Unknown model type: {}'.format(checkpoint['model']))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)


    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    scheduler_state_dict = checkpoint['scheduler_state_dict']
    train_rmse_losses = checkpoint.get('train_rmse_losses', [])
    valid_rmse_losses = checkpoint.get('valid_rmse_losses', [])
    train_jpld_rmse_losses = checkpoint.get('train_jpld_rmse_losses', [])
    valid_jpld_rmse_losses = checkpoint.get('valid_jpld_rmse_losses', [])
    best_valid_rmse = checkpoint.get('best_valid_rmse', float('inf'))

    return model, optimizer, epoch, iteration, train_losses, valid_losses, scheduler_state_dict, train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses, best_valid_rmse



def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--jpld_dir', type=str, default='jpld/webdataset', help='JPLD GIM dataset directory')
    parser.add_argument('--celestrak_file_name', type=str, default='celestrak/kp_ap_processed_timeseries.csv', help='CelesTrak dataset file name')
    parser.add_argument('--omniweb_dir', type=str, default='omniweb_karman_2025', help='OMNIWeb dataset directory')
    parser.add_argument('--omniweb_columns', nargs='+', default=['omniweb__sym_d__[nT]', 'omniweb__sym_h__[nT]', 'omniweb__asy_d__[nT]', 'omniweb__bx_gse__[nT]', 'omniweb__by_gse__[nT]', 'omniweb__bz_gse__[nT]', 'omniweb__speed__[km/s]', 'omniweb__vx_velocity__[km/s]', 'omniweb__vy_velocity__[km/s]', 'omniweb__vz_velocity__[km/s]'], help='List of OMNIWeb dataset columns to use')
    parser.add_argument('--set_file_name', type=str, default='set/karman-2025_data_sw_data_set_sw.csv', help='SET dataset file name')
    parser.add_argument('--sdocore_file_name', type=str, default='sdocore/sdo_core_dataset_21504.h5', help='Name of the SDOCore dataset file')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    parser.add_argument('--date_start', type=str, default='2010-05-13T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-08-01T00:00:00', help='End date')
    parser.add_argument('--date_dilation', type=int, default=1, help='Dilation factor for the construction of sequence starting points, e.g. 1 means every delta_minutes, 2 means every 2 * delta_minutes, etc.')

    # parser.add_argument('--date_start', type=str, default='2024-04-19T00:00:00', help='Start date')
    # parser.add_argument('--date_end', type=str, default='2024-04-20T00:00:00', help='End date')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time step in minutes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--eval_mode', type=str, choices=['long_horizon', 'fixed_lead_time', 'all'], default='all', help='Type of evaluation to run in test mode.')
    parser.add_argument('--lead_times', nargs='+', type=int, default=[15, 30, 60, 90, 120], help='A list of lead times in minutes for fixed-lead-time evaluation.')
    
    
    parser.add_argument('--model_type', type=str,
    choices=['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLinear', 'IonCastLSTMSDO',
             'IonCastLSTM-ablation-JPLD', 'IonCastLSTM-ablation-JPLDSunMoon',
             'IonCastLinear-ablation-JPLD', 'IonCastPersistence-ablation-JPLD',
             'SphericalFourierNeuralOperatorModel'],
    default='IonCastLSTM',
    help='Type of model to use')

    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_evals', type=int, default=4, help='Number of samples for evaluation')
    parser.add_argument('--context_window', type=int, default=4, help='Context window size for the model')
    parser.add_argument('--prediction_window', type=int, default=1, help='Evaluation window size for the model')
    parser.add_argument('--valid_event_id', nargs='*', default=validation_events_4, help='Validation event IDs to use for evaluation at the end of each epoch')
    parser.add_argument('--valid_event_seen_id', nargs='*', default=None, help='Event IDs to use for evaluation at the end of each epoch, where the event was a part of the training set')
    parser.add_argument('--max_valid_samples', type=int, default=1000, help='Maximum number of validation samples to use for evaluation')
    parser.add_argument('--test_event_id', nargs='*', default=['G0H6-201706111800', 'G4H9-202303231800'], help='Test event IDs to use for evaluation')
    parser.add_argument('--forecast_max_time_steps', type=int, default=48, help='Maximum number of time steps to evaluate for each test event')
    parser.add_argument('--model_file', type=str, help='Path to the model file to load for testing')
    parser.add_argument('--sun_moon_extra_time_steps', type=int, default=1, help='Number of extra time steps ahead to include in the dataset for Sun and Moon geometry')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate for the model')

    parser.add_argument('--aux_weight', type=float, default=1.0, help='Weight for auxiliary loss terms (SFNO)')
    parser.add_argument('--n_harmonics', type=int, default=1, help='Fourier positional encoding harmonics (SFNO)')
    parser.add_argument('--sfno_width', type=int, default=64)
    parser.add_argument('--sfno_depth', type=int, default=8)
    parser.add_argument('--sfno_modes_lat', type=int, default=32)
    parser.add_argument('--sfno_modes_lon', type=int, default=64)
    parser.add_argument('--n_sunlocked_heads', type=int, default=360)
    parser.add_argument('--area_weighted_loss', action='store_true', help='Enable area-weighted loss in SFNO')

    parser.add_argument('--head_blend_sigma', type=float, default=2.0,
                    help='Gaussian sigma in head-index units for sun-locked blending')
    parser.add_argument('--head_smooth_reg', type=float, default=1e-4)
    parser.add_argument('--lon_tv_reg', type=float, default=5e-6)
    parser.add_argument('--lon_highfreq_reg', type=float, default=0.0)
    parser.add_argument('--lon_highfreq_kmin', type=int, default=72)

    parser.add_argument('--lon_blur_sigma_deg', type=float, default=5.0,
        help='Longitude-only Gaussian blur on μ (degrees). 0 disables.')


    # --- Spectral backend selection ---
    parser.add_argument(
        "--spectral_backend",
        type=str,
        default="sht",
        choices=["rfft", "sht"],
        help="Spectral operator backend: 'rfft' or 'sht' default (torch_harmonics).",
    )

    # --- add near other SFNO args ---
    parser.add_argument('--aux_dim', type=int, default=0,
                        help='Number of auxiliary output channels for SFNO')
    # Requires Python 3.9+ for BooleanOptionalAction
    parser.add_argument('--mc_dropout', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable/disable MC dropout in SFNO (default: enabled)')

    # Mixed-precision + memory format
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable bfloat16 autocast on CUDA (safe with A100)')
    parser.add_argument('--channels_last', action=argparse.BooleanOptionalAction, default=True,
                        help='Use NHWC (channels_last) tensors for faster convs')

    # Throttle per-iteration logging to reduce CPU/IPC overhead
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log/refresh rate (iterations). Higher -> less overhead')



    
    parser.add_argument('--jpld_weight', type=float, default=20.0, help='Weight for the JPLD loss in the total loss calculation')
    parser.add_argument('--save_all_models', action='store_true', help='If set, save all models during training, not just the last one')
    parser.add_argument('--save_all_channels', action='store_true', help='If set, save all channels in the forecast video, not just the JPLD channel')
    parser.add_argument('--valid_every_nth_epoch', type=int, default=1, help='Validate every nth epoch')
    parser.add_argument('--cache_dir', type=str, default=None, help='If set, build an on-disk cache for all training batches, to speed up training (WARNING: this will take a lot of disk space, ~terabytes per year)')
    parser.add_argument('--no_model_checkpoint', action='store_true', help='If set, do not save model checkpoints during training')
    parser.add_argument('--no_valid', action='store_true', help='If set, do not run validation during training')
    parser.add_argument('--no_eval', action='store_true', help='If set, do not run evaluation (event videos etc.) during training, but do compute the validation loss')
    
    # Weights & Biases options
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb_project', type=str, default='Ionosphere')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_notes', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='*', default=None)
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable W&B (same as --wandb_mode disabled)')

    args = parser.parse_args()
    # --- W&B setup (unchanged) ---
    if args.wandb_disabled:
        args.wandb_mode = 'disabled'
    wandb_config = vars(args).copy()
    if args.wandb_mode != 'disabled' and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            notes=args.wandb_notes,
            tags=args.wandb_tags,
            config=wandb_config,
            dir=args.target_dir,
            mode=args.wandb_mode
        )

    # If this run is controlled by a W&B Sweep, let sweep params override CLI args.
    args = _apply_wandb_config_overrides(args)

    # Recompute after overrides so sweeps can flip it.
    use_sht = (getattr(args, "spectral_backend", "sht").lower() == "sht")


    # If target_dir contains placeholders, resolve them now (works with or without sweeps)
    args.target_dir = _materialize_target_dir_placeholders(args.target_dir)

    args_cache_affecting_keys = {'data_dir', 
                                 'jpld_dir', 
                                 'celestrak_file_name', 
                                 'omniweb_dir', 
                                 'omniweb_columns', 
                                 'set_file_name',
                                 'sdocore_file_name',
                                 'date_start', 
                                 'date_end', 
                                 'date_dilation',
                                 'delta_minutes', 
                                 'batch_size', 
                                 'model_type', 
                                 'context_window', 
                                 'prediction_window', 
                                 'valid_event_id', 
                                 'valid_event_seen_id', 
                                 'forecast_max_time_steps',
                                 'sun_moon_extra_time_steps',
                                }
    args_cache_affecting = {k: v for k, v in vars(args).items() if k in args_cache_affecting_keys}
    args_cache_affecting_hash = md5_hash_str(str(args_cache_affecting))

    os.makedirs(args.target_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = os.path.join(args.target_dir, f'log-{timestamp}.txt')

    set_random_seed(args.seed)
    device = torch.device(args.device)

    use_amp = bool(getattr(args, "amp", False)) and (device.type == "cuda")
    # before the training loop, after you compute use_amp
    dcast = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    use_channels_last = bool(getattr(args, "channels_last", False)) and (device.type == "cuda")

    print(f"AMP enabled: {use_amp}")
    print(f"Channels-last enabled: {use_channels_last}")

    global _lat_grid, _lon_grid
    _lat_grid, _lon_grid = _get_latlon_grids()

    with Tee(log_file):
        print(description)
        print('Log file:', log_file)
        print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
        print('Config:')
        print(f"AMP enabled: {use_amp}")
        print(f"Channels-last enabled: {use_channels_last}")
        pprint.pprint(vars(args), depth=2, width=50)

        start_time = datetime.datetime.now()
        print('Start time: {}'.format(start_time))

        if args.mode == 'train':
            print('\n*** Training mode\n')

            if args.batch_size < args.num_evals:
                print(f'Warning: Batch size {args.batch_size} is less than num_evals {args.num_evals}. Using the batch size for num_evals.')
                args.num_evals = args.batch_size

            date_start = datetime.datetime.fromisoformat(args.date_start)
            date_end = datetime.datetime.fromisoformat(args.date_end)

            training_sequence_length = args.context_window + args.prediction_window

            dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)
            dataset_celestrak_file_name = os.path.join(args.data_dir, args.celestrak_file_name)
            dataset_omniweb_dir = os.path.join(args.data_dir, args.omniweb_dir)
            dataset_set_file_name = os.path.join(args.data_dir, args.set_file_name)
            dataset_sdocore_file_name = os.path.join(args.data_dir, args.sdocore_file_name)

            print('Processing excluded dates')

            datasets_sunmoon_valid = []

            date_exclusions = []
            if args.valid_event_id:
                for event_id in args.valid_event_id:
                    print('Excluding event ID: {}'.format(event_id))
                    if event_id not in event_catalog:
                        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                    event = event_catalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(event['date_start']) - datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                    exclusion_end = datetime.datetime.fromisoformat(event['date_end'])
                    date_exclusions.append((exclusion_start, exclusion_end))

                    datasets_sunmoon_valid.append(SunMoonGeometry(date_start=exclusion_start, date_end=exclusion_end, extra_time_steps=args.sun_moon_extra_time_steps))

            dataset_sunmoon_valid = Union(datasets=datasets_sunmoon_valid)

            # if args.model_type == 'VAE1':
            #     dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
            #     dataset_train = dataset_jpld_train
            #     dataset_valid = dataset_jpld_valid
            if args.model_type in ['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLinear', 'SphericalFourierNeuralOperatorModel']:
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_jpld_valid = JPLD(dataset_jpld_dir, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                # dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_celestrak_train = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end, return_as_image_size=(180, 360))
                dataset_celestrak_valid = CelesTrak(dataset_celestrak_file_name, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end, return_as_image_size=(180, 360))
                dataset_omniweb_train = OMNIWeb(dataset_omniweb_dir, date_start=date_start, date_end=date_end, column=args.omniweb_columns, return_as_image_size=(180, 360))
                dataset_omniweb_valid = OMNIWeb(dataset_omniweb_dir, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end, column=args.omniweb_columns, return_as_image_size=(180, 360))
                dataset_set_train = SET(dataset_set_file_name, date_start=date_start, date_end=date_end, return_as_image_size=(180, 360))
                dataset_set_valid = SET(dataset_set_file_name, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end, return_as_image_size=(180, 360))
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train, dataset_celestrak_train, dataset_omniweb_train, dataset_set_train], sequence_length=training_sequence_length, dilation=args.date_dilation)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid, dataset_celestrak_valid, dataset_omniweb_valid, dataset_set_valid], sequence_length=training_sequence_length, dilation=args.date_dilation)
            elif args.model_type in ['IonCastLSTM-ablation-JPLD', 'IonCastLinear-ablation-JPLD', 'IonCastPersistence-ablation-JPLD']:
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_jpld_valid = JPLD(dataset_jpld_dir, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end)
                dataset_train = Sequences(datasets=[dataset_jpld_train], sequence_length=training_sequence_length, dilation=args.date_dilation)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid], sequence_length=training_sequence_length, dilation=args.date_dilation)
            elif args.model_type == 'IonCastLSTM-ablation-JPLDSunMoon':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_jpld_valid = JPLD(dataset_jpld_dir, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train], sequence_length=training_sequence_length, dilation=args.date_dilation)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid], sequence_length=training_sequence_length, dilation=args.date_dilation)
            elif args.model_type == 'IonCastLSTMSDO':
                # SDO model uses only JPLD + SunMoonGeometry for image channels, SDOCore for context
                # Limit all datasets to SDO data range (2010-2018) since SDOCore is required
                from datetime import datetime as dt
                sdo_end_date = dt.fromisoformat('2018-08-17T04:48:00')
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=sdo_end_date, date_exclusions=date_exclusions)
                dataset_jpld_valid = JPLD(dataset_jpld_dir, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=sdo_end_date, extra_time_steps=args.sun_moon_extra_time_steps)
                # dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_sdocore_train = SDOCore(dataset_sdocore_file_name, date_start=date_start, date_end=sdo_end_date)
                dataset_sdocore_valid = SDOCore(dataset_sdocore_file_name, date_start=dataset_sunmoon_valid.date_start, date_end=dataset_sunmoon_valid.date_end)
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train, dataset_sdocore_train], sequence_length=training_sequence_length, dilation=args.date_dilation)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid, dataset_sdocore_valid], sequence_length=training_sequence_length, dilation=args.date_dilation)
            else:
                raise ValueError('Unknown model type: {}'.format(args.model_type))

            # Pick seen events within the training data, if not given
            if args.valid_event_seen_id is None:
                num_seen_events = max(2, len(args.valid_event_id))
                date_start_plus_context = dataset_train.date_start + datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                event_catalog_within_training_set = event_catalog.filter(date_start=date_start_plus_context, date_end=dataset_train.date_end).exclude(date_exclusions=date_exclusions)
                
                # Additional filtering: ensure no event's context window overlaps with excluded date ranges
                event_catalog_no_context_overlap = event_catalog_within_training_set.exclude_context_overlap(date_exclusions, args.context_window * args.delta_minutes)
                
                if len(event_catalog_no_context_overlap) > 0:
                    # Sample from the filtered list
                    args.valid_event_seen_id = event_catalog_no_context_overlap.sample(min(num_seen_events, len(event_catalog_no_context_overlap))).ids()
                    print('\nUsing validation events seen during training: {}\n'.format(args.valid_event_seen_id))
                else:
                    print('\nNo validation events seen during training found within the training set. Using empty list.\n')
                    args.valid_event_seen_id = []

            print('\nTrain size: {:,}'.format(len(dataset_train)))
            print('Valid size: {:,}'.format(len(dataset_valid)))

            if args.cache_dir:
                # use the hash of the entire args object as the directory suffix for the cached dataset
                train_cache_dir = os.path.join(args.cache_dir, 'train-' + args_cache_affecting_hash)
                _persist_kwargs = {}
                if args.num_workers > 0:
                    _persist_kwargs = dict(persistent_workers=True, prefetch_factor=4)
                train_loader = CachedDataLoader(dataset_train, 
                                                batch_size=args.batch_size, 
                                                cache_dir=train_cache_dir, 
                                                num_workers=args.num_workers, 
                                                shuffle=True,
                                                pin_memory=True,
                                                **_persist_kwargs,
                                                name='train_loader')

                valid_cache_dir = os.path.join(args.cache_dir, 'valid-' + args_cache_affecting_hash)
                valid_loader = CachedDataLoader(dataset_valid, 
                                                batch_size=args.batch_size, 
                                                cache_dir=valid_cache_dir, 
                                                num_workers=args.num_workers, 
                                                shuffle=False,
                                                pin_memory=True,
                                                **_persist_kwargs,
                                                name='valid_loader')
            else:
                # No on-disk caching
                common_kwargs = dict(batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
                if args.num_workers > 0:
                    common_kwargs.update(persistent_workers=True, prefetch_factor=4)
                # Prefer GPU-pinned page tables when available (PyTorch 2.1+)
                if torch.cuda.is_available() and args.device.startswith('cuda'):
                    try:
                        common_kwargs['pin_memory_device'] = 'cuda'
                    except TypeError:
                        pass  # older torch

                train_loader = DataLoader(dataset_train, shuffle=True, **common_kwargs)

                if args.max_valid_samples is not None and len(dataset_valid) > args.max_valid_samples:
                    print('Using a random subset of {:,} samples for validation'.format(args.max_valid_samples))
                    indices = random.sample(range(len(dataset_valid)), args.max_valid_samples)
                    sampler = SubsetRandomSampler(indices)
                    vl_kwargs = dict(**common_kwargs)
                    vl_kwargs.pop('persistent_workers', None)
                    vl_kwargs.pop('prefetch_factor', None)
                    valid_loader = DataLoader(dataset_valid, sampler=sampler, **vl_kwargs)
                else:
                    valid_loader = DataLoader(dataset_valid, shuffle=False, **common_kwargs)


            print()

            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = glob.glob('{}/epoch-*-model.pth'.format(args.target_dir))
            if len(model_files) > 0:
                model_files.sort()
                model_file = model_files[-1]
                print('Resuming training from model file: {}'.format(model_file))
                model, optimizer, epoch, iteration, train_losses, valid_losses, scheduler_state_dict, train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses, best_valid_rmse = load_model(model_file, device)
                if getattr(model, "name", "") == "SphericalFourierNeuralOperatorModel":
                    model.output_blur_sigma = 0.85
                    model.head_blend_sigma = args.head_blend_sigma
                    model.lon_blur_sigma_deg = args.lon_blur_sigma_deg
                epoch_start = epoch + 1
                iteration = iteration + 1
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
                scheduler.load_state_dict(scheduler_state_dict)
                print('Next epoch    : {:,}'.format(epoch_start))
                print('Next iteration: {:,}'.format(iteration+1))
            else:
                print('Creating new model')
                if args.model_type == 'IonCastConvLSTM':
                    total_channels = 58  # JPLD + Sun and Moon geometry + CelesTrak + OMNIWeb + SET
                    name = 'IonCastConvLSTM'
                    model = IonCastConvLSTM(input_channels=total_channels, output_channels=total_channels,
                                            context_window=args.context_window, prediction_window=args.prediction_window,
                                            dropout=args.dropout, name=name)
                elif args.model_type == 'IonCastLSTM':
                    total_channels = 58
                    name = 'IonCastLSTM'
                    model = IonCastLSTM(input_channels=total_channels, output_channels=total_channels,
                                        context_window=args.context_window, dropout=args.dropout, name=name)
                elif args.model_type == 'IonCastLSTM-ablation-JPLD':
                    total_channels = 1
                    name = 'IonCastLSTM-ablation-JPLD'
                    model = IonCastLSTM(input_channels=total_channels, output_channels=total_channels,
                                        context_window=args.context_window, dropout=args.dropout, name=name)
                elif args.model_type == 'IonCastLSTM-ablation-JPLDSunMoon':
                    total_channels = 37  # 1 (JPLD) + 36 (SunMoonGeometry default)
                    name = 'IonCastLSTM-ablation-JPLDSunMoon'
                    model = IonCastLSTM(input_channels=total_channels, output_channels=total_channels,
                                        context_window=args.context_window, dropout=args.dropout, name=name)
                elif args.model_type == 'IonCastLinear':
                    total_channels = 58
                    name = 'IonCastLinear'
                    model = IonCastLinear(input_channels=total_channels, output_channels=total_channels,
                                        context_window=args.context_window, name=name)
                elif args.model_type == 'IonCastLinear-ablation-JPLD':
                    total_channels = 1
                    name = 'IonCastLinear-ablation-JPLD'
                    model = IonCastLinear(input_channels=total_channels, output_channels=total_channels,
                                        context_window=args.context_window, name=name)
                elif args.model_type == 'IonCastPersistence-ablation-JPLD':
                    total_channels = 1
                    name = 'IonCastPersistence-ablation-JPLD'
                    model = IonCastPersistence(input_channels=total_channels, output_channels=total_channels,
                                            context_window=args.context_window, name=name)
                elif args.model_type == 'IonCastLSTMSDO':
                    total_channels = 37
                    name = 'IonCastLSTMSDO'
                    model = IonCastLSTMSDO(input_channels=total_channels, output_channels=total_channels,
                                        context_window=args.context_window, dropout=args.dropout, sdo_dim=21504, name=name)
                elif args.model_type == 'SphericalFourierNeuralOperatorModel':
                    if SphericalFourierNeuralOperatorModel is None:
                        raise ImportError('SphericalFourierNeuralOperatorModel not available')
                    # Probe 1 batch to derive SFNO in_channels after positional encodings
                    probe = next(iter(train_loader))
                    jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, times = probe
                    H, W = 180, 360
                    celestrak_seq = ensure_grid(celestrak_seq, target_channels=2, H=H, W=W)
                    omniweb_seq   = ensure_grid(omniweb_seq,   target_channels=len(args.omniweb_columns), H=H, W=W)
                    set_seq       = ensure_grid(set_seq,       target_channels=9, H=H, W=W)
                    combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)
                    x_last = combined_seq[:, args.context_window - 1, :, :, :]         # (B, C, H, W)
                    # Build sun-locked & QD grids per sample
                    B = x_last.shape[0]
                    sunlocked_grids, qd_lat_list, qd_lon_list = [], [], []
                    step_idx = max(0, args.context_window - 1)
                    for b in range(B):
                        dtb = _safe_time_at(times, b, step_idx)
                        subsolar_lon = _get_subsolar_longitude(dtb)
                        sunlocked_lon_grid = ((_lon_grid - subsolar_lon + 360) % 360)
                        qd_lat_f, qd_lon_f = _load_qd_grid_for_year(dtb.year)
                        sunlocked_grids.append(sunlocked_lon_grid)
                        qd_lat_list.append(qd_lat_f)
                        qd_lon_list.append(qd_lon_f)
                    sunlocked_grids = np.stack(sunlocked_grids, axis=0)
                    coord_grids = [
                        np.repeat(_lat_grid[None, ...], B, axis=0),
                        np.repeat(_lon_grid[None, ...], B, axis=0),
                        np.stack(qd_lat_list, axis=0),
                        np.stack(qd_lon_list, axis=0),
                        sunlocked_grids,
                    ]
                    PE_FLAGS = [True, False, True, False, False]   # lat, lon, qd_lat, qd_lon, sunlocked
                    x_input = add_fourier_positional_encoding(
                        x_last, coord_grids, n_harmonics=args.n_harmonics, concat_flags=PE_FLAGS
                    )
                    in_channels = x_input.shape[1]
                    print(f"[SFNO] Derived in_channels={in_channels}")
                    # Instantiate SFNO (tweak hyperparams as you like)
                    model = SphericalFourierNeuralOperatorModel(
                        in_channels=in_channels,
                        trunk_width=getattr(args, "sfno_width", 64),
                        trunk_depth=getattr(args, "sfno_depth", 8),
                        modes_lat=getattr(args, "sfno_modes_lat", 32),
                        modes_lon=getattr(args, "sfno_modes_lon", 64),
                        aux_dim=getattr(args, "aux_dim", 0),
                        tasks=("vtec",),
                        out_shapes={"vtec": (1, "grid")},            # per-step channels; horizon via AR
                        probabilistic=True,
                        dropout=getattr(args, "dropout", 0.2),
                        mc_dropout=getattr(args, "mc_dropout", True),
                        n_sunlocked_heads=getattr(args, "n_sunlocked_heads", 360),
                        context_window=args.context_window,
                        prediction_window=args.prediction_window,
                        use_sht=use_sht,                              # <-- IMPORTANT: rFFT vs SHT
                        area_weighted_loss=getattr(args, "area_weighted_loss", False),
                        head_smooth_reg=args.head_smooth_reg,
                        lon_tv_reg=args.lon_tv_reg,
                        lon_highfreq_reg=args.lon_highfreq_reg,
                        lon_highfreq_kmin=args.lon_highfreq_kmin,
                        lon_blur_sigma_deg=args.lon_blur_sigma_deg,
                    )
                    model.name = "SphericalFourierNeuralOperatorModel"
                    #below adds a little smoothing of μ (3×3 Gaussian)
                    model.output_blur_sigma = 1.2
                    model.head_blend_sigma = args.head_blend_sigma
                    model.lon_blur_sigma_deg = args.lon_blur_sigma_deg

                else:
                    raise ValueError('Unknown model type: {}'.format(args.model_type))


                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
                iteration = 0
                epoch_start = 0
                train_losses = []
                valid_losses = []
                train_rmse_losses = []
                valid_rmse_losses = []
                train_jpld_rmse_losses = []
                valid_jpld_rmse_losses = []
                best_valid_rmse = float('inf')

                model = model.to(device)


            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if wandb is not None and args.wandb_mode != 'disabled':
                wandb.watch(model, log="all", log_freq=100)
            print('\nNumber of parameters: {:,}\n'.format(num_params))
            
            for epoch in range(epoch_start, args.epochs):
                print('\n*** Epoch {:,}/{:,} started'.format(epoch+1, args.epochs))
                print('*** Training')
                # Training
                model.train()
                with tqdm(total=len(train_loader)) as pbar:

                    for i, batch in enumerate(train_loader):
                        # Zero grads up-front (slightly faster with set_to_none)
                        optimizer.zero_grad(set_to_none=True)

                        # STEP 6: move the whole batch to device once (and set channels_last on 4D tensors)
                        batch = _move_batch_to_device(batch, device, channels_last=use_channels_last)

                        # STEP 5: choose AMP context (bfloat16 on CUDA) or a no-op context
                        amp_ctx = torch_amp.autocast('cuda', dtype=dcast) if use_amp else contextlib.nullcontext()


                        with amp_ctx:
                            # ---- compute loss exactly once (no duplicate branch) ----
                            if model.name in ['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLinear']:
                                jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                                combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)
                                loss, rmse_t, jpld_rmse_t = model.loss(combined_seq, jpld_weight=args.jpld_weight)

                            elif model.name in ['IonCastLSTM-ablation-JPLD', 'IonCastLinear-ablation-JPLD', 'IonCastPersistence-ablation-JPLD']:
                                jpld_seq, _ = batch
                                loss, rmse_t, jpld_rmse_t = model.loss(jpld_seq, jpld_weight=args.jpld_weight)

                            elif model.name == 'IonCastLSTM-ablation-JPLDSunMoon':
                                jpld_seq, sunmoon_seq, _ = batch
                                combined_seq = torch.cat((jpld_seq, sunmoon_seq), dim=2)
                                loss, rmse_t, jpld_rmse_t = model.loss(combined_seq, jpld_weight=args.jpld_weight)

                            elif model.name == 'IonCastLSTMSDO':
                                jpld_seq, sunmoon_seq, sdo_seq, _ = batch
                                combined_seq = torch.cat((jpld_seq, sunmoon_seq), dim=2)
                                sdo_context = sdo_seq[:, :args.context_window, :]
                                loss, rmse_t, jpld_rmse_t = model.loss(combined_seq, sdo_context, jpld_weight=args.jpld_weight)

                            elif args.model_type == 'SphericalFourierNeuralOperatorModel':
                                # ---- true multi-step AR rollout in AMP context ----
                                jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, times = batch

                                H, W = 180, 360
                                celestrak_seq = ensure_grid(celestrak_seq, target_channels=2, H=H, W=W)
                                omniweb_seq   = ensure_grid(omniweb_seq,   target_channels=len(args.omniweb_columns), H=H, W=W)
                                set_seq       = ensure_grid(set_seq,       target_channels=9, H=H, W=W)

                                combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)
                                B, T, Ctot, H, W = combined_seq.shape
                                assert T >= args.context_window + args.prediction_window, "Sequence too short for context+prediction"

                                P = args.prediction_window
                                cur = combined_seq[:, args.context_window - 1].clone()     # (B, Ctot, H, W)
                                total_loss = 0.0
                                rmse_steps = []
                                slots_per_day = max(1, int(round(1440.0 / float(args.delta_minutes))))

                                for step in range(P):
                                    step_idx = args.context_window - 1 + step

                                    # Build sun-locked & QD grids for THIS step
                                    sunlocked_grids, qd_lat_list, qd_lon_list = [], [], []
                                    for b in range(B):
                                        dtb = _safe_time_at(times, b, step_idx)
                                        subsolar_lon = _get_subsolar_longitude(dtb)
                                        sunlocked_lon_grid = ((_lon_grid - subsolar_lon + 360) % 360)
                                        qd_lat_f, qd_lon_f = _load_qd_grid_for_year(dtb.year)
                                        sunlocked_grids.append(sunlocked_lon_grid)
                                        qd_lat_list.append(qd_lat_f)
                                        qd_lon_list.append(qd_lon_f)
                                    sunlocked_grids = np.stack(sunlocked_grids, axis=0)

                                    coord_grids = [
                                        np.repeat(_lat_grid[None, ...], B, axis=0),
                                        np.repeat(_lon_grid[None, ...], B, axis=0),
                                        np.stack(qd_lat_list, axis=0),
                                        np.stack(qd_lon_list, axis=0),
                                        sunlocked_grids,
                                    ]

                                    # Append Fourier PE each step (input is one frame + PE)
                                    x_input = add_fourier_positional_encoding(
                                        cur, coord_grids, n_harmonics=args.n_harmonics,
                                        concat_flags=[True, False, True, False, False]
                                    )

                                    # Sun-locked head indices for attention routing
                                    n_heads = int(getattr(model, 'n_sunlocked_heads', 360))
                                    sunlocked_idx_np = _quantize_degrees_to_heads(sunlocked_grids, n_heads)
                                    sunlocked_idx = torch.tensor(sunlocked_idx_np, dtype=torch.long, device=device)

                                    out = model(
                                        x_input,
                                        sunlocked_idx,
                                        sunlocked_deg=torch.tensor(sunlocked_grids, dtype=torch.float32, device=device),
                                        head_blend_sigma=args.head_blend_sigma,
                                    )

                                    if isinstance(out, dict) and 'vtec' in out:
                                        pred, logvar = out['vtec']             # (B,1,H,W) each
                                        yhat = pred
                                    else:
                                        pred = out
                                        logvar = None
                                        yhat = pred

                                    # Target frame at t = context + step
                                    y_t = jpld_seq[:, args.context_window + step, 0:1, :, :]

                                    # Physics-informed loss if available, else plain MSE
                                    if isinstance(out, dict) and 'vtec' in out:
                                        times_target = [_safe_time_at(times, b, args.context_window + step) for b in range(B)]
                                        loss_step, _, _, _ = _sfno_weighted_loss(
                                            pred, logvar, y_t, iteration,
                                            jpld_weight=args.jpld_weight, aux_weight=args.aux_weight,
                                            times=times_target, slots_per_day=slots_per_day,
                                            delta_minutes=args.delta_minutes, tol_slots=1
                                        )
                                    else:
                                        loss_step = torch.mean((pred - y_t) ** 2)

                                    # --- anti-stripe / vertical-noise regularizers ---
                                    if getattr(model, 'head_smooth_reg', 0.0) > 0.0:
                                        loss_step = loss_step + model.head_smooth_reg * model._head_smoothness_penalty()
                                    if getattr(model, 'lon_tv_reg', 0.0) > 0.0:
                                        loss_step = loss_step + model.lon_tv_reg * model._lon_tv_penalty(pred)
                                    if getattr(model, 'lon_highfreq_reg', 0.0) > 0.0:
                                        loss_step = loss_step + model.lon_highfreq_reg * model._lon_highfreq_penalty(pred)


                                    total_loss = total_loss + loss_step
                                    rmse_steps.append(_rmse_tecu(
                                        pred[:, 0:1] if pred.dim() == 4 and pred.shape[1] > 1 else pred, y_t
                                    ))

                                    # Build next-step frame (AR): keep exogenous truth, replace VTEC with our prediction
                                    next_frame = combined_seq[:, args.context_window + step].clone()
                                    next_frame[:, 0:1] = yhat
                                    cur = next_frame

                                loss = total_loss / float(P)
                                rmse_t = torch.stack(rmse_steps).mean()
                                jpld_rmse_t = rmse_t

                            else:
                                raise ValueError('Unknown model type: {}'.format(model.name if hasattr(model, "name") else args.model_type))

                        # Standard backward + step (bfloat16 AMP does not need a GradScaler)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        iteration += 1

                        # Convert to plain floats exactly once for logging
                        def _to_float(x):
                            return float(x.detach().item()) if torch.is_tensor(x) else float(x)

                        loss_val = _to_float(loss)
                        rmse_val = _to_float(rmse_t)
                        jpld_rmse_val = _to_float(jpld_rmse_t)

                        train_losses.append((iteration, loss_val))
                        train_rmse_losses.append((iteration, rmse_val))
                        train_jpld_rmse_losses.append((iteration, jpld_rmse_val))

                        if wandb is not None and args.wandb_mode != 'disabled':
                            wandb.log({
                                'train/loss': loss_val,
                                'train/rmse': rmse_val,
                                'train/jpld_rmse': jpld_rmse_val,
                                'train/epoch': epoch + 1,
                                'train/iteration': iteration,
                                'lr': optimizer.param_groups[0]['lr'],
                            }, step=iteration)

                        pbar.set_description(
                            f'Epoch {epoch + 1}/{args.epochs}, MSE: {loss_val:.4f}, RMSE: {rmse_val:.4f}, JPLD RMSE: {jpld_rmse_val:.4f}'
                        )

                        pbar.update(1)

                # Validation
                if (not args.no_valid) and ((epoch + 1) % args.valid_every_nth_epoch == 0):
                    print('*** Validation')
                    model.eval()
                    if hasattr(model, "set_mc_dropout"):
                        model.set_mc_dropout(False)
                    valid_loss = 0.0
                    valid_rmse_loss = 0.0
                    valid_jpld_rmse_loss = 0.0

                    with torch.no_grad():
                        for batch in tqdm(valid_loader, desc='Validation', leave=False):
                            if model.name in ['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLinear']:
                                jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                                jpld_seq = jpld_seq.to(device); sunmoon_seq = sunmoon_seq.to(device)
                                celestrak_seq = celestrak_seq.to(device); omniweb_seq = omniweb_seq.to(device); set_seq = set_seq.to(device)
                                combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)
                                loss, rmse_t, jpld_rmse_t = model.loss(combined_seq, jpld_weight=args.jpld_weight)

                            elif model.name in ['IonCastLSTM-ablation-JPLD', 'IonCastLinear-ablation-JPLD', 'IonCastPersistence-ablation-JPLD']:
                                jpld_seq, _ = batch
                                jpld_seq = jpld_seq.to(device)
                                loss, rmse_t, jpld_rmse_t = model.loss(jpld_seq, jpld_weight=args.jpld_weight)

                            elif model.name == 'IonCastLSTM-ablation-JPLDSunMoon':
                                jpld_seq, sunmoon_seq, _ = batch
                                jpld_seq = jpld_seq.to(device); sunmoon_seq = sunmoon_seq.to(device)
                                combined_seq = torch.cat((jpld_seq, sunmoon_seq), dim=2)
                                loss, rmse_t, jpld_rmse_t = model.loss(combined_seq, jpld_weight=args.jpld_weight)

                            elif model.name == 'IonCastLSTMSDO':
                                jpld_seq, sunmoon_seq, sdo_seq, _ = batch
                                jpld_seq = jpld_seq.to(device); sunmoon_seq = sunmoon_seq.to(device); sdo_seq = sdo_seq.to(device)
                                combined_seq = torch.cat((jpld_seq, sunmoon_seq), dim=2)
                                sdo_context = sdo_seq[:, :args.context_window, :]
                                loss, rmse_t, jpld_rmse_t = model.loss(combined_seq, sdo_context, jpld_weight=args.jpld_weight)

                            elif model.name == 'SphericalFourierNeuralOperatorModel':
                                jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, times = batch
                                jpld_seq = jpld_seq.to(device); sunmoon_seq = sunmoon_seq.to(device)
                                celestrak_seq = celestrak_seq.to(device); omniweb_seq = omniweb_seq.to(device); set_seq = set_seq.to(device)

                                # Ensure fixed grid channel counts for exogenous data
                                H, W = 180, 360
                                celestrak_seq = ensure_grid(celestrak_seq, target_channels=2, H=H, W=W)
                                omniweb_seq   = ensure_grid(omniweb_seq,   target_channels=len(args.omniweb_columns), H=H, W=W)
                                set_seq       = ensure_grid(set_seq,       target_channels=9, H=H, W=W)

                                combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)
                                B, T, Ctot, H, W = combined_seq.shape
                                assert T >= args.context_window + args.prediction_window, "Sequence too short for validation"

                                # Autoregressive rollout over P prediction steps
                                P = args.prediction_window
                                cur = combined_seq[:, args.context_window - 1].clone()
                                total_loss = 0.0
                                rmse_steps = []
                                slots_per_day = max(1, int(round(1440.0 / float(args.delta_minutes))))

                                for step in range(P):
                                    step_idx = args.context_window - 1 + step
                                    sunlocked_grids, qd_lat_list, qd_lon_list = [], [], []
                                    for b in range(B):
                                        dtb = _safe_time_at(times, b, step_idx)
                                        subsolar_lon = _get_subsolar_longitude(dtb)
                                        sunlocked_lon_grid = ((_lon_grid - subsolar_lon + 360) % 360)
                                        qd_lat_f, qd_lon_f = _load_qd_grid_for_year(dtb.year)
                                        sunlocked_grids.append(sunlocked_lon_grid)
                                        qd_lat_list.append(qd_lat_f)
                                        qd_lon_list.append(qd_lon_f)
                                    sunlocked_grids = np.stack(sunlocked_grids, axis=0)

                                    coord_grids = [
                                        np.repeat(_lat_grid[None, ...], B, axis=0),
                                        np.repeat(_lon_grid[None, ...], B, axis=0),
                                        np.stack(qd_lat_list, axis=0),
                                        np.stack(qd_lon_list, axis=0),
                                        sunlocked_grids,
                                    ]

                                    # better (lat, lon, qd_lat, qd_lon, sunlocked):
                                    x_input = add_fourier_positional_encoding(
                                        cur, coord_grids, n_harmonics=args.n_harmonics,
                                        concat_flags=[True, False, True, False, False]
                                    )

                                    n_heads = int(getattr(model, 'n_sunlocked_heads', 360))
                                    sunlocked_idx_np = _quantize_degrees_to_heads(sunlocked_grids, n_heads)
                                    sunlocked_idx = torch.tensor(sunlocked_idx_np, dtype=torch.long, device=device)

                                    out = model(
                                        x_input,
                                        sunlocked_idx,
                                        sunlocked_deg=torch.tensor(sunlocked_grids, dtype=torch.float32, device=device),
                                        head_blend_sigma=args.head_blend_sigma,
                                    )

                                    if isinstance(out, dict) and 'vtec' in out:
                                        pred, logvar = out['vtec']
                                        yhat = pred
                                    else:
                                        pred = out
                                        logvar = None
                                        yhat = pred

                                    # Target = ground truth JPLD at t = context + step
                                    y_t = jpld_seq[:, args.context_window + step, 0:1, :, :]

                                    # Physics-informed loss if available, else plain MSE
                                    if isinstance(out, dict) and 'vtec' in out:
                                        times_target = [_safe_time_at(times, b, args.context_window + step) for b in range(B)]
                                        loss_step, _, _, _ = _sfno_weighted_loss(
                                            pred, logvar, y_t, iteration,
                                            jpld_weight=args.jpld_weight, aux_weight=args.aux_weight,
                                            times=times_target, slots_per_day=slots_per_day,
                                            delta_minutes=args.delta_minutes, tol_slots=1
                                        )
                                    else:
                                        loss_step = torch.mean((pred - y_t) ** 2)

                                    # --- anti-stripe / vertical-noise regularizers ---
                                    if getattr(model, 'head_smooth_reg', 0.0) > 0.0:
                                        loss_step = loss_step + model.head_smooth_reg * model._head_smoothness_penalty()
                                    if getattr(model, 'lon_tv_reg', 0.0) > 0.0:
                                        loss_step = loss_step + model.lon_tv_reg * model._lon_tv_penalty(pred)
                                    if getattr(model, 'lon_highfreq_reg', 0.0) > 0.0:
                                        loss_step = loss_step + model.lon_highfreq_reg * model._lon_highfreq_penalty(pred)


                                    total_loss = total_loss + loss_step
                                    rmse_steps.append(_rmse_tecu(pred[:, 0:1] if pred.dim() == 4 and pred.shape[1] > 1 else pred, y_t))

                                    # Build next input frame: replace VTEC channel with prediction (AR)
                                    next_frame = combined_seq[:, args.context_window + step].clone()
                                    next_frame[:, 0:1] = yhat
                                    cur = next_frame

                                loss = total_loss / float(P)
                                rmse_t = torch.stack(rmse_steps).mean()
                                jpld_rmse_t = rmse_t

                            else:
                                raise ValueError('Unknown model type: {}'.format(model.name if hasattr(model, "name") else args.model_type))

                            # ---- accumulate for ALL branches ----
                            valid_loss += float(loss.detach().item())
                            valid_rmse_loss += float(rmse_t.detach().item())
                            valid_jpld_rmse_loss += float(jpld_rmse_t.detach().item())

                    denom = max(1, len(valid_loader))
                    valid_loss /= denom
                    valid_rmse_loss /= denom
                    valid_jpld_rmse_loss /= denom

                    valid_losses.append((iteration, valid_loss))
                    valid_rmse_losses.append((iteration, valid_rmse_loss))
                    valid_jpld_rmse_losses.append((iteration, valid_jpld_rmse_loss))

                    print(f'Validation Loss: {valid_loss:.4f}, Validation RMSE: {valid_rmse_loss:.4f}, Validation JPLD RMSE: {valid_jpld_rmse_loss:.4f}')

                    # W&B validation metrics
                    if wandb is not None and args.wandb_mode != 'disabled':
                        wandb.log({
                            'valid/loss': float(valid_loss),
                            'valid/rmse': float(valid_rmse_loss),
                            'valid/jpld_rmse': float(valid_jpld_rmse_loss),
                            'epoch': epoch + 1,
                            'iteration': iteration
                        }, step=iteration)

                    scheduler.step(valid_rmse_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Current learning rate: {current_lr:.6f}')


                    if not args.no_eval:
                        file_name_prefix = os.path.join(args.target_dir, f'epoch-{epoch + 1:02d}-')

                        # Save model
                        model_file = f'{file_name_prefix}model.pth'
                        if args.no_model_checkpoint:
                            print('Skipping model saving due to --no_model_checkpoint flag')
                        else:
                            save_model(model, optimizer, scheduler, epoch, iteration, train_losses, valid_losses, train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses, best_valid_rmse, model_file,)
                        if not args.save_all_models:
                            # Remove previous model files if not saving all models
                            previous_model_files = glob.glob(os.path.join(args.target_dir, 'epoch-*-model.pth'))
                            for previous_model_file in previous_model_files:
                                if previous_model_file != model_file:
                                    print(f'Removing previous model file: {previous_model_file}')
                                    os.remove(previous_model_file)

                        # Define consistent colors for plotting
                        color_loss = 'tab:blue'
                        color_rmse_all = 'tab:blue'  # Use blue for All Channels RMSE as requested
                        color_rmse_jpld = 'tab:green'

                        # Plot losses
                        plot_file = f'{file_name_prefix}loss.pdf'
                        print(f'Saving loss plot to {plot_file}')
                        plt.figure(figsize=(10, 5))
                        if train_losses:
                            plt.plot(*zip(*train_losses), label='Training', color=color_loss, alpha=0.5)
                        if valid_losses:
                            plt.plot(*zip(*valid_losses), label='Validation', color=color_loss, linestyle='--', marker='o')
                        plt.xlabel('Iteration')
                        plt.ylabel('MSE Loss')
                        plt.yscale('log')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(plot_file)
                        
                        # Also save as PNG for W&B upload
                        if wandb is not None and args.wandb_mode != 'disabled':
                            png_file = plot_file.replace('.pdf', '.png')
                            plt.savefig(png_file, dpi=300, bbox_inches='tight')
                            plot_name = os.path.splitext(os.path.basename(plot_file))[0]
                            try:
                                wandb.log({f"plots/{plot_name}": wandb.Image(png_file)})
                            except Exception as e:
                                print(f"Warning: Could not upload plot {plot_name}: {e}")
                        
                        plt.close()

                        # Plot RMSE losses
                        plot_rmse_file = f'{file_name_prefix}metrics-rmse.pdf'
                        print(f'Saving RMSE plot to {plot_rmse_file}')
                        plt.figure(figsize=(10, 5))
                        if train_rmse_losses:
                            plt.plot(*zip(*train_rmse_losses), label='Training (All Channels)', color=color_rmse_all, alpha=0.5)
                        if valid_rmse_losses:
                            plt.plot(*zip(*valid_rmse_losses), label='Validation (All Channels)', color=color_rmse_all, linestyle='--', marker='o')
                        if train_jpld_rmse_losses:
                            plt.plot(*zip(*train_jpld_rmse_losses), label='Training (JPLD)', color=color_rmse_jpld, alpha=0.5)
                        if valid_jpld_rmse_losses:
                            plt.plot(*zip(*valid_jpld_rmse_losses), label='Validation (JPLD)', color=color_rmse_jpld, linestyle='--', marker='o')
                        plt.xlabel('Iteration')
                        plt.ylabel('RMSE')
                        plt.yscale('log')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(plot_rmse_file)
                        
                        # Also save as PNG for W&B upload
                        if wandb is not None and args.wandb_mode != 'disabled':
                            png_file = plot_rmse_file.replace('.pdf', '.png')
                            plt.savefig(png_file, dpi=300, bbox_inches='tight')
                            plot_name = os.path.splitext(os.path.basename(plot_rmse_file))[0]
                            try:
                                wandb.log({f"plots/{plot_name}": wandb.Image(png_file)})
                            except Exception as e:
                                print(f"Warning: Could not upload plot {plot_name}: {e}")
                        
                        plt.close()

                        # Plot model eval results
                        model.eval()
                        with torch.no_grad():
                            if (model.name in ['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLSTMSDO', 'IonCastLinear',
                   'IonCastLSTM-ablation-JPLD', 'IonCastLSTM-ablation-JPLDSunMoon',
                   'IonCastLinear-ablation-JPLD', 'IonCastPersistence-ablation-JPLD',
                   'SphericalFourierNeuralOperatorModel']):

                                # --- EVALUATION ON UNSEEN VALIDATION EVENTS ---
                                saved_video_categories = set()
                                metric_event_id = []
                                metric_jpld_rmse = []
                                metric_jpld_mae = []
                                metric_jpld_unnormalized_rmse = []
                                metric_jpld_unnormalized_mae = []
                                metric_jpld_unnormalized_rmse_low_lat = []
                                metric_jpld_unnormalized_rmse_mid_lat = []
                                metric_jpld_unnormalized_rmse_high_lat = []
                                
                                # Fixed-lead-time metrics collection
                                fixed_lead_time_metrics = []
                                fixed_lead_time_event_ids = []
                                
                                if args.valid_event_id:
                                    for i, event_id in enumerate(args.valid_event_id):
                                        print(f'\n--- Evaluating validation event: {event_id} ---')
                                        event_category = event_id.split('-')[0][:2]
                                        save_video = False
                                        if event_category not in saved_video_categories:
                                            save_video = True
                                            saved_video_categories.add(event_category)

                                        # --- Long Horizon Evaluation ---
                                        if args.eval_mode in ['long_horizon', 'all']:
                                            
                                            jpld_rmse, jpld_mae, jpld_unnormalized_rmse_val, jpld_unnormalized_mae_val, jpld_unnormalized_rmse_low_lat_val, jpld_unnormalized_rmse_mid_lat_val, jpld_unnormalized_rmse_high_lat_val = eval_forecast_long_horizon(model, dataset_valid, event_catalog, event_id, file_name_prefix+'valid', save_video, False, save_video, args)
                                            metric_event_id.append(event_id)
                                            metric_jpld_rmse.append(jpld_rmse)
                                            metric_jpld_mae.append(jpld_mae)
                                            metric_jpld_unnormalized_rmse.append(jpld_unnormalized_rmse_val)
                                            metric_jpld_unnormalized_mae.append(jpld_unnormalized_mae_val)
                                            metric_jpld_unnormalized_rmse_low_lat.append(jpld_unnormalized_rmse_low_lat_val)
                                            metric_jpld_unnormalized_rmse_mid_lat.append(jpld_unnormalized_rmse_mid_lat_val)
                                            metric_jpld_unnormalized_rmse_high_lat.append(jpld_unnormalized_rmse_high_lat_val)

                                        # --- Fixed Lead Time Evaluation ---
                                        if args.eval_mode in ['fixed_lead_time', 'all']:
                                            lead_time_errors, event_id_returned = eval_forecast_fixed_lead_time(model, dataset_valid, event_catalog, event_id, args.lead_times, file_name_prefix+'valid', save_video, False, save_video, args)
                                            fixed_lead_time_metrics.append(lead_time_errors)
                                            fixed_lead_time_event_ids.append(event_id_returned)

                                # Save metrics from long-horizon eval
                                if metric_event_id:
                                    metrics_file_prefix = f'{file_name_prefix}valid-long-horizon-metrics'
                                    save_metrics(metric_event_id, metric_jpld_rmse, metric_jpld_mae, metric_jpld_unnormalized_rmse, metric_jpld_unnormalized_mae, metric_jpld_unnormalized_rmse_low_lat, metric_jpld_unnormalized_rmse_mid_lat, metric_jpld_unnormalized_rmse_high_lat, metrics_file_prefix)
                                    
                                    # Upload evaluation metrics to W&B
                                    if wandb is not None and args.wandb_mode != 'disabled':
                                        # Upload as structured data for W&B visualization
                                        for i, event_id in enumerate(metric_event_id):
                                            wandb.log({
                                                f'eval_metrics/{event_id}/jpld_rmse': metric_jpld_rmse[i],
                                                f'eval_metrics/{event_id}/jpld_mae': metric_jpld_mae[i],
                                                f'eval_metrics/{event_id}/jpld_unnormalized_rmse': metric_jpld_unnormalized_rmse[i],
                                                f'eval_metrics/{event_id}/jpld_unnormalized_mae': metric_jpld_unnormalized_mae[i],
                                                f'eval_metrics/{event_id}/jpld_unnormalized_rmse_low_lat': metric_jpld_unnormalized_rmse_low_lat[i],
                                                f'eval_metrics/{event_id}/jpld_unnormalized_rmse_mid_lat': metric_jpld_unnormalized_rmse_mid_lat[i],
                                                f'eval_metrics/{event_id}/jpld_unnormalized_rmse_high_lat': metric_jpld_unnormalized_rmse_high_lat[i],
                                                'epoch': epoch + 1
                                            })
                                        
                                        # Also upload CSV file as artifact if it exists
                                        csv_file = f'{metrics_file_prefix}.csv'
                                        if os.path.exists(csv_file):
                                            try:
                                                artifact = wandb.Artifact(f'validation_metrics_epoch_{epoch+1}', type='evaluation_metrics')
                                                artifact.add_file(csv_file)
                                                wandb.log_artifact(artifact)
                                            except Exception as e:
                                                print(f'Warning: Could not upload metrics CSV to W&B: {e}')
                                
                                # Aggregate and plot fixed-lead-time metrics
                                if fixed_lead_time_metrics:
                                    plot_file_prefix = f'{file_name_prefix}valid'
                                    aggregate_and_plot_fixed_lead_time_metrics(fixed_lead_time_metrics, fixed_lead_time_event_ids, plot_file_prefix)
                                    
                                    # Upload fixed-lead-time metrics CSV to W&B if it exists
                                    if wandb is not None and args.wandb_mode != 'disabled':
                                        csv_file = f'{plot_file_prefix}_fixed_lead_time_metrics_aggregated.csv'
                                        if os.path.exists(csv_file):
                                            try:
                                                artifact = wandb.Artifact(f'fixed_lead_time_metrics_epoch_{epoch+1}', type='evaluation_metrics')
                                                artifact.add_file(csv_file)
                                                wandb.log_artifact(artifact)
                                            except Exception as e:
                                                print(f'Warning: Could not upload fixed-lead-time metrics CSV to W&B: {e}')

                                # --- EVALUATION ON SEEN VALIDATION EVENTS ---
                                saved_video_categories_seen = set()
                                # Fixed-lead-time metrics collection for seen events
                                fixed_lead_time_metrics_seen = []
                                fixed_lead_time_event_ids_seen = []
                                
                                if args.valid_event_seen_id:
                                    for i, event_id in enumerate(args.valid_event_seen_id):
                                        event_category = event_id.split('-')[0][:2]
                                        save_video = False
                                        if event_category not in saved_video_categories_seen:
                                            save_video = True
                                            saved_video_categories_seen.add(event_category)                                    
                                        print(f'\n--- Evaluating seen validation event: {event_id} ---')
                                        # --- Long Horizon Evaluation (Seen) ---
                                        if args.eval_mode in ['long_horizon', 'all']:
                                            # Note: We don't save metrics for 'seen' events to avoid clutter, just the video.
                                            eval_forecast_long_horizon(model, dataset_train, event_catalog, event_id, file_name_prefix+'valid-seen', save_video, False, save_video, args)
                                        
                                        # --- Fixed Lead Time Evaluation (Seen) ---
                                        if args.eval_mode in ['fixed_lead_time', 'all']:
                                            lead_time_errors_seen, event_id_returned_seen = eval_forecast_fixed_lead_time(model, dataset_train, event_catalog, event_id, args.lead_times, file_name_prefix+'valid-seen', save_video, False, save_video, args)
                                            fixed_lead_time_metrics_seen.append(lead_time_errors_seen)
                                            fixed_lead_time_event_ids_seen.append(event_id_returned_seen)
                                
                                # Aggregate and plot fixed-lead-time metrics for seen events
                                if fixed_lead_time_metrics_seen:
                                    plot_file_prefix_seen = f'{file_name_prefix}valid-seen'
                                    aggregate_and_plot_fixed_lead_time_metrics(fixed_lead_time_metrics_seen, fixed_lead_time_event_ids_seen, plot_file_prefix_seen)

                        # --- Best Model Checkpointing Logic ---
                        if valid_rmse_loss < best_valid_rmse:
                            best_valid_rmse = valid_rmse_loss
                            print(f'\n*** New best validation RMSE: {best_valid_rmse:.4f}***\n')
                            # copy model checkpoint and all plots/videos to the best model directory
                            best_model_dir = os.path.join(args.target_dir, 'best_model')
                            print(f'Saving best model to {best_model_dir}')
                            # delete the previous best model directory if it exists
                            if os.path.exists(best_model_dir):
                                shutil.rmtree(best_model_dir)
                            os.makedirs(best_model_dir, exist_ok=True)
                            for file in os.listdir(args.target_dir):
                                if file.startswith(os.path.basename(file_name_prefix)) and (file.endswith('.pdf') or file.endswith('.png') or file.endswith('.mp4') or file.endswith('.pth') or file.endswith('.csv')):
                                    shutil.copyfile(os.path.join(args.target_dir, file), os.path.join(best_model_dir, file))

        elif args.mode == 'test':
            print('*** Testing mode\n')

            if args.model_type == 'IonCastPersistence-ablation-JPLD':
                # Create persistence model directly without loading from file
                print('Creating IonCastPersistence-ablation-JPLD model for testing')
                total_channels = 1  # JPLD (1)
                name = 'IonCastPersistence-ablation-JPLD'
                model = IonCastPersistence(input_channels=total_channels, output_channels=total_channels, context_window=args.context_window, name=name)
                model = model.to(device)
                model.eval()
            else:
                if not args.model_file:
                    raise ValueError("A --model_file must be specified for testing mode when not using persistence model.")
                
                print(f'Loading model from {args.model_file}')
                model, optimizer, _, _, _, _, _, _, _, _, _, _ = load_model(args.model_file, device)
                if getattr(model, "name", "") == "SphericalFourierNeuralOperatorModel":
                    model.output_blur_sigma = 0.85
                    model.head_blend_sigma = args.head_blend_sigma
                    model.lon_blur_sigma_deg = args.lon_blur_sigma_deg
                model.eval()
                model = model.to(device)
                if use_channels_last:
                    model = model.to(memory_format=torch.channels_last)
            
            if not args.test_event_id:
                print("No --test_event_id provided. Exiting test mode.")
                return

            with torch.no_grad():
                # Initialize collections for test metrics
                test_fixed_lead_time_metrics = []
                test_fixed_lead_time_event_ids = []
                
                for event_id in args.test_event_id:
                    if event_id not in event_catalog:
                        raise ValueError(f'Event ID {event_id} not found in EventCatalog')
                    
                    event = event_catalog[event_id]
                    event_start = datetime.datetime.fromisoformat(event['date_start'])
                    event_end = datetime.datetime.fromisoformat(event['date_end'])
                    
                    # Define a data window large enough for all evaluation types
                    max_lead_time = max(args.lead_times) if args.lead_times else 0
                    buffer_start = event_start - datetime.timedelta(minutes=max_lead_time + model.context_window * args.delta_minutes)
                    
                    print(f'\n--- Preparing data for Event: {event_id} ---')
                    if model.name in ['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLinear', 'IonCastLSTM-ablation-JPLDSunMoon', 'SphericalFourierNeuralOperatorModel']:
                        # Other models use all 5 datasets
                        dataset_jpld = JPLD(os.path.join(args.data_dir, args.jpld_dir), date_start=buffer_start, date_end=event_end)
                        dataset_sunmoon = SunMoonGeometry(date_start=buffer_start, date_end=event_end, extra_time_steps=args.sun_moon_extra_time_steps)
                        dataset_celestrak = CelesTrak(os.path.join(args.data_dir, args.celestrak_file_name), date_start=buffer_start, date_end=event_end, return_as_image_size=(180, 360))
                        dataset_omniweb = OMNIWeb(os.path.join(args.data_dir, args.omniweb_dir), date_start=buffer_start, date_end=event_end, column=args.omniweb_columns, return_as_image_size=(180, 360))
                        dataset_set = SET(os.path.join(args.data_dir, args.set_file_name), date_start=buffer_start, date_end=event_end, return_as_image_size=(180, 360))
                        dataset = Sequences(datasets=[dataset_jpld, dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set], delta_minutes=args.delta_minutes, sequence_length=1) # sequence_length doesn't matter here
                    elif model.name == 'IonCastLSTMSDO':
                        # SDO model uses only JPLD + SunMoonGeometry + SDOCore
                        dataset_jpld = JPLD(os.path.join(args.data_dir, args.jpld_dir), date_start=buffer_start, date_end=event_end)
                        dataset_sunmoon = SunMoonGeometry(date_start=buffer_start, date_end=event_end, extra_time_steps=args.sun_moon_extra_time_steps)
                        dataset_sdocore = SDOCore(os.path.join(args.data_dir, args.sdocore_file_name), date_start=buffer_start, date_end=event_end)
                        dataset = Sequences(datasets=[dataset_jpld, dataset_sunmoon, dataset_sdocore], delta_minutes=args.delta_minutes, sequence_length=1) # sequence_length doesn't matter here
                    elif model.name in ['IonCastLSTM-ablation-JPLD', 'IonCastLinear-ablation-JPLD', 'IonCastPersistence-ablation-JPLD']:
                        # Ablation models use only JPLD dataset
                        dataset_jpld = JPLD(os.path.join(args.data_dir, args.jpld_dir), date_start=buffer_start, date_end=event_end)
                        dataset = Sequences(datasets=[dataset_jpld], delta_minutes=args.delta_minutes, sequence_length=1) # sequence_length doesn't matter here
                    else:
                        raise ValueError(f'Unsupported model: {model.name}')

                    file_name_prefix = os.path.join(args.target_dir, 'test')

                    if args.eval_mode in ['long_horizon', 'all']:
                        save_video = True
                        save_numpy = True
                        eval_forecast_long_horizon(model, dataset, event_catalog, event_id, file_name_prefix, save_video, save_numpy, save_video, args)

                    if args.eval_mode in ['fixed_lead_time', 'all']:
                        save_video = True
                        save_numpy = True
                        lead_time_errors_test, event_id_returned_test = eval_forecast_fixed_lead_time(model, dataset, event_catalog, event_id, args.lead_times, file_name_prefix, save_video, save_numpy, save_video, args)
                        test_fixed_lead_time_metrics.append(lead_time_errors_test)
                        test_fixed_lead_time_event_ids.append(event_id_returned_test)

                    # Force cleanup
                    del dataset_jpld
                    if model.name in ['IonCastConvLSTM', 'IonCastLSTM', 'IonCastLinear', 'IonCastLSTM-ablation-JPLDSunMoon', 'SphericalFourierNeuralOperatorModel']:
                        del dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set
                    elif model.name == 'IonCastLSTMSDO':
                        del dataset_sunmoon, dataset_sdocore
                    del dataset
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Aggregate and plot fixed-lead-time metrics for test events
                if test_fixed_lead_time_metrics:
                    plot_file_prefix_test = os.path.join(args.target_dir, 'test')
                    aggregate_and_plot_fixed_lead_time_metrics(test_fixed_lead_time_metrics, test_fixed_lead_time_event_ids, plot_file_prefix_test)

        else:
            raise ValueError('Unknown mode: {}'.format(args.mode))

        # Upload any remaining plots from best_model directory
        if args.mode == 'train' and wandb is not None and args.wandb_mode != 'disabled':
            best_model_dir = os.path.join(args.target_dir, 'best_model')
            if os.path.exists(best_model_dir):
                png_files = glob.glob(os.path.join(best_model_dir, '*.png'))
                for png_file in png_files:
                    try:
                        plot_name = f"best_model/{os.path.splitext(os.path.basename(png_file))[0]}"
                        wandb.log({f"plots/{plot_name}": wandb.Image(png_file)})
                    except Exception as e:
                        print(f"Warning: Could not upload best model plot {png_file}: {e}")
        
        if wandb is not None and args.wandb_mode != 'disabled':
            wandb.finish()
        
        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()


# Example
# python run.py --data_dir /disk2-ssd-8tb/data/2025-hl-ionosphere --mode train --target_dir ./train-1 --num_workers 4 --batch_size 4 --model_type IonCastConvLSTM --epochs 2 --learning_rate 1e-3 --weight_decay 0.0 --context_window 4 --prediction_window 4 --num_evals 4 --date_start 2023-07-01T00:00:00 --date_end 2023-08-01T00:00:00

