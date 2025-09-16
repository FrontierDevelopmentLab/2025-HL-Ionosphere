
# physics_losses.py

import torch

# --- Optional W&B (safe import) ---
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


# -------------------------
# Component losses
# -------------------------
def continuity_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Penalize large spatial gradients (finite differences in latitude & longitude).
    pred: [B, C, H, W]
    """
    dy = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
    dx = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
    return dy + dx


def _slot_index_from_time(dt, slots_per_day: int) -> int:
    """Map a datetime to a discrete time-of-day slot [0, slots_per_day-1]."""
    minutes = dt.hour * 60 + dt.minute
    slot = int(round((minutes / 1440.0) * slots_per_day))
    if slot == slots_per_day:
        slot = 0
    return slot


def _circular_distance(a: torch.Tensor, b: torch.Tensor, period: int) -> torch.Tensor:
    """
    Pairwise circular distance between integer slots in [0, period-1].
    a: [B], b: [B] -> returns [B, B]
    """
    diff = torch.abs(a[:, None] - b[None, :]).float()
    return torch.minimum(diff, period - diff)


def periodicity_loss(
    pred: torch.Tensor,
    times=None,
    slots_per_day: int = 96,
    bandwidth_slots: float = 2.0,
    exclude_self: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Soft (kernel-weighted) phase consistency across the batch.
    For each sample i, form a weighted average of other samples j based on the
    circular time-of-day distance, then penalize (pred_i - weighted_mean_i)^2.

    Works even when every sample is at a different slot.

    Args:
        pred: [B, C, H, W] predictions for the current target step per batch sample.
        times: list/sequence of python datetimes of length B (target timestamps).
        slots_per_day: number of discrete time-of-day slots (e.g., 96 for 15-min cadence).
        bandwidth_slots: Gaussian kernel width in slots (larger = smoother/less strict).
        exclude_self: if True, zero-out self-weight so it compares to peers only.
    """
    if times is None or len(times) == 0 or pred.dim() != 4:
        return torch.zeros((), device=pred.device)

    B = pred.shape[0]
    if B != len(times):
        return torch.zeros((), device=pred.device)

    # slot indices
    with torch.no_grad():
        slots = torch.tensor(
            [_slot_index_from_time(t, slots_per_day) for t in times],
            device=pred.device,
            dtype=torch.long,
        )

    # pairwise circular distances in slot space
    D = _circular_distance(slots, slots, slots_per_day)  # [B, B]

    # Gaussian kernel on circular distance
    sigma2 = max(bandwidth_slots, 1e-6) ** 2
    W = torch.exp(- (D * D) / (2.0 * sigma2))  # [B, B]

    if exclude_self:
        W.fill_diagonal_(0.0)

    # row-normalize weights (avoid div-by-zero)
    Z = W.sum(dim=1, keepdim=True)  # [B, 1]
    Wn = W / (Z + eps)              # [B, B]

    # weighted neighbor mean for each sample i
    # pred: [B, C, H, W] -> expand dims to batch-matmul
    # We'll reshape to [B, C*H*W], apply Wn, then reshape back.
    B, C, H, W_ = pred.shape
    pred_flat = pred.view(B, C * H * W_)
    neighbor_mean_flat = Wn @ pred_flat                     # [B, C*H*W]
    neighbor_mean = neighbor_mean_flat.view(B, C, H, W_)    # [B, C, H, W]

    # squared error to neighbor mean
    se = (pred - neighbor_mean) ** 2
    loss = se.mean()

    # W&B diagnostics (how much neighbor mass is nonzero)
    if _WANDB_AVAILABLE:
        try:
            # effective neighbors: avg normalized weight mass (excl self), also raw mass
            avg_raw_mass = float(Z.mean().detach())
            # fraction of rows with at least one nonzero neighbor weight
            frac_connected = float(((Z > eps).float().mean()).detach())
            wandb.log(
                {
                    "periodicity/avg_raw_neighbor_mass": avg_raw_mass,
                    "periodicity/frac_connected_rows": frac_connected,
                    "periodicity/bandwidth_slots": float(bandwidth_slots),
                },
                commit=False,
            )
        except Exception:
            pass

    return loss


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Laplacian smoothness penalty over the spatial grid."""
    lap = (
        -4 * pred
        + torch.roll(pred, 1, dims=2)
        + torch.roll(pred, -1, dims=2)
        + torch.roll(pred, 1, dims=3)
        + torch.roll(pred, -1, dims=3)
    )
    return torch.mean(torch.abs(lap))


# -------------------------
# Loss balancing hyperparameters
# -------------------------
LAMBDA_CONTINUITY = 0.05
LAMBDA_PERIODICITY = 0.05
LAMBDA_SMOOTHNESS = 0.05


# -------------------------
# Top-level combined loss
# -------------------------
def geophysics_informed_loss(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    iteration: int = None,
    *,
    times=None,
    slots_per_day: int = 96,
    tol_slots: int = 0,           # (kept for API compat; not used by soft kernel)
    bandwidth_slots: float = 2.0, # new: kernel width in slots
) -> torch.Tensor:
    """
    Combined data term (probabilistic NLL) + physics priors.

    Args:
        pred:  [B, C, H, W]
        logvar:[B, C, H, W] (broadcastable to pred)
        target:[B, C, H, W]
        iteration: int for logging
        times: list of datetimes (len==B) for the target step (phase info)
        slots_per_day: slots per 24h (e.g., 96 for 15-min cadence)
        bandwidth_slots: Gaussian kernel width for phase matching
    """
    # Data term: Gaussian NLL
    logvar = torch.clamp(logvar, min=-5, max=5)
    nll = 0.5 * (logvar + ((pred - target) ** 2) / logvar.exp())
    probabilistic_loss = nll.mean()

    # Physics priors
    loss_continuity = continuity_loss(pred)
    loss_periodicity = periodicity_loss(
        pred, times=times, slots_per_day=slots_per_day, bandwidth_slots=bandwidth_slots
    )
    loss_smoothness = smoothness_loss(pred)

    loss = (
        probabilistic_loss
        + LAMBDA_CONTINUITY * loss_continuity
        + LAMBDA_PERIODICITY * loss_periodicity
        + LAMBDA_SMOOTHNESS * loss_smoothness
    )

    # Console + W&B logging every 100 iters
    if iteration is not None and (iteration % 100 == 0):
        msg = (
            f"Iter {iteration} | "
            f"Prob: {probabilistic_loss.item():.4f} | "
            f"Cont: {loss_continuity.item():.4f} | "
            f"Per: {loss_periodicity.item():.4f} | "
            f"Smooth: {loss_smoothness.item():.4f} | "
            f"Total: {loss.item():.4f}"
        )
        print(msg)

        if _WANDB_AVAILABLE:
            try:
                wandb.log(
                    {
                        "loss/total": float(loss.detach()),
                        "loss/prob": float(probabilistic_loss.detach()),
                        "loss/continuity": float(loss_continuity.detach()),
                        "loss/periodicity": float(loss_periodicity.detach()),
                        "loss/smoothness": float(loss_smoothness.detach()),
                        "train/iteration": int(iteration),
                    },
                    step=int(iteration),
                )
            except Exception:
                pass

    return loss
