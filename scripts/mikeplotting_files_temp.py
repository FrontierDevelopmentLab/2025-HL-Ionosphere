import os
import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
import argparse

from dataset_jpld import JPLD
from dataset_sequences import Sequences
from dataset_sunmoongeometry import SunMoonGeometry  # Uncomment if you use this dataset!
from model_sphericalFNO import SphericalFourierNeuralOperatorModel

def print_model_debug_info(checkpoint, x_context=None, model=None, func=''):
    print("\n========== DEBUG INFO [{}] ==========".format(func))
    print("Checkpoint: model_in_channels:", checkpoint.get('model_in_channels'))
    if 'model_state_dict' in checkpoint:
        in_proj_keys = [k for k in checkpoint['model_state_dict'] if 'in_proj' in k]
        print("Checkpoint state_dict in_proj keys:", in_proj_keys)
        if 'in_proj.weight' in checkpoint['model_state_dict']:
            print("Checkpoint in_proj.weight shape:", checkpoint['model_state_dict']['in_proj.weight'].shape)
    if x_context is not None:
        print("x_context.shape:", x_context.shape)
    if model is not None and hasattr(model, 'in_proj'):
        print("Model in_proj.weight shape:", model.in_proj.weight.shape)
    print("====================================\n")

def load_model_config_from_checkpoint(checkpoint, prediction_window=None):
    config = dict(
        in_channels=checkpoint['model_in_channels'],
        trunk_width=checkpoint['model_trunk_width'],
        trunk_depth=checkpoint['model_trunk_depth'],
        modes_lat=checkpoint['model_modes_lat'],
        modes_lon=checkpoint['model_modes_lon'],
        aux_dim=checkpoint['model_aux_dim'],
        tasks=checkpoint['model_tasks'],
        out_shapes=checkpoint['model_out_shapes'],
        probabilistic=checkpoint['model_probabilistic'],
        dropout=checkpoint['model_dropout'],
        mc_dropout=checkpoint['model_mc_dropout'],
        add_posenc=checkpoint.get('model_add_posenc', False)
    )
    if prediction_window is not None and isinstance(config['out_shapes'], dict):
        for k in config['out_shapes']:
            if isinstance(config['out_shapes'][k], tuple) and config['out_shapes'][k][1] == "grid":
                config['out_shapes'][k] = (prediction_window, "grid")
    return config

def animate_n_step_predictions(
    target_dir, data_dir,
    model_type='SphericalFourierNeuralOperator',
    context_window=12, idx=0,
    total_steps=96, out_file='prediction_vs_target_tN.mp4',
    prediction_window=4,  # Must be >= lead_index+1
    lead_index=3          # 0-based: 0=t+1, 1=t+2, 2=t+3, 3=t+4
):
    import matplotlib.animation as animation

    # --- Model Loading ---
    ckpts = sorted(glob.glob(os.path.join(target_dir, 'epoch-*-model.pth')))
    assert ckpts, "No checkpoints found!"
    checkpoint = torch.load(ckpts[-1], map_location='cpu')
    model_config = load_model_config_from_checkpoint(checkpoint, prediction_window=prediction_window)
    expected_in_channels = model_config['in_channels']

    model = SphericalFourierNeuralOperatorModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Data Loading ---
    jpld = JPLD(os.path.join(data_dir, 'jpld/webdataset'))
    sunmoon = SunMoonGeometry(date_start=jpld.date_start, date_end=jpld.date_end, delta_minutes=15)
    seq_dataset = Sequences([jpld, sunmoon], delta_minutes=15, sequence_length=context_window + total_steps)
    batch = seq_dataset[idx]
    jpld_seq, sunmoon_seq = batch[:2]
    actual_in_channels = jpld_seq.shape[1] + sunmoon_seq.shape[1]
    print(f"[ANIMATE N-STEP] jpld_seq.shape = {jpld_seq.shape}, sunmoon_seq.shape = {sunmoon_seq.shape}")
    print(f"[ANIMATE N-STEP] Expected in_channels: {expected_in_channels}, got {actual_in_channels}")
    if actual_in_channels != expected_in_channels:
        raise RuntimeError(
            f"Channel mismatch: Data has {actual_in_channels}, but model expects {expected_in_channels}.\n"
            f"jpld features: {jpld_seq.shape[1]}, sunmoon features: {sunmoon_seq.shape[1]}"
        )

    x_input = torch.cat([jpld_seq, sunmoon_seq], dim=1)
    x_context = x_input[:context_window].clone()
    y_seq = jpld_seq[context_window:context_window+total_steps].clone()  # [total_steps, 1, 180, 360]

    preds = []
    with torch.no_grad():
        x_curr = x_context.clone()
        for t in range(total_steps):
            x_in = x_curr[-1].unsqueeze(0)
            out = model(x_in)
            pred, logvar = out['vtec']
            if pred.shape[1] <= lead_index:
                preds.append(np.full_like(pred[0, 0].cpu().numpy(), np.nan))
            else:
                pred_frame = pred[0, lead_index]
                preds.append(pred_frame.cpu().numpy())
            if t+1 < total_steps:
                # --- FIX HERE ---
                next_jpld = y_seq[t, :, :, :].unsqueeze(0)  # [1, 1, 180, 360]
                next_sunmoon = sunmoon_seq[context_window + t, :, :, :].unsqueeze(0)  # [1, 18, 180, 360]
                next_full = torch.cat([next_jpld, next_sunmoon], dim=1)  # [1, 19, 180, 360]
                x_curr = torch.cat([x_curr[1:], next_full], dim=0)

    preds = np.array(preds)    # [T, 180, 360]
    gt = y_seq[:, 0].cpu().numpy()  # [T, 180, 360]

    # --- UNNORMALIZE ---
    vtec_preds = JPLD.unnormalize(torch.from_numpy(preds)).numpy()
    vtec_gt = JPLD.unnormalize(torch.from_numpy(gt)).numpy()
    # --- vmin/vmax for colorbars
    vmin, vmax = vtec_gt.min(), vtec_gt.max()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ims = []
    for t in range(total_steps):
        im_pred = axs[0].imshow(vtec_preds[t], animated=True, cmap='jet', vmin=vmin, vmax=vmax)
        im_gt   = axs[1].imshow(vtec_gt[t], animated=True, cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title(f'Prediction (t+{lead_index+1})')
        axs[1].set_title(f'Target (Ground Truth)')
        ims.append([im_pred, im_gt])
    fig.colorbar(ims[0][0], ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(ims[0][1], ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
    ani.save(out_file)
    plt.close(fig)
    print(f'Animation saved to {out_file}')


def plot_loss_from_checkpoint(target_dir):
    ckpts = sorted(glob.glob(os.path.join(target_dir, 'epoch-*-model.pth')))
    if not ckpts:
        print("No checkpoints found!")
        return
    checkpoint = torch.load(ckpts[-1], map_location='cpu')
    model_config = load_model_config_from_checkpoint(checkpoint)
    print("[plot_loss_from_checkpoint] Model config from checkpoint:")
    for k, v in model_config.items():
        print(f"{k}: {v}")
    model = SphericalFourierNeuralOperatorModel(**model_config)
    print_model_debug_info(checkpoint, None, model, func='plot_loss_from_checkpoint')
    print("Checkpoint heads.vtec.weight shape:", checkpoint['model_state_dict']['heads.vtec.weight'].shape)
    print("Current model heads.vtec.weight shape:", model.heads['vtec'].weight.shape)
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    plt.figure(figsize=(10, 5))
    plt.plot(*zip(*train_losses), label='Training')
    plt.plot(*zip(*valid_losses), label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss Curves')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pred_vs_target(target_dir, data_dir, context_window=12, prediction_window=8, idx=0):
    ckpts = sorted(glob.glob(os.path.join(target_dir, 'epoch-*-model.pth')))
    assert ckpts, "No checkpoints found!"
    checkpoint = torch.load(ckpts[-1], map_location='cpu')
    model_config = load_model_config_from_checkpoint(checkpoint, prediction_window=prediction_window)
    expected_in_channels = model_config['in_channels']

    jpld = JPLD(os.path.join(data_dir, 'jpld/webdataset'))
    sunmoon = SunMoonGeometry(date_start=jpld.date_start, date_end=jpld.date_end, delta_minutes=15)
    sequence_length = context_window + prediction_window
    seq_dataset = Sequences([jpld, sunmoon], delta_minutes=15, sequence_length=sequence_length)

    batch = seq_dataset[idx]
    jpld_seq, sunmoon_seq = batch[:2]
    actual_in_channels = jpld_seq.shape[1] + sunmoon_seq.shape[1]
    print(f"jpld_seq.shape = {jpld_seq.shape}, sunmoon_seq.shape = {sunmoon_seq.shape}")
    print(f"Expected in_channels from checkpoint: {expected_in_channels}, got {actual_in_channels}")
    if actual_in_channels != expected_in_channels:
        raise RuntimeError(
            f"Channel mismatch: Data has {actual_in_channels}, but model expects {expected_in_channels}.\n"
            f"jpld features: {jpld_seq.shape[1]}, sunmoon features: {sunmoon_seq.shape[1]}"
        )

    x_input = torch.cat([jpld_seq, sunmoon_seq], dim=1)
    x_context = x_input[:context_window].clone()
    y_seq = jpld_seq[context_window:context_window+prediction_window].clone()  # target

    model = SphericalFourierNeuralOperatorModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds = []
    x_curr = x_context.clone()
    with torch.no_grad():
        for step in range(prediction_window):
            x_in = x_curr[-1].unsqueeze(0)
            out = model(x_in)
            pred, logvar = out['vtec']
            pred_frame = pred[0, 0]
            preds.append(pred_frame.cpu().numpy())
            if step + 1 < prediction_window:
                next_jpld = pred_frame.unsqueeze(0).unsqueeze(1)  # [1,1,180,360]
                next_sunmoon = sunmoon_seq[context_window + step, :, :, :].unsqueeze(0)  # [1,18,180,360]
                next_full = torch.cat([next_jpld, next_sunmoon], dim=1)  # [1,19,180,360]
                x_curr = torch.cat([x_curr[1:], next_full], dim=0)

    preds = np.array(preds)    # [prediction_window, 180, 360]
    gt = y_seq[:, 0].cpu().numpy()  # [prediction_window, 180, 360]

    # --- UNNORMALIZE ---
    vtec_preds = JPLD.unnormalize(torch.from_numpy(preds)).numpy()
    vtec_gt = JPLD.unnormalize(torch.from_numpy(gt)).numpy()

    # --- vmin/vmax in TECU for good colorbars
    vmin, vmax = vtec_gt.min(), vtec_gt.max()
    print("Global vmin/vmax:", vmin, vmax)

    for t in range(prediction_window):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Prediction t+{t+1}')
        plt.imshow(vtec_preds[t], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(label='VTEC (TECU)')
        plt.subplot(1, 2, 2)
        plt.title(f'Target t+{t+1}')
        plt.imshow(vtec_gt[t], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(label='VTEC (TECU)')
        plt.tight_layout()
        plt.savefig(f"prediction_vs_target_{t:02d}.png")
        plt.close()

def animate_prediction(
    target_dir, data_dir,
    model_type='SphericalFourierNeuralOperator',
    context_window=12, prediction_window=8, idx=0,
    total_steps=96, out_file='gim_pred_vs_gt.mp4',
    lead_index=0
):
    import matplotlib.animation as animation
    import pandas as pd
    ckpts = sorted(glob.glob(os.path.join(target_dir, 'epoch-*-model.pth')))
    checkpoint = torch.load(ckpts[-1], map_location='cpu')
    model_config = load_model_config_from_checkpoint(checkpoint, prediction_window=prediction_window)
    expected_in_channels = model_config['in_channels']

    jpld = JPLD(os.path.join(data_dir, 'jpld/webdataset'))
    sunmoon = SunMoonGeometry(date_start=jpld.date_start, date_end=jpld.date_end, delta_minutes=15)
    seq_dataset = Sequences([jpld, sunmoon], delta_minutes=15, sequence_length=context_window + total_steps)
    batch = seq_dataset[idx]
    jpld_seq, sunmoon_seq = batch[:2]
    actual_in_channels = jpld_seq.shape[1] + sunmoon_seq.shape[1]
    print(f"[ANIMATE] jpld_seq.shape = {jpld_seq.shape}, sunmoon_seq.shape = {sunmoon_seq.shape}")
    print(f"[ANIMATE] Expected in_channels: {expected_in_channels}, got {actual_in_channels}")
    if actual_in_channels != expected_in_channels:
        raise RuntimeError(
            f"Channel mismatch: Data has {actual_in_channels}, but model expects {expected_in_channels}.\n"
            f"jpld features: {jpld_seq.shape[1]}, sunmoon features: {sunmoon_seq.shape[1]}"
        )
    x_input = torch.cat([jpld_seq, sunmoon_seq], dim=1)
    x_context = x_input[:context_window].clone()
    y_seq = jpld_seq[context_window:context_window+total_steps].clone()  # target

    model = SphericalFourierNeuralOperatorModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds = []
    x_curr = x_context.clone()
    with torch.no_grad():
        for step in range(prediction_window):
            x_in = x_curr[-1].unsqueeze(0)
            out = model(x_in)
            pred, logvar = out['vtec']
            pred_frame = pred[0, 0]
            preds.append(pred_frame.cpu().numpy())
            if step + 1 < prediction_window:
                next_jpld = pred_frame.unsqueeze(0).unsqueeze(1)  # [1,1,180,360]
                next_sunmoon = sunmoon_seq[context_window + step, :, :, :].unsqueeze(0)  # [1,18,180,360]
                next_full = torch.cat([next_jpld, next_sunmoon], dim=1)  # [1,19,180,360]
                x_curr = torch.cat([x_curr[1:], next_full], dim=0)

    preds = np.array(preds)    # [T, 180, 360]
    gt = y_seq[:, 0].cpu().numpy()  # [T, 180, 360]

    # --- UNNORMALIZE ---
    vtec_preds = JPLD.unnormalize(preds)
    vtec_gt = JPLD.unnormalize(gt)

    # --- vmin/vmax in TECU for good colorbars
    vmin, vmax = vtec_gt.min(), vtec_gt.max()
    print("Global vmin/vmax:", vmin, vmax)

    for t in range(prediction_window):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Prediction t+{t+1}')
        plt.imshow(vtec_preds[t], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(label='VTEC (TECU)')
        plt.subplot(1, 2, 2)
        plt.title(f'Target t+{t+1}')
        plt.imshow(vtec_gt[t], cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(label='VTEC (TECU)')
        plt.tight_layout()
        plt.savefig(f"prediction_vs_target_{t:02d}.png")
        plt.close()
    print(f'Animation saved to {out_file}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True, help='Path to training outputs/checkpoints')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ionosphere-data')
    parser.add_argument('--plot_loss', action='store_true', help='Plot loss curves')
    parser.add_argument('--plot_pred', action='store_true', help='Show prediction vs. ground truth images')
    parser.add_argument('--animate_pred', action='store_true', help='Save animation of prediction vs. ground truth')
    parser.add_argument('--animate_one_step_pred', action='store_true', help='Animate one-step-ahead predictions for 24 hours')
    parser.add_argument('--context_window', type=int, default=12)
    parser.add_argument('--prediction_window', type=int, default=4)
    parser.add_argument('--idx', type=int, default=0, help='Index of the sequence to visualize')
    parser.add_argument('--animate_n_step_pred', action='store_true', help='Animate t+N step-ahead predictions for 24 hours')
    parser.add_argument('--lead_index', type=int, default=0, help='Index for N-step ahead prediction (0=t+1, 1=t+2, ...)')

    args = parser.parse_args()

    if args.plot_loss:
        plot_loss_from_checkpoint(args.target_dir)
    if args.plot_pred:
        plot_pred_vs_target(
            args.target_dir, args.data_dir,
            context_window=args.context_window,
            prediction_window=args.prediction_window,
            idx=args.idx
        )
    if args.animate_pred:
        animate_prediction(
            args.target_dir, args.data_dir,
            context_window=args.context_window,
            prediction_window=args.prediction_window,
            idx=args.idx
        )
    if args.animate_n_step_pred:
        animate_n_step_predictions(
            args.target_dir, args.data_dir,
            context_window=args.context_window,
            prediction_window=args.prediction_window,
            idx=args.idx,
            total_steps=96,
            out_file=f'prediction_vs_target_t{args.lead_index+1}.mp4',
            lead_index=args.lead_index
        )

# --- Example command to run ---
# python explore_results.py --target_dir /mnt/disks/disk-main-data-1/sfno3/ --data_dir /mnt/disks/disk-main-data-1/ --plot_pred --context_window 12 --idx 0