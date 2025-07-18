import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import glob


from util import Tee
from util import set_random_seed
from models import VAE1
from datasets import JPLDGIMDataset


matplotlib.use('Agg')


def plot_global_ionosphere_map(ax, image, cmap='jet', vmin=None, vmax=None, title=None):
    """
    Plots a 180x360 global ionosphere image on a given Cartopy axes.
    
    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Axes with a Cartopy projection.
        image (np.ndarray): 2D numpy array with shape (180, 360), representing lat [-90, 90], lon [-180, 180].
        cmap (str): Colormap to use for imshow.
        vmin (float): Minimum value for colormap normalization.
        vmax (float): Maximum value for colormap normalization.
    """
    if image.shape != (180, 360):
        raise ValueError("Input image must have shape (180, 360) corresponding to lat [-90, 90], lon [-180, 180].")

    im = ax.imshow(
        image,
        extent=[-180, 180, -90, 90],
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    
    ax.coastlines()
    if title is not None:
        ax.set_title(title, fontsize=12)

    return im


def save_gim_plot(gim, file_name, cmap='jet', vmin=None, vmax=None, title=None):
    """
    Plots a single 180x360 global ionosphere image using GridSpec,
    with a colorbar aligned to the full height of the imshow map.
    """
    print(f'Plotting GIM to {file_name}')
    
    if gim.shape != (180, 360):
        raise ValueError("Input image must have shape (180, 360) corresponding to lat [-90, 90], lon [-180, 180].")
    
    fig = plt.figure(figsize=(10, 5))
    
    # GridSpec: one row, two columns
    gs = fig.add_gridspec(
        1, 2, width_ratios=[20, 1], wspace=0.05,
        left=0.05, right=0.98, top=0.9, bottom=0.1
    )
    
    # Main plot
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    im = plot_global_ionosphere_map(ax, gim, cmap=cmap, vmin=vmin, vmax=vmax, title=title)
    
    # Colorbar axis — NOT a projection axis
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("TEC (TECU)")
    
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    plt.close()

# Save a sequence of GIM images as a video, exactly the same as save_gim_plot but for a sequence of images
def save_gim_video(gim_sequence, file_name, cmap='jet', vmin=None, vmax=None, titles=None, fps=2):
    print(f'Saving GIM video to {file_name}')

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05, left=0.05, right=0.98, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = None

    def update(frame):
        nonlocal cbar
        ax.clear()
        im = plot_global_ionosphere_map(ax, gim_sequence[frame], cmap=cmap, vmin=vmin, vmax=vmax, title=titles[frame] if titles else None)
        if cbar is None:
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("TEC (TECU)")
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(gim_sequence), blit=True, interval=1000/fps)
    ani.save(file_name, dpi=150, writer='ffmpeg')
    plt.close()


def save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, eval_data, file_name):
    print('Saving model to {}'.format(file_name))
    if isinstance(model, VAE1):
        checkpoint = {
            'model': 'VAE1',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'model_z_dim': model.z_dim,
            'eval_data': eval_data
        }
    else:
        raise ValueError('Unknown model type: {}'.format(model))
    torch.save(checkpoint, file_name)


def load_model(file_name, device):
    checkpoint = torch.load(file_name, weights_only=False)
    if checkpoint['model'] == 'VAE1':
        model_z_dim = checkpoint['model_z_dim']
        model = VAE1(z_dim=model_z_dim)
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
    eval_data = checkpoint['eval_data']
    return model, optimizer, epoch, iteration, train_losses, valid_losses, eval_data


def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--jpld_filename', type=str, default='jpld_gim/parquet/jpld_gim_201005130000_202407312345.parquet', help='JPLD GIM dataset directory')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    # parser.add_argument('--date_start', type=str, default='2010-05-13T00:00:00', help='Start date')
    # parser.add_argument('--date_end', type=str, default='2024-08-01T00:00:00', help='End date')
    parser.add_argument('--date_start', type=str, default='2024-07-01T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-07-03T00:00:00', help='End date')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')    
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--model_type', type=str, choices=['VAE1'], default='VAE1', help='Type of model to use')
    parser.add_argument('--valid_proportion', type=float, default=0.15, help='Proportion of data to use for validation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_evals', type=int, default=4, help='Number of samples for evaluation')

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    log_file = os.path.join(args.target_dir, 'log.txt')

    set_random_seed(args.seed)
    device = torch.device(args.device)

    with Tee(log_file):
        print(description)
        print('Log file:', log_file)
        print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
        print('Config:')
        pprint.pprint(vars(args), depth=2, width=50)

        start_time = datetime.datetime.now()
        print('Start time: {}'.format(start_time))

        if args.mode == 'train':
            print('Training mode selected.')

            if args.batch_size < args.num_evals:
                print(f'Warning: Batch size {args.batch_size} is less than num_evals {args.num_evals}. Using the batch size for num_evals.')
                args.num_evals = args.batch_size

            date_start = datetime.datetime.fromisoformat(args.date_start)
            date_end = datetime.datetime.fromisoformat(args.date_end)

            dataset_jpld_filename = os.path.join(args.data_dir, args.jpld_filename)
            dataset_jpld = JPLDGIMDataset(dataset_jpld_filename, date_start=date_start, date_end=date_end, normalize=True)

            valid_size = int(args.valid_proportion * len(dataset_jpld))
            train_size = len(dataset_jpld) - valid_size
            dataset_train, dataset_valid = random_split(dataset_jpld, [train_size, valid_size])

            print('\nTrain size: {:,}'.format(len(dataset_train)))
            print('Valid size: {:,}'.format(len(dataset_valid)))

            train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = glob.glob('{}/epoch-*-model.pth'.format(args.target_dir))
            if len(model_files) > 0:
                model_files.sort()
                model_file = model_files[-1]
                print('Resuming training from model file: {}'.format(model_file))
                model, optimizer, epoch, iteration, train_losses, valid_losses, eval_data = load_model(model_file, device)
                epoch_start = epoch + 1
                iteration = iteration + 1
                print('Next epoch    : {:,}'.format(epoch_start+1))
                print('Next iteration: {:,}'.format(iteration+1))
            else:
                print('Creating new model')
                if args.model_type == 'VAE1':
                    model = VAE1(z_dim=512, sigma_vae=False)
                else:
                    raise ValueError('Unknown model type: {}'.format(args.model_type))

                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                iteration = 0
                epoch_start = 0
                train_losses = []
                valid_losses = []
                eval_data = {
                    'eval_reconstructions': None, # numpy array of shape (num_evals, num_epochs, 180, 360)
                    'eval_reconstructions_originals': None, # numpy array of shape (num_evals, 180, 360)
                    'eval_reconstructions_dates': None, # list of dates of length num_evals
                    'eval_samples': None # numpy array of shape (num_evals, num_epochs, 180, 360)
                }
                model = model.to(device)

            model.train()

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('\nNumber of parameters: {:,}\n'.format(num_params))
            
            for epoch in range(epoch_start, args.epochs):
                print('\n*** Epoch {:,}/{:,} started'.format(epoch+1, args.epochs))
                print('*** Training')
                # Training
                with tqdm(total=len(train_loader)) as pbar:
                    for i, batch in enumerate(train_loader):

                        jpld, _ = batch
                        jpld = jpld.to(device)

                        optimizer.zero_grad()
                        loss = model.loss(jpld)
                        loss.backward()

                        optimizer.step()

                        iteration += 1

                        train_losses.append((iteration, float(loss)))
                        pbar.set_description(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}')
                        pbar.update(1)

                # Validation
                print('*** Validation')
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for jpld, _ in valid_loader:
                        jpld = jpld.to(device)
                        loss = model.loss(jpld)
                        valid_loss += loss.item()
                valid_loss /= len(valid_loader)
                valid_losses.append((iteration, valid_loss))
                print(f'Validation Loss: {valid_loss:.4f}')

                file_name_prefix = f'epoch-{epoch + 1:02d}-'

                # Save model
                model_file = os.path.join(args.target_dir, f'{file_name_prefix}model.pth')
                save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, eval_data, model_file)

                # Plot losses
                plot_file = os.path.join(args.target_dir, f'{file_name_prefix}loss.pdf')
                print(f'Saving plot to {plot_file}')
                plt.figure(figsize=(10, 5))
                plt.plot(*zip(*train_losses), label='Training')
                plt.plot(*zip(*valid_losses), label='Validation')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.grid(True)
                plt.legend()
                plt.savefig(plot_file)
                plt.close()

                # Plot model outputs
                model.eval()
                with torch.no_grad():
                    num_evals = args.num_evals
                    # Set random seed for reproducibility of evaluation samples across epochs
                    rng_state = torch.get_rng_state()
                    torch.manual_seed(args.seed)

                    # Reconstruct a batch from the validation set
                    jpld_orig, jpld_orig_dates = next(iter(valid_loader))
                    jpld_orig = jpld_orig[:num_evals]
                    jpld_orig_dates = jpld_orig_dates[:num_evals]

                    jpld_orig = jpld_orig.to(device)
                    jpld_recon, _, _ = model.forward(jpld_orig)
                    jpld_orig_unnormalized = JPLDGIMDataset.unnormalize(jpld_orig)
                    jpld_recon_unnormalized = JPLDGIMDataset.unnormalize(jpld_recon)

                    # Sample a batch from the model
                    jpld_sample = model.sample(n=num_evals)
                    jpld_sample_unnormalized = JPLDGIMDataset.unnormalize(jpld_sample)
                    jpld_sample_unnormalized = jpld_sample_unnormalized.clamp(0, 100)
                    torch.set_rng_state(rng_state)
                    # Resume with the original random state

                    if eval_data['eval_reconstructions_originals'] is None:
                        eval_data['eval_reconstructions_originals'] = jpld_orig_unnormalized.cpu().numpy()
                        eval_data['eval_reconstructions_dates'] = jpld_orig_dates

                    # eval_data = {
                    #     'eval_reconstructions': None, # numpy array of shape (num_epochs, num_evals, 1, 180, 360)
                    #     'eval_reconstructions_originals': None, # numpy array of shape (num_evals, 1, 180, 360)
                    #     'eval_reconstructions_dates': None, # list of dates of length num_evals
                    #     'eval_samples': None # numpy array of shape (num_epochs, num_evals, 1, 180, 360)
                    # }

                    if eval_data['eval_reconstructions'] is None:
                        eval_data['eval_reconstructions'] = jpld_recon_unnormalized.cpu().numpy().reshape(1, num_evals, 1, 180, 360) # first dimension is the epoch
                        eval_data['eval_samples'] = jpld_sample_unnormalized.cpu().numpy().reshape(1, num_evals, 1, 180, 360) # first dimension is the epoch
                    else:
                        eval_data['eval_reconstructions'] = np.concatenate((eval_data['eval_reconstructions'], jpld_recon_unnormalized.cpu().numpy().reshape(1, num_evals, 1, 180, 360)), axis=0)
                        eval_data['eval_samples'] = np.concatenate((eval_data['eval_samples'], jpld_sample.cpu().numpy().reshape(1, num_evals, 1, 180, 360)), axis=0)

                    # Save plots
                    for i in range(num_evals):
                        date = jpld_orig_dates[i]
                        date_str = datetime.datetime.fromisoformat(date).strftime('%Y-%m-%d %H:%M:%S')

                        recon_original_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-original-{i+1:02d}.pdf')
                        save_gim_plot(jpld_orig_unnormalized[i][0].cpu().numpy(), recon_original_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str}')

                        recon_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-{i+1:02d}.pdf')
                        save_gim_plot(jpld_recon_unnormalized[i][0].cpu().numpy(), recon_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str} (Reconstruction)')

                        sample_file = os.path.join(args.target_dir, f'{file_name_prefix}sample-{i+1:02d}.pdf')
                        save_gim_plot(jpld_sample_unnormalized[i][0].cpu().numpy(), sample_file, vmin=0, vmax=100, title='JPLD GIM TEC (Sampled from model)')

                        # Save a video of the reconstructions for this evaluation
                        recon_video_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-{i+1:02d}.mp4')
                        save_gim_video(
                            eval_data['eval_reconstructions'][:, i, 0, :, :],
                            recon_video_file,
                            vmin=0, vmax=100,
                            titles=[f'JPLD GIM TEC, {date_str} (Reconstruction), Epoch {e+1}' for e in range(eval_data['eval_reconstructions'].shape[0])])
                        
                        # Save a video of the samples for this evaluation
                        sample_video_file = os.path.join(args.target_dir, f'{file_name_prefix}sample-{i+1:02d}.mp4')
                        save_gim_video(
                            eval_data['eval_samples'][:, i, 0, :, :],
                            sample_video_file,
                            vmin=0, vmax=100,
                            titles=[f'JPLD GIM TEC (Sampled from model), Epoch {e+1}' for e in range(eval_data['eval_samples'].shape[0])])

        elif args.mode == 'test':
            raise NotImplementedError("Testing mode is not implemented yet.")

        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()


# Example
# python run.py --data_dir /disk2-ssd-8tb/data/2025-hl-ionosphere --mode train --target_dir ./train-1 --num_workers 4 --batch_size 4
