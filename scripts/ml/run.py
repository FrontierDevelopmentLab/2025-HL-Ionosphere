import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


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


def plot_gims(gims, file_name, titles=None):
    num_samples = gims.shape[0]

    if titles is None:
        titles = [f'GIM TEC' for _ in range(num_samples)]

    if len(titles) != num_samples:
            raise ValueError("Number of titles must match number of samples.")

    print('Plotting {} samples to {}'.format(num_samples, file_name))
    
    # find the best grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure with Cartopy projection for each subplot
    fig = plt.figure(figsize=(8 * grid_size + 1, 4 * grid_size))  # Extra width for colorbar
    
    # Create main grid for subplots
    gs = fig.add_gridspec(grid_size, grid_size + 1, width_ratios=[1] * grid_size + [0.05])
    
    ims = []  # Store image objects for colorbar
    
    for i in range(num_samples):
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size], projection=ccrs.PlateCarree())
        gim = gims[i, 0]
        print(gim.min(), gim.max(), gim.mean(), gim.std())
        im = plot_global_ionosphere_map(ax, gim, cmap='jet', vmin=0, vmax=100, title=titles[i])
        ims.append(im)
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(num_samples, grid_size * grid_size):
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size])
        ax.axis('off')
    
    # Add colorbar on the right side
    if ims:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(ims[0], cax=cbar_ax, label='TEC (TECU)')
        cbar.set_ticks([0, 20, 40, 60, 80, 100])

    print(f'Saving GIM plot to {file_name}')
    plt.tight_layout()
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    plt.close()


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
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--valid_proportion', type=float, default=0.15, help='Proportion of data to use for validation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_samples', type=int, default=9, help='Number of samples for various operations')

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


            model = VAE1(z_dim=512, sigma_vae=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            iteration = 0
            train_losses = []
            valid_losses = []
            model = model.to(device)

            model.train()

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('\nNumber of parameters: {:,}\n'.format(num_params))
            
            for epoch in range(args.epochs):
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
                # print(f'\nValidating')
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

                # sample a batch from the VAE
                model.eval()
                with torch.no_grad():
                    rng_state = torch.get_rng_state()
                    torch.manual_seed(args.seed)

                    # Reconstruct a batch from the validation set
                    jpld_orig, jpld_orig_dates = next(iter(valid_loader))
                    jpld_orig = jpld_orig.to(device)
                    jpld_recon, _, _ = model.forward(jpld_orig)
                    jpld_orig = JPLDGIMDataset.unnormalize(jpld_orig)
                    jpld_recon = JPLDGIMDataset.unnormalize(jpld_recon)

                    # Sample a batch from the model
                    jpld_sample = model.sample(n=args.num_samples)
                    jpld_sample = JPLDGIMDataset.unnormalize(jpld_sample)
                    torch.set_rng_state(rng_state)

                    print(jpld_orig.shape, jpld_recon.shape, jpld_sample.shape)
                    jpld_orig_dates = ['JPLD GIM TEC, ' + date for date in jpld_orig_dates]

                    recon_original_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-original.pdf')
                    plot_gims(jpld_orig.cpu().numpy(), recon_original_file, titles=jpld_orig_dates)

                    recon_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction.pdf')
                    plot_gims(jpld_recon.cpu().numpy(), recon_file, titles=jpld_orig_dates)

                    sample_file = os.path.join(args.target_dir, f'{file_name_prefix}sample.pdf')
                    plot_gims(jpld_sample.cpu().numpy(), sample_file)


        elif args.mode == 'test':
            raise NotImplementedError("Testing mode is not implemented yet.")

        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()


# Example
# python run.py --data_dir /disk2-ssd-8tb/data/2025-hl-ionosphere --mode train --target_dir ./train-1 --num_workers 4 --batch_size 4
