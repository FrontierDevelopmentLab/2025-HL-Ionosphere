import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import glob
import imageio

from util import Tee
from util import set_random_seed
from util import stack_as_channels
from model_vae import VAE1
from model_convlstm import IonCastConvLSTM
from dataset_jpld import JPLD
from dataset_sequences import Sequences
from dataset_union import Union
from dataset_sunmoongeometry import SunMoonGeometry
from dataset_celestrak import CelesTrak
from dataset_omniweb import OMNIWeb
from dataset_set import SET
from events import EventCatalog

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
        title (str): Title for the plot.
    """
    if image.shape != (180, 360):
        raise ValueError("Input image must have shape (180, 360), but got shape {}.".format(image.shape))

    im = ax.imshow(
        image,
        extent=[-180, 180, -90, 90],
        origin='upper',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    
    ax.coastlines()
    if title is not None:
        ax.set_title(title, fontsize=12, loc='left')

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
    # gim_sequence has shape (num_frames, 180, 360)
    print(f'Saving GIM video to {file_name}')

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05, left=0.05, right=0.92, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    cbar_ax = fig.add_subplot(gs[0, 1])
    
    # Initialize with first frame
    im = plot_global_ionosphere_map(ax, gim_sequence[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                                   title=titles[0] if titles else None)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("TEC (TECU)")

    def update(frame):
        # Update the image data instead of clearing
        new_im = plot_global_ionosphere_map(ax, gim_sequence[frame], cmap=cmap, vmin=vmin, vmax=vmax, 
                                           title=titles[frame] if titles else None)
        return [new_im]

    ani = animation.FuncAnimation(fig, update, frames=len(gim_sequence), blit=False, 
                                 interval=1000/fps, repeat=False)
    ani.save(file_name, dpi=150, writer='ffmpeg', extra_args=['-pix_fmt', 'yuv420p'])
    plt.close()


def save_gim_video_comparison(gim_sequence_top, gim_sequence_bottom, file_name, cmap='jet', vmin=None, vmax=None, 
                                       titles_top=None, titles_bottom=None, fps=2, max_frames=None):
    """
    Pre-render all frames to avoid memory accumulation during animation.
    Now includes colorbars in each frame.
    """
    # Ensure both sequences have the same length
    if len(gim_sequence_top) != len(gim_sequence_bottom):
        raise ValueError(f"Sequences must have same length: {len(gim_sequence_top)} vs {len(gim_sequence_bottom)}")
    
    if max_frames is not None:
        if max_frames <= 0 or max_frames > len(gim_sequence_top):
            raise ValueError(f"max_frames must be between 1 and {len(gim_sequence_top)}")
        gim_sequence_top = gim_sequence_top[:max_frames]
        gim_sequence_bottom = gim_sequence_bottom[:max_frames]
        if titles_top:
            titles_top = titles_top[:max_frames]
        if titles_bottom:
            titles_bottom = titles_bottom[:max_frames]

    print(f'Saving GIM video to {file_name}')
    
    # Pre-render all frames as numpy arrays
    frames = []
    for i in tqdm(range(len(gim_sequence_top)), desc="Rendering frames"):
        # Create temporary figure for this frame with colorbars
        fig_temp = plt.figure(figsize=(10.88, 10.88))
        gs = fig_temp.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], 
                                  wspace=0.05, hspace=0.15, left=0.05, right=0.92, top=0.95, bottom=0.05)
        
        # Plot frame - maps
        ax_top = fig_temp.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_bottom = fig_temp.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        
        # Plot frame - colorbar axes
        cbar_ax_top = fig_temp.add_subplot(gs[0, 1])
        cbar_ax_bottom = fig_temp.add_subplot(gs[1, 1])
        
        # Create the maps and get the image objects for colorbars
        im_top = plot_global_ionosphere_map(ax_top, gim_sequence_top[i], cmap=cmap, vmin=vmin, vmax=vmax,
                                           title=titles_top[i] if titles_top else None)
        im_bottom = plot_global_ionosphere_map(ax_bottom, gim_sequence_bottom[i], cmap=cmap, vmin=vmin, vmax=vmax,
                                              title=titles_bottom[i] if titles_bottom else None)
        
        # Add colorbars
        cbar_top = fig_temp.colorbar(im_top, cax=cbar_ax_top)
        cbar_top.set_label("TEC (TECU)")
        
        cbar_bottom = fig_temp.colorbar(im_bottom, cax=cbar_ax_bottom)
        cbar_bottom.set_label("TEC (TECU)")
        
        # Convert to array - fix deprecation warning
        fig_temp.canvas.draw()
        frame_array = np.frombuffer(fig_temp.canvas.buffer_rgba(), dtype=np.uint8)
        frame_array = frame_array.reshape(fig_temp.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        frame_array = frame_array[:, :, :3]
        frames.append(frame_array)
        
        plt.close(fig_temp)
    
    with imageio.get_writer(file_name, format='mp4', fps=fps, codec='libx264', 
                       output_params=['-pix_fmt', 'yuv420p', '-loglevel', 'error']) as writer:
        for frame in tqdm(frames, desc="Writing video   "):
            writer.append_data(frame)


def run_forecast(model, dataset, date_start, date_end, date_forecast_start, title, file_name, args):
    if not isinstance(model, (IonCastConvLSTM)):
        raise ValueError('Model must be an instance of IonCastConvLSTM')
    if date_start > date_end:
        raise ValueError('date_start must be before date_end')
    if date_forecast_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes) < date_start:
        raise ValueError('date_forecast_start must be at least context_window * delta_minutes after date_start')
    if date_forecast_start >= date_end:
        raise ValueError('date_forecast_start must be before date_end')
    # date_forecast_start must be an integer multiple of args.delta_minutes from date_start
    if (date_forecast_start - date_start).total_seconds() % (args.delta_minutes * 60) != 0:
        raise ValueError('date_forecast_start must be an integer multiple of args.delta_minutes from date_start')

    print('Context start date : {}'.format(date_start))
    print('Forecast start date: {}'.format(date_forecast_start))
    print('End date           : {}'.format(date_end))

    if date_end > date_forecast_start + datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes):
        date_end = date_forecast_start + datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes)
        print('Adjusted end date  : {} ({} time steps after forecast start)'.format(date_end, args.forecast_max_time_steps))

    sequence_start = date_start
    sequence_end = date_end
    sequence_length = int((sequence_end - sequence_start).total_seconds() / 60 / args.delta_minutes)
    print('Sequence length: {}'.format(sequence_length))
    sequence = [sequence_start + datetime.timedelta(minutes=args.delta_minutes * i) for i in range(sequence_length)]
    # find the index of the date_forecast_start in the list sequence
    if date_forecast_start not in sequence:
        raise ValueError('date_forecast_start must be in the sequence')
    sequence_forecast_start_index = sequence.index(date_forecast_start)
    sequence_prediction_window = sequence_length - (sequence_forecast_start_index) # TODO: should this be sequence_length - (sequence_forecast_start_index + 1)
    sequence_forecast = sequence[sequence_forecast_start_index:]

    sequence_data = dataset.get_sequence_data(sequence)
    jpld_seq = sequence_data[0]  # Original data
    sunmoon_seq = sequence_data[1]  # Sun and Moon geometry data
    celestrak_seq = sequence_data[2]  # CelesTrak data
    device = next(model.parameters()).device
    jpld_seq = jpld_seq.to(device) # sequence_length, channels, 180, 360
    sunmoon_seq = sunmoon_seq.to(device) # sequence_length, channels, 180, 360
    celestrak_seq = celestrak_seq.to(device) # sequence_length, channels, 180, 360
    celestrak_seq = celestrak_seq.view(celestrak_seq.shape + (1, 1)).expand(-1, 2, 180, 360)
    omniweb_seq = sequence_data[3]  # OMNIWeb data
    omniweb_seq = omniweb_seq.to(device)  # sequence_length, channels, 180, 360
    omniweb_seq = omniweb_seq.view(omniweb_seq.shape + (1, 1)).expand(-1, 13, 180, 360)
    set_seq = sequence_data[4]  # SET data
    set_seq = set_seq.to(device)  # sequence_length, channels, 180, 360
    set_seq = set_seq.view(set_seq.shape + (1, 1)).expand(-1, 4, 180, 360)

    combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=1)  # Combine along the channel dimension

    combined_seq_context = combined_seq[:sequence_forecast_start_index]  # Context data for forecast
    combined_seq_forecast = model.predict(combined_seq_context.unsqueeze(0), prediction_window=sequence_prediction_window).squeeze(0)

    jpld_forecast = combined_seq_forecast[:, 0]  # Extract JPLD channels from the forecast
    jpld_original = jpld_seq[sequence_forecast_start_index:]

    jpld_original_unnormalized = JPLD.unnormalize(jpld_original)
    jpld_forecast_unnormalized = JPLD.unnormalize(jpld_forecast).clamp(0, 140)

    forecast_mins_ahead = ['{} mins'.format((j + 1) * 15) for j in range(sequence_prediction_window)]
    titles_original = [f'JPLD GIM TEC Original: {d} - {title}' for d in sequence_forecast]
    titles_forecast = [f'JPLD GIM TEC Forecast: {d} ({forecast_mins_ahead[i]}) - {title}' for i, d in enumerate(sequence_forecast)]

    save_gim_video_comparison(
        gim_sequence_top=jpld_original_unnormalized.cpu().numpy().reshape(-1, 180, 360),
        gim_sequence_bottom=jpld_forecast_unnormalized.cpu().numpy().reshape(-1, 180, 360),
        file_name=file_name,
        vmin=0, vmax=100,
        titles_top=titles_original,
        titles_bottom=titles_forecast
    )


def save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, file_name):
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
        }
    elif isinstance(model, IonCastConvLSTM):
        checkpoint = {
            'model': 'IonCastConvLSTM',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'model_input_channels': model.input_channels,
            'model_output_channels': model.output_channels,
            'model_hidden_dim': model.hidden_dim,
            'model_num_layers': model.num_layers,
            'model_context_window': model.context_window,
            'model_prediction_window': model.prediction_window,
        }
    else:
        raise ValueError('Unknown model type: {}'.format(model))
    torch.save(checkpoint, file_name)


def load_model(file_name, device):
    checkpoint = torch.load(file_name, weights_only=False)
    if checkpoint['model'] == 'VAE1':
        model_z_dim = checkpoint['model_z_dim']
        model = VAE1(z_dim=model_z_dim)
    elif checkpoint['model'] == 'IonCastConvLSTM':
        model_input_channels = checkpoint['model_input_channels']
        model_output_channels = checkpoint['model_output_channels']
        model_hidden_dim = checkpoint['model_hidden_dim']
        model_num_layers = checkpoint['model_num_layers']
        model_context_window = checkpoint['model_context_window']
        model_prediction_window = checkpoint['model_prediction_window']
        model = IonCastConvLSTM(input_channels=model_input_channels, output_channels=model_output_channels,
                                hidden_dim=model_hidden_dim, num_layers=model_num_layers,
                                context_window=model_context_window, prediction_window=model_prediction_window)
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
    return model, optimizer, epoch, iteration, train_losses, valid_losses



def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--jpld_dir', type=str, default='jpld/webdataset', help='JPLD GIM dataset directory')
    parser.add_argument('--celestrak_file_name', type=str, default='celestrak/kp_ap_processed_timeseries.csv', help='CelesTrak dataset file name')
    parser.add_argument('--omniweb_dir', type=str, default='omniweb_karman_2025', help='OMNIWeb dataset directory')
    parser.add_argument('--omniweb_columns', nargs='+', default=['omniweb__sym_d__[nT]', 'omniweb__sym_h__[nT]', 'omniweb__asy_d__[nT]', 'omniweb__bx_gse__[nT]', 'omniweb__by_gse__[nT]', 'omniweb__bz_gse__[nT]', 'omniweb__speed__[km/s]', 'omniweb__vx_velocity__[km/s]', 'omniweb__vy_velocity__[km/s]', 'omniweb__vz_velocity__[km/s]'], help='List of OMNIWeb dataset columns to use')
    parser.add_argument('--set_file_name', type=str, default='set/space_env_tech_indices_Indices_F10_processed.csv', help='SET dataset file name')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    # parser.add_argument('--date_start', type=str, default='2010-05-13T00:00:00', help='Start date')
    # parser.add_argument('--date_end', type=str, default='2024-08-01T00:00:00', help='End date')
    parser.add_argument('--date_start', type=str, default='2023-04-19T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-04-22T00:00:00', help='End date')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time step in minutes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')    
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--model_type', type=str, choices=['VAE1', 'IonCastConvLSTM'], default='VAE1', help='Type of model to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_evals', type=int, default=4, help='Number of samples for evaluation')
    parser.add_argument('--context_window', type=int, default=16, help='Context window size for the model')
    parser.add_argument('--prediction_window', type=int, default=1, help='Evaluation window size for the model')
    parser.add_argument('--valid_event_id', nargs='*', default=['G2H3-202303230900'], help='Validation event IDs to use for evaluation at the end of each epoch')
    parser.add_argument('--valid_event_seen_id', nargs='*', default=['G0H3-202404192100'], help='Event IDs to use for evaluation at the end of each epoch, where the event was a part of the training set')
    parser.add_argument('--test_event_id', nargs='*', default=['G2H3-202303230900', 'G1H9-202302261800', 'G1H3-202302261800', 'G0H9-202302160900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--forecast_max_time_steps', type=int, default=48, help='Maximum number of time steps to evaluate for each test event')
    parser.add_argument('--model_file', type=str, help='Path to the model file to load for testing')
    parser.add_argument('--sun_moon_extra_time_steps', type=int, default=1, help='Number of extra time steps ahead to include in the dataset for Sun and Moon geometry')

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

            print('Processing excluded dates')

            datasets_jpld_valid = []

            date_exclusions = []
            if args.valid_event_id:
                for event_id in args.valid_event_id:
                    print('Excluding event ID: {}'.format(event_id))
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                    event = EventCatalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(event['date_start']) - datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                    exclusion_end = datetime.datetime.fromisoformat(event['date_end'])
                    date_exclusions.append((exclusion_start, exclusion_end))

                    datasets_jpld_valid.append(JPLD(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))

            dataset_jpld_valid = Union(datasets=datasets_jpld_valid)


            if args.model_type == 'VAE1':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = dataset_jpld_train
                dataset_valid = dataset_jpld_valid
            elif args.model_type == 'IonCastConvLSTM':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_celestrak_train = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end)
                dataset_celestrak_valid = CelesTrak(dataset_celestrak_file_name, date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end)
                dataset_omniweb_train = OMNIWeb(dataset_omniweb_dir, date_start=date_start, date_end=date_end, columns=args.omniweb_columns)
                dataset_omniweb_valid = OMNIWeb(dataset_omniweb_dir, date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end, columns=args.omniweb_columns)
                dataset_set_train = SET(dataset_set_file_name, date_start=date_start, date_end=date_end)
                dataset_set_valid = SET(dataset_set_file_name, date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end)
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train, dataset_celestrak_train, dataset_omniweb_train, dataset_set_train], sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid, dataset_celestrak_valid, dataset_omniweb_valid, dataset_set_valid], sequence_length=training_sequence_length)
            else:
                raise ValueError('Unknown model type: {}'.format(args.model_type))

            print('\nTrain size: {:,}'.format(len(dataset_train)))
            print('Valid size: {:,}'.format(len(dataset_valid)))

            train_loader = DataLoader(
                dataset_train, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
            valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = glob.glob('{}/epoch-*-model.pth'.format(args.target_dir))
            if len(model_files) > 0:
                model_files.sort()
                model_file = model_files[-1]
                print('Resuming training from model file: {}'.format(model_file))
                model, optimizer, epoch, iteration, train_losses, valid_losses = load_model(model_file, device)
                epoch_start = epoch + 1
                iteration = iteration + 1
                print('Next epoch    : {:,}'.format(epoch_start+1))
                print('Next iteration: {:,}'.format(iteration+1))
            else:
                print('Creating new model')
                if args.model_type == 'VAE1':
                    model = VAE1(z_dim=512, sigma_vae=False)
                elif args.model_type == 'IonCastConvLSTM':
                    model = IonCastConvLSTM(input_channels=56, output_channels=56, context_window=args.context_window, prediction_window=args.prediction_window)
                else:
                    raise ValueError('Unknown model type: {}'.format(args.model_type))

                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                iteration = 0
                epoch_start = 0
                train_losses = []
                valid_losses = []

                model = model.to(device)

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('\nNumber of parameters: {:,}\n'.format(num_params))
            
            for epoch in range(epoch_start, args.epochs):
                print('\n*** Epoch {:,}/{:,} started'.format(epoch+1, args.epochs))
                print('*** Training')
                # Training
                model.train()
                with tqdm(total=len(train_loader)) as pbar:
                    for i, batch in enumerate(train_loader):
                        # a = 1/0
                        optimizer.zero_grad()

                        if args.model_type == 'VAE1':
                            jpld, _ = batch
                            jpld = jpld.to(device)

                            loss = model.loss(jpld)
                        elif args.model_type == 'IonCastConvLSTM':
                            jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            sunmoon_seq = sunmoon_seq.to(device)
                            celestrak_seq = celestrak_seq.to(device)
                            celestrak_seq = celestrak_seq.view(celestrak_seq.shape + (1, 1)).expand(-1, -1, 2, 180, 360)
                            omniweb_seq = omniweb_seq.to(device)
                            omniweb_seq = omniweb_seq.view(omniweb_seq.shape + (1, 1)).expand(-1, -1, 13, 180, 360)
                            set_seq = set_seq.to(device)
                            set_seq = set_seq.view(set_seq.shape + (1, 1)).expand(-1, -1, 4, 180, 360)

                            combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2) # Combine along the channel dimension

                            loss = model.loss(combined_seq, context_window=args.context_window)
                        else:
                            raise ValueError('Unknown model type: {}'.format(args.model_type))
                        
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
                    for batch in valid_loader:
                        if args.model_type == 'VAE1':
                            jpld, _ = batch
                            jpld = jpld.to(device)
                            loss = model.loss(jpld)
                        elif args.model_type == 'IonCastConvLSTM':
                            jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            sunmoon_seq = sunmoon_seq.to(device)
                            celestrak_seq = celestrak_seq.to(device)
                            celestrak_seq = celestrak_seq.view(celestrak_seq.shape + (1, 1)).expand(-1, -1, 2, 180, 360)
                            omniweb_seq = omniweb_seq.to(device)
                            omniweb_seq = omniweb_seq.view(omniweb_seq.shape + (1, 1)).expand(-1, -1, 13, 180, 360)
                            set_seq = set_seq.to(device)
                            set_seq = set_seq.view(set_seq.shape + (1, 1)).expand(-1, -1, 4, 180, 360)

                            combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)  # Combine along the channel dimension
                            loss = model.loss(combined_seq, context_window=args.context_window)
                        else:
                            raise ValueError('Unknown model type: {}'.format(args.model_type))
                        valid_loss += loss.item()
                valid_loss /= len(valid_loader)
                valid_losses.append((iteration, valid_loss))
                print(f'Validation Loss: {valid_loss:.4f}')

                file_name_prefix = f'epoch-{epoch + 1:02d}-'

                # Save model
                model_file = os.path.join(args.target_dir, f'{file_name_prefix}model.pth')
                save_model(model, optimizer, epoch, iteration, train_losses, valid_losses, model_file)

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

                # Plot model eval results
                model.eval()
                with torch.no_grad():
                    num_evals = args.num_evals

                    if args.model_type == 'VAE1':
                        # Set random seed for reproducibility of evaluation samples across epochs
                        rng_state = torch.get_rng_state()
                        torch.manual_seed(args.seed)

                        # Reconstruct a batch from the validation set
                        jpld_orig, jpld_orig_dates = next(iter(valid_loader))
                        jpld_orig = jpld_orig[:num_evals]
                        jpld_orig_dates = jpld_orig_dates[:num_evals]

                        jpld_orig = jpld_orig.to(device)
                        jpld_recon, _, _ = model.forward(jpld_orig)
                        jpld_orig_unnormalized = JPLD.unnormalize(jpld_orig)
                        jpld_recon_unnormalized = JPLD.unnormalize(jpld_recon)

                        # Sample a batch from the model
                        jpld_sample = model.sample(n=num_evals)
                        jpld_sample_unnormalized = JPLD.unnormalize(jpld_sample)
                        jpld_sample_unnormalized = jpld_sample_unnormalized.clamp(0, 140)
                        torch.set_rng_state(rng_state)
                        # Resume with the original random state

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


                    elif args.model_type == 'IonCastConvLSTM':
                        if args.valid_event_id:
                            for event_id in args.valid_event_id:
                                if event_id not in EventCatalog:
                                    raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                                event = EventCatalog[event_id]
                                event_start, event_end, max_kp, = event['date_start'], event['date_end'], event['max_kp']
                                event_start = datetime.datetime.fromisoformat(event_start)
                                event_end = datetime.datetime.fromisoformat(event_end)

                                print('* Validating event ID: {}'.format(event_id))
                                date_start = event_start - datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                                date_forecast_start = event_start
                                date_end = event_end
                                file_name = os.path.join(args.target_dir, f'{file_name_prefix}valid-event-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                title = f'Event: {event_id}, Kp={max_kp}'
                                run_forecast(model, dataset_valid, date_start, date_end, date_forecast_start, title, file_name, args)

                        if args.valid_event_seen_id:
                            for event_id in args.valid_event_seen_id:
                                if event_id not in EventCatalog:
                                    raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                                event = EventCatalog[event_id]
                                event_start, event_end, max_kp = event['date_start'], event['date_end'], event['max_kp']
                                event_start = datetime.datetime.fromisoformat(event_start)
                                event_end = datetime.datetime.fromisoformat(event_end)

                                print('* Validating seen event ID: {}'.format(event_id))
                                date_start = event_start - datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                                date_forecast_start = event_start
                                date_end = event_end
                                file_name = os.path.join(args.target_dir, f'{file_name_prefix}valid-event-seen-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                title = f'Event: {event_id}, Kp={max_kp}'
                                run_forecast(model, dataset_train, date_start, date_end, date_forecast_start, title, file_name, args)


        elif args.mode == 'test':

            print('*** Testing mode\n')

            model, _, _, _, _, _ = load_model(args.model_file, device)
            model.eval()
            model = model.to(device)

            with torch.no_grad():
                tests_to_run = []
                if args.test_event_id:
                    for event_id in args.test_event_id:
                        if event_id not in EventCatalog:
                            raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                        event = EventCatalog[event_id]
                        date_start, date_end, max_kp = event['date_start'], event['date_end'], event['max_kp']
                        event_start = datetime.datetime.fromisoformat(date_start)
                        event_end = datetime.datetime.fromisoformat(date_end)

                        print('* Testing event ID: {}'.format(event_id))
                        date_start = event_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                        date_forecast_start = event_start
                        date_end = event_end
                        file_name = os.path.join(args.target_dir, f'test-event-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                        title = f'Event: {event_id}, Kp={max_kp}'
                        tests_to_run.append((date_start, date_end, date_forecast_start, title, file_name))

                else:
                    print('No test events specified, will use date_start and date_end arguments')
                    event_start = datetime.datetime.fromisoformat(args.date_start)
                    event_end = datetime.datetime.fromisoformat(args.date_end)
                    date_start = event_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                    date_forecast_start = event_start
                    date_end = event_end
                    file_name = os.path.join(args.target_dir, f'test-{event_start.strftime("%Y%m%d%H%M")}-{event_end.strftime("%Y%m%d%H%M")}.mp4')
                    title = f'Test from {event_start.strftime("%Y-%m-%d %H:%M:%S")} to {event_end.strftime("%Y-%m-%d %H:%M:%S")}'
                    tests_to_run.append((date_start, date_end, date_forecast_start, title, file_name))

                dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)
                dataset_celestrak_file_name = os.path.join(args.data_dir, args.celestrak_file_name)
                training_sequence_length = args.context_window + args.prediction_window

                print('Running tests:')
                for i, (date_start, date_end, date_forecast_start, title, file_name) in enumerate(tests_to_run):
                    print(f'\n\n* Testing event {i+1}/{len(tests_to_run)}: {title}')
                    # Create dataset for each test individually with date filtering
                    dataset_jpld = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end)
                    dataset_sunmoon = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                    dataset_celestrak = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end)
                    dataset_omniweb = OMNIWeb(os.path.join(args.data_dir, args.omniweb_dir), date_start=date_start, date_end=date_end, columns=args.omniweb_columns)
                    dataset_set = SET(os.path.join(args.data_dir, args.set_file_name), date_start=date_start, date_end=date_end)
                    dataset = Sequences(datasets=[dataset_jpld, dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                    run_forecast(model, dataset, date_start, date_end, date_forecast_start, title, file_name, args)
                    
                    # Force cleanup
                    del dataset_jpld, dataset
                    torch.cuda.empty_cache()
        else:
            raise ValueError('Unknown mode: {}'.format(args.mode))

        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()


# Example
# python run.py --data_dir /disk2-ssd-8tb/data/2025-hl-ionosphere --mode train --target_dir ./train-1 --num_workers 4 --batch_size 4 --model_type IonCastConvLSTM --epochs 2 --learning_rate 1e-3 --weight_decay 0.0 --context_window 4 --prediction_window 4 --num_evals 4 --date_start 2023-07-01T00:00:00 --date_end 2023-08-01T00:00:00
