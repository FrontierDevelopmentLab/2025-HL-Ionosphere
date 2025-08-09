import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import glob
import imageio
import shutil
import random
import csv

from util import Tee
from util import set_random_seed
from util import md5_hash_str
# from model_vae import VAE1
from model_convlstm import IonCastConvLSTM
from model_lstm import IonCastLSTM
from dataset_jpld import JPLD
from dataset_sequences import Sequences
from dataset_union import Union
from dataset_sunmoongeometry import SunMoonGeometry
from dataset_celestrak import CelesTrak
from dataset_omniweb import OMNIWeb
from dataset_set import SET
from dataloader_cached import CachedDataLoader
from events import EventCatalog, validation_events_1, validation_events_2, validation_events_3

event_catalog = EventCatalog()

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
                                       titles_top=None, titles_bottom=None, fps=2, max_frames=None, cbar_label='TEC (TECU)', fig_title=None):
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
        if fig_title:
            fig_temp.suptitle(fig_title, fontsize=12)
        gs = fig_temp.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], 
                                  wspace=0.05, hspace=0.15, left=0.05, right=0.92, top=0.92, bottom=0.05)
        
        # Plot frame - maps
        ax_top = fig_temp.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_bottom = fig_temp.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        
        # Plot frame - colorbar axes
        cbar_ax_top = fig_temp.add_subplot(gs[0, 1])
        cbar_ax_bottom = fig_temp.add_subplot(gs[1, 1])
        
        # Create the top map and get its image object. This will use the function's vmin/vmax or auto-scale.
        im_top = plot_global_ionosphere_map(ax_top, gim_sequence_top[i], cmap=cmap, vmin=vmin, vmax=vmax,
                                           title=titles_top[i] if titles_top else None)
        
        # Get the effective color limits from the top plot to ensure the bottom plot uses the exact same scale.
        vmin_actual, vmax_actual = im_top.get_clim()
        
        # Create the bottom map using the same color limits as the top map.
        im_bottom = plot_global_ionosphere_map(ax_bottom, gim_sequence_bottom[i], cmap=cmap, vmin=vmin_actual, vmax=vmax_actual,
                                              title=titles_bottom[i] if titles_bottom else None)
        
        # Add colorbars. They will now be identical.
        cbar_top = fig_temp.colorbar(im_top, cax=cbar_ax_top)
        cbar_top.set_label(cbar_label)

        cbar_bottom = fig_temp.colorbar(im_bottom, cax=cbar_ax_bottom)
        cbar_bottom.set_label(cbar_label)

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


def run_forecast(model, dataset, date_start, date_end, date_forecast_start, args):
    if not isinstance(model, (IonCastConvLSTM)) and not isinstance(model, IonCastLSTM):
        raise ValueError('Model must be an instance of IonCastConvLSTM or IonCastLSTM')
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

    sequence_start_date = date_start
    sequence_end_date = date_end
    sequence_length = int((sequence_end_date - sequence_start_date).total_seconds() / 60 / args.delta_minutes)
    sequence_dates = [sequence_start_date + datetime.timedelta(minutes=args.delta_minutes * i) for i in range(sequence_length)]
    # find the index of the date_forecast_start in the list sequence
    if date_forecast_start not in sequence_dates:
        raise ValueError('date_forecast_start must be in the sequence')
    sequence_forecast_start_index = sequence_dates.index(date_forecast_start)
    sequence_prediction_window = sequence_length - (sequence_forecast_start_index) # TODO: should this be sequence_length - (sequence_forecast_start_index + 1)
    sequence_forecast_dates = sequence_dates[sequence_forecast_start_index:]
    print(f'Sequence length    : {sequence_length} ({sequence_forecast_start_index} context + {sequence_prediction_window} forecast)')

    sequence_data = dataset.get_sequence_data(sequence_dates)
    jpld_seq_data = sequence_data[0]  # Original data
    sunmoon_seq_data = sequence_data[1]  # Sun and Moon geometry data
    celestrak_seq_data = sequence_data[2]  # CelesTrak data
    device = next(model.parameters()).device
    jpld_seq_data = jpld_seq_data.to(device) # sequence_length, channels, 180, 360
    sunmoon_seq_data = sunmoon_seq_data.to(device) # sequence_length, channels, 180, 360
    celestrak_seq_data = celestrak_seq_data.to(device) # sequence_length, channels, 180, 360
    omniweb_seq_data = sequence_data[3]  # OMNIWeb data
    omniweb_seq_data = omniweb_seq_data.to(device)  # sequence_length, channels, 180, 360
    set_seq_data = sequence_data[4]  # SET data
    set_seq_data = set_seq_data.to(device)  # sequence_length, channels, 180, 360

    combined_seq_data = torch.cat((jpld_seq_data, sunmoon_seq_data, celestrak_seq_data, omniweb_seq_data, set_seq_data), dim=1)  # Combine along the channel dimension

    combined_seq_data_context = combined_seq_data[:sequence_forecast_start_index]  # Context data for forecast
    combined_seq_data_original = combined_seq_data[sequence_forecast_start_index:]  # Original data for forecast
    combined_seq_data_forecast = model.predict(combined_seq_data_context.unsqueeze(0), prediction_window=sequence_prediction_window).squeeze(0)

    jpld_forecast = combined_seq_data_forecast[:, 0]  # Extract JPLD channels from the forecast
    jpld_original = combined_seq_data_original[:, 0]

    jpld_original_unnormalized = JPLD.unnormalize(jpld_original)
    jpld_forecast_unnormalized = JPLD.unnormalize(jpld_forecast).clamp(0, 140)

    return jpld_forecast, jpld_original, jpld_forecast_unnormalized, jpld_original_unnormalized, combined_seq_data_original, combined_seq_data_forecast, sequence_start_date, sequence_forecast_dates, sequence_prediction_window


def eval_forecast(model, dataset, event_catalog, event_id, file_name_prefix, save_video, args):
    if event_id not in event_catalog:
        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
    event = event_catalog[event_id]
    event_start, event_end, max_kp, = event['date_start'], event['date_end'], event['max_kp']
    event_start = datetime.datetime.fromisoformat(event_start)
    event_end = datetime.datetime.fromisoformat(event_end)

    print('\n* Forecasting')
    print('Event ID           : {}'.format(event_id))
    date_start = event_start - datetime.timedelta(minutes=args.context_window * args.delta_minutes)
    date_forecast_start = event_start
    date_end = event_end
    file_name = os.path.join(args.target_dir, f'{file_name_prefix}-event-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
    title = f'Event: {event_id}, Kp={max_kp}'

    jpld_forecast, jpld_original, jpld_forecast_unnormalized, jpld_original_unnormalized, combined_seq_data_original, combined_seq_data_forecast, sequence_start_date, sequence_forecast_dates, sequence_prediction_window = run_forecast(model, dataset, date_start, date_end, date_forecast_start, args)

    jpld_rmse = torch.nn.functional.mse_loss(jpld_forecast, jpld_original, reduction='mean').sqrt().item()
    print('JPLD RMSE          : {}'.format(jpld_rmse))
    jpld_mae = torch.nn.functional.l1_loss(jpld_forecast, jpld_original, reduction='mean').item()
    print('JPLD MAE           : {}'.format(jpld_mae))

    jpld_unnormalized_rmse = torch.nn.functional.mse_loss(jpld_forecast_unnormalized, jpld_original_unnormalized, reduction='mean').sqrt().item()
    print(f'\033[92mJPLD RMSE (TECU)   : {jpld_unnormalized_rmse}\033[0m')
    jpld_unnormalized_mae = torch.nn.functional.l1_loss(jpld_forecast_unnormalized, jpld_original_unnormalized, reduction='mean').item()
    print(f'\033[96mJPLD MAE (TECU)    : {jpld_unnormalized_mae}\033[0m')

    # --- Regional Metrics ---
    latitudes = np.linspace(-90, 90, 180)
    
    # Low-latitude mask
    low_lat_mask = (latitudes >= -20) & (latitudes <= 20)
    jpld_unnormalized_rmse_low_lat = torch.nn.functional.mse_loss(
        jpld_forecast_unnormalized[:, low_lat_mask, :],
        jpld_original_unnormalized[:, low_lat_mask, :]
    ).sqrt().item()
    print(f'JPLD Low-Latitude RMSE (TECU) : {jpld_unnormalized_rmse_low_lat:.4f}')

    # Mid-latitude mask
    mid_lat_mask = ((latitudes > 20) & (latitudes <= 60)) | ((latitudes < -20) & (latitudes >= -60))
    jpld_unnormalized_rmse_mid_lat = torch.nn.functional.mse_loss(
        jpld_forecast_unnormalized[:, mid_lat_mask, :],
        jpld_original_unnormalized[:, mid_lat_mask, :]
    ).sqrt().item()
    print(f'JPLD Mid-Latitude RMSE (TECU) : {jpld_unnormalized_rmse_mid_lat:.4f}')
    
    # High-latitude mask
    high_lat_mask = (latitudes > 60) | (latitudes < -60)
    jpld_unnormalized_rmse_high_lat = torch.nn.functional.mse_loss(
        jpld_forecast_unnormalized[:, high_lat_mask, :],
        jpld_original_unnormalized[:, high_lat_mask, :]
    ).sqrt().item()
    print(f'JPLD High-Latitude RMSE (TECU): {jpld_unnormalized_rmse_high_lat:.4f}')

    fig_title = title + f' - RMSE: {jpld_unnormalized_rmse:.2f} TECU - MAE: {jpld_unnormalized_mae:.2f} TECU'
    forecast_mins_ahead = ['{} mins'.format((j + 1) * 15) for j in range(sequence_prediction_window)]
    titles_original = [f'JPLD GIM TEC Ground Truth: {d}' for d in sequence_forecast_dates]
    titles_forecast = [f'JPLD GIM TEC Forecast: {d} - Autoregressive rollout from {sequence_start_date} ({forecast_mins_ahead[i]})' for i, d in enumerate(sequence_forecast_dates)]

    if save_video:
        save_gim_video_comparison(
            gim_sequence_top=jpld_original_unnormalized.cpu().numpy().reshape(-1, 180, 360),
            gim_sequence_bottom=jpld_forecast_unnormalized.cpu().numpy().reshape(-1, 180, 360),
            file_name=file_name,
            vmin=0, vmax=120,
            titles_top=titles_original,
            titles_bottom=titles_forecast,
            fig_title=fig_title
        )

        if args.save_all_channels:
            num_channels = combined_seq_data_original.shape[1]
            for i in range(num_channels):
                channel_original = combined_seq_data_original[:, i]
                channel_forecast = combined_seq_data_forecast[:, i]
                channel_original_unnormalized = channel_original
                channel_forecast_unnormalized = channel_forecast

                titles_channel_original = [f'Channel {i} Original: {d} - {title}' for d in sequence_forecast_dates]
                titles_channel_forecast = [f'Channel {i} Forecast: {d} ({forecast_mins_ahead[i]}) - {title}' for i, d in enumerate(sequence_forecast_dates)]

                file_name_channel = os.path.join(os.path.dirname(file_name), os.path.basename(file_name).replace('.mp4', f'_channel_{i:02d}.mp4'))
                save_gim_video_comparison(
                    gim_sequence_top=channel_original_unnormalized.cpu().numpy().reshape(-1, 180, 360),
                    gim_sequence_bottom=channel_forecast_unnormalized.cpu().numpy().reshape(-1, 180, 360),
                    file_name=file_name_channel,
                    # vmin=0, vmax=100,
                    titles_top=titles_channel_original,
                    titles_bottom=titles_channel_forecast,
                    fig_title=fig_title,
                    cbar_label=''
                )
                print(f'Saved channel {i} forecast video to {file_name_channel}')
    
    return jpld_rmse, jpld_mae, jpld_unnormalized_rmse, jpld_unnormalized_mae, jpld_unnormalized_rmse_low_lat, jpld_unnormalized_rmse_mid_lat, jpld_unnormalized_rmse_high_lat

def save_metrics(event_id, jpld_rmse, jpld_mae, jpld_unnormalized_rmse, jpld_unnormalized_mae, jpld_unnormalized_rmse_low_lat, jpld_unnormalized_rmse_mid_lat, jpld_unnormalized_rmse_high_lat, file_name_prefix):
    # Save metrics to a CSV file
    num_events = len(event_id)
    file_name_csv = os.path.join(file_name_prefix + '.csv')
    print(f'Saving metrics to {file_name_csv}')
    with open(file_name_csv, 'w', newline='') as csvfile:
        fieldnames = ['event_id', 'jpld_rmse', 'jpld_mae', 'jpld_unnormalized_rmse', 'jpld_unnormalized_mae', 'jpld_unnormalized_rmse_low_lat', 'jpld_unnormalized_rmse_mid_lat', 'jpld_unnormalized_rmse_high_lat']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_events):
            writer.writerow({
                'event_id': event_id[i],
                'jpld_rmse': jpld_rmse[i],
                'jpld_mae': jpld_mae[i],
                'jpld_unnormalized_rmse': jpld_unnormalized_rmse[i],
                'jpld_unnormalized_mae': jpld_unnormalized_mae[i],
                'jpld_unnormalized_rmse_low_lat': jpld_unnormalized_rmse_low_lat[i],
                'jpld_unnormalized_rmse_mid_lat': jpld_unnormalized_rmse_mid_lat[i],
                'jpld_unnormalized_rmse_high_lat': jpld_unnormalized_rmse_high_lat[i]
            })

    jpld_rmse = np.array(jpld_rmse)
    jpld_mae = np.array(jpld_mae)
    jpld_unnormalized_rmse = np.array(jpld_unnormalized_rmse)
    jpld_unnormalized_mae = np.array(jpld_unnormalized_mae)
    jpld_unnormalized_rmse_low_lat = np.array(jpld_unnormalized_rmse_low_lat)
    jpld_unnormalized_rmse_mid_lat = np.array(jpld_unnormalized_rmse_mid_lat)
    jpld_unnormalized_rmse_high_lat = np.array(jpld_unnormalized_rmse_high_lat)

    # add rows with mean, std, min, max of all metrics
    with open(file_name_csv, 'a', newline='') as csvfile:
        fieldnames = ['event_id', 'jpld_rmse', 'jpld_mae', 'jpld_unnormalized_rmse', 'jpld_unnormalized_mae', 'jpld_unnormalized_rmse_low_lat', 'jpld_unnormalized_rmse_mid_lat', 'jpld_unnormalized_rmse_high_lat']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'event_id': 'all-mean',
            'jpld_rmse': np.mean(jpld_rmse),
            'jpld_mae': np.mean(jpld_mae),
            'jpld_unnormalized_rmse': np.mean(jpld_unnormalized_rmse),
            'jpld_unnormalized_mae': np.mean(jpld_unnormalized_mae),
            'jpld_unnormalized_rmse_low_lat': np.mean(jpld_unnormalized_rmse_low_lat),
            'jpld_unnormalized_rmse_mid_lat': np.mean(jpld_unnormalized_rmse_mid_lat),
            'jpld_unnormalized_rmse_high_lat': np.mean(jpld_unnormalized_rmse_high_lat)
        })
        writer.writerow({
            'event_id': 'all-std',
            'jpld_rmse': np.std(jpld_rmse),
            'jpld_mae': np.std(jpld_mae),
            'jpld_unnormalized_rmse': np.std(jpld_unnormalized_rmse),
            'jpld_unnormalized_mae': np.std(jpld_unnormalized_mae),
            'jpld_unnormalized_rmse_low_lat': np.std(jpld_unnormalized_rmse_low_lat),
            'jpld_unnormalized_rmse_mid_lat': np.std(jpld_unnormalized_rmse_mid_lat),
            'jpld_unnormalized_rmse_high_lat': np.std(jpld_unnormalized_rmse_high_lat)
        })
        writer.writerow({
            'event_id': 'all-min',
            'jpld_rmse': np.min(jpld_rmse),
            'jpld_mae': np.min(jpld_mae),
            'jpld_unnormalized_rmse': np.min(jpld_unnormalized_rmse),
            'jpld_unnormalized_mae': np.min(jpld_unnormalized_mae),
            'jpld_unnormalized_rmse_low_lat': np.min(jpld_unnormalized_rmse_low_lat),
            'jpld_unnormalized_rmse_mid_lat': np.min(jpld_unnormalized_rmse_mid_lat),
            'jpld_unnormalized_rmse_high_lat': np.min(jpld_unnormalized_rmse_high_lat)
        })
        writer.writerow({
            'event_id': 'all-max',
            'jpld_rmse': np.max(jpld_rmse),
            'jpld_mae': np.max(jpld_mae),
            'jpld_unnormalized_rmse': np.max(jpld_unnormalized_rmse),
            'jpld_unnormalized_mae': np.max(jpld_unnormalized_mae),
            'jpld_unnormalized_rmse_low_lat': np.max(jpld_unnormalized_rmse_low_lat),
            'jpld_unnormalized_rmse_mid_lat': np.max(jpld_unnormalized_rmse_mid_lat),
            'jpld_unnormalized_rmse_high_lat': np.max(jpld_unnormalized_rmse_high_lat)
        })

    # add rows with mean, std, min, max of all metrics for subsets of events starting with G0, G1, G2, G3, G4, G5
    for prefix in ['G0', 'G1', 'G2', 'G3', 'G4', 'G5']:
        indices = [i for i, event in enumerate(event_id) if event.startswith(prefix)]
        if len(indices) > 0:
            jpld_rmse_subset = jpld_rmse[indices]
            jpld_mae_subset = jpld_mae[indices]
            jpld_unnormalized_rmse_subset = jpld_unnormalized_rmse[indices]
            jpld_unnormalized_mae_subset = jpld_unnormalized_mae[indices]
            with open(file_name_csv, 'a', newline='') as csvfile:
                fieldnames = ['event_id', 'jpld_rmse', 'jpld_mae', 'jpld_unnormalized_rmse', 'jpld_unnormalized_mae', 'jpld_unnormalized_rmse_low_lat', 'jpld_unnormalized_rmse_mid_lat', 'jpld_unnormalized_rmse_high_lat']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'event_id': f'{prefix}-mean',
                    'jpld_rmse': np.mean(jpld_rmse_subset),
                    'jpld_mae': np.mean(jpld_mae_subset),
                    'jpld_unnormalized_rmse': np.mean(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_mae': np.mean(jpld_unnormalized_mae_subset),
                    'jpld_unnormalized_rmse_low_lat': np.mean(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_mid_lat': np.mean(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_high_lat': np.mean(jpld_unnormalized_rmse_subset)
                })
                writer.writerow({
                    'event_id': f'{prefix}-std',
                    'jpld_rmse': np.std(jpld_rmse_subset),
                    'jpld_mae': np.std(jpld_mae_subset),
                    'jpld_unnormalized_rmse': np.std(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_mae': np.std(jpld_unnormalized_mae_subset),
                    'jpld_unnormalized_rmse_low_lat': np.std(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_mid_lat': np.std(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_high_lat': np.std(jpld_unnormalized_rmse_subset)
                })
                writer.writerow({
                    'event_id': f'{prefix}-min',
                    'jpld_rmse': np.min(jpld_rmse_subset),
                    'jpld_mae': np.min(jpld_mae_subset),
                    'jpld_unnormalized_rmse': np.min(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_mae': np.min(jpld_unnormalized_mae_subset),
                    'jpld_unnormalized_rmse_low_lat': np.min(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_mid_lat': np.min(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_high_lat': np.min(jpld_unnormalized_rmse_subset)
                })
                writer.writerow({
                    'event_id': f'{prefix}-max',
                    'jpld_rmse': np.max(jpld_rmse_subset),
                    'jpld_mae': np.max(jpld_mae_subset),
                    'jpld_unnormalized_rmse': np.max(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_mae': np.max(jpld_unnormalized_mae_subset),
                    'jpld_unnormalized_rmse_low_lat': np.max(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_mid_lat': np.max(jpld_unnormalized_rmse_subset),
                    'jpld_unnormalized_rmse_high_lat': np.max(jpld_unnormalized_rmse_subset)
                })

    # save a metrics figure (pdf) with four histograms of all metrics for all events
    file_name_hist = os.path.join(file_name_prefix + '-histograms.pdf')
    print(f'Saving metrics histograms to {file_name_hist}')

    # Prepare all data for main row and for each G0-G5 subset
    prefixes = ['all', 'G0', 'G1', 'G2', 'G3', 'G4', 'G5']
    metrics_dict = {}

    # NOAA color scale for G1-G5, and a custom color for G0
    prefix_colors = {
        'all': 'black',
        'G0': '#bdbdbd',   # Gray for G0 (custom, not in NOAA)
        'G1': '#ffff00',   # Yellow
        'G2': '#ffcc00',   # Orange-yellow
        'G3': '#ff9900',   # Orange
        'G4': '#ff0000',   # Red
        'G5': '#990000',   # Dark red
    }

    # Convert all to numpy arrays for easier indexing
    event_id = np.array(event_id)
    jpld_rmse = np.array(jpld_rmse)
    jpld_mae = np.array(jpld_mae)
    jpld_unnormalized_rmse = np.array(jpld_unnormalized_rmse)
    jpld_unnormalized_mae = np.array(jpld_unnormalized_mae)
    jpld_unnormalized_rmse_low_lat = np.array(jpld_unnormalized_rmse_low_lat)
    jpld_unnormalized_rmse_mid_lat = np.array(jpld_unnormalized_rmse_mid_lat)
    jpld_unnormalized_rmse_high_lat = np.array(jpld_unnormalized_rmse_high_lat)

    # All events
    metrics_dict['all'] = {
        'jpld_rmse': jpld_rmse,
        'jpld_mae': jpld_mae,
        'jpld_unnormalized_rmse': jpld_unnormalized_rmse,
        'jpld_unnormalized_mae': jpld_unnormalized_mae,
        'jpld_unnormalized_rmse_low_lat': jpld_unnormalized_rmse_low_lat,
        'jpld_unnormalized_rmse_mid_lat': jpld_unnormalized_rmse_mid_lat,
        'jpld_unnormalized_rmse_high_lat': jpld_unnormalized_rmse_high_lat,
    }

    # Subsets by prefix
    for prefix in prefixes[1:]:
        idx = np.char.startswith(event_id.astype(str), prefix)
        metrics_dict[prefix] = {
            'jpld_rmse': jpld_rmse[idx],
            'jpld_mae': jpld_mae[idx],
            'jpld_unnormalized_rmse': jpld_unnormalized_rmse[idx],
            'jpld_unnormalized_mae': jpld_unnormalized_mae[idx],
            'jpld_unnormalized_rmse_low_lat': jpld_unnormalized_rmse_low_lat[idx],
            'jpld_unnormalized_rmse_mid_lat': jpld_unnormalized_rmse_mid_lat[idx],
            'jpld_unnormalized_rmse_high_lat': jpld_unnormalized_rmse_high_lat[idx],
        }

    n_rows = len(prefixes)
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    metric_names = [
        ('jpld_rmse', 'JPLD RMSE'),
        ('jpld_mae', 'JPLD MAE'),
        ('jpld_unnormalized_rmse', 'JPLD RMSE (TECU)'),
        ('jpld_unnormalized_mae', 'JPLD MAE (TECU)'),
        ('jpld_unnormalized_rmse_low_lat', 'JPLD Low Lat RMSE (TECU)'),
        ('jpld_unnormalized_rmse_mid_lat', 'JPLD Mid Lat RMSE (TECU)'),
        ('jpld_unnormalized_rmse_high_lat', 'JPLD High Lat RMSE (TECU)'),
    ]

    for row, prefix in enumerate(prefixes):
        metrics = metrics_dict[prefix]
        color = prefix_colors.get(prefix, 'black')
        for col, (key, title) in enumerate(metric_names):
            ax = axes[row, col]
            data = metrics[key]
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
                ax.axvline(np.mean(data), color='gray', linestyle='dashed', linewidth=1, label='Mean')
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    if prefix == 'all':
                        ax.set_ylabel('All events')
                    else:
                        ax.set_ylabel(f'{prefix} events')
                ax.legend()
                ax.ticklabel_format(style='plain', axis='x')
            else:
                # No data: turn off axis
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_name_hist)
    plt.close()


def save_model(model, optimizer, scheduler, epoch, iteration, train_losses, valid_losses, train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses, best_valid_rmse, file_name):
    print('Saving model to {}'.format(file_name))
    # if isinstance(model, VAE1):
    #     checkpoint = {
    #         'model': 'VAE1',
    #         'epoch': epoch,
    #         'iteration': iteration,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict(),
    #         'train_losses': train_losses,
    #         'valid_losses': valid_losses,
    #         'model_z_dim': model.z_dim,
    #     }
    if isinstance(model, IonCastConvLSTM):
        checkpoint = {
            'model': 'IonCastConvLSTM',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses,
            'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses,
            'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels,
            'model_output_channels': model.output_channels,
            'model_hidden_dim': model.hidden_dim,
            'model_num_layers': model.num_layers,
            'model_context_window': model.context_window,
            'model_prediction_window': model.prediction_window,
            'model_dropout': model.dropout,
        }
    elif isinstance(model, IonCastLSTM):
        checkpoint = {
            'model': 'IonCastLSTM',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_rmse_losses': train_rmse_losses,
            'valid_rmse_losses': valid_rmse_losses,
            'train_jpld_rmse_losses': train_jpld_rmse_losses,
            'valid_jpld_rmse_losses': valid_jpld_rmse_losses,
            'best_valid_rmse': best_valid_rmse,
            'model_input_channels': model.input_channels,
            'model_output_channels': model.output_channels,
            'model_hidden_dim': model.hidden_dim,
            'model_lstm_dim': model.lstm_dim,
            'model_num_layers': model.num_layers,
            'model_context_window': model.context_window,
            'model_dropout': model.dropout,
        }
    else:
        raise ValueError('Unknown model type: {}'.format(model))
    torch.save(checkpoint, file_name)


def load_model(file_name, device):
    checkpoint = torch.load(file_name, weights_only=False)
    # if checkpoint['model'] == 'VAE1':
    #     model_z_dim = checkpoint['model_z_dim']
    #     model = VAE1(z_dim=model_z_dim)
    if checkpoint['model'] == 'IonCastConvLSTM':
        model_input_channels = checkpoint['model_input_channels']
        model_output_channels = checkpoint['model_output_channels']
        model_hidden_dim = checkpoint['model_hidden_dim']
        model_num_layers = checkpoint['model_num_layers']
        model_context_window = checkpoint['model_context_window']
        model_prediction_window = checkpoint['model_prediction_window']
        model_dropout = checkpoint['model_dropout']
        model = IonCastConvLSTM(input_channels=model_input_channels, output_channels=model_output_channels,
                                hidden_dim=model_hidden_dim, num_layers=model_num_layers,
                                context_window=model_context_window, prediction_window=model_prediction_window,
                                dropout=model_dropout)
    elif checkpoint['model'] == 'IonCastLSTM':
        model_input_channels = checkpoint['model_input_channels']
        model_output_channels = checkpoint['model_output_channels']
        model_hidden_dim = checkpoint['model_hidden_dim']
        model_lstm_dim = checkpoint['model_lstm_dim']
        model_num_layers = checkpoint['model_num_layers']
        model_context_window = checkpoint['model_context_window']
        model_dropout = checkpoint['model_dropout']
        model = IonCastLSTM(input_channels=model_input_channels, output_channels=model_output_channels,
                            hidden_dim=model_hidden_dim, lstm_dim=model_lstm_dim, num_layers=model_num_layers,
                            context_window=model_context_window, dropout=model_dropout)
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
    train_rmse_losses = checkpoint['train_rmse_losses']
    valid_rmse_losses = checkpoint['valid_rmse_losses']
    train_jpld_rmse_losses = checkpoint['train_jpld_rmse_losses']
    valid_jpld_rmse_losses = checkpoint['valid_jpld_rmse_losses']
    best_valid_rmse = checkpoint['best_valid_rmse']

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
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    # parser.add_argument('--date_start', type=str, default='2010-05-13T00:00:00', help='Start date')
    # parser.add_argument('--date_end', type=str, default='2024-08-01T00:00:00', help='End date')
    parser.add_argument('--date_start', type=str, default='2024-04-19T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-04-20T00:00:00', help='End date')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time step in minutes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--model_type', type=str, choices=['IonCastConvLSTM', 'IonCastLSTM'], default='IonCastLSTM', help='Type of model to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_evals', type=int, default=4, help='Number of samples for evaluation')
    parser.add_argument('--context_window', type=int, default=4, help='Context window size for the model')
    parser.add_argument('--prediction_window', type=int, default=1, help='Evaluation window size for the model')
    parser.add_argument('--valid_event_id', nargs='*', default=validation_events_2, help='Validation event IDs to use for evaluation at the end of each epoch')
    parser.add_argument('--valid_event_seen_id', nargs='*', default=None, help='Event IDs to use for evaluation at the end of each epoch, where the event was a part of the training set')
    parser.add_argument('--max_valid_samples', type=int, default=1000, help='Maximum number of validation samples to use for evaluation')
    parser.add_argument('--test_event_id', nargs='*', default=['G2H3-202303230900', 'G1H9-202302261800', 'G1H3-202302261800', 'G0H9-202302160900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--forecast_max_time_steps', type=int, default=48, help='Maximum number of time steps to evaluate for each test event')
    parser.add_argument('--model_file', type=str, help='Path to the model file to load for testing')
    parser.add_argument('--sun_moon_extra_time_steps', type=int, default=1, help='Number of extra time steps ahead to include in the dataset for Sun and Moon geometry')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate for the model')
    parser.add_argument('--jpld_weight', type=float, default=20.0, help='Weight for the JPLD loss in the total loss calculation')
    parser.add_argument('--save_all_models', action='store_true', help='If set, save all models during training, not just the last one')
    parser.add_argument('--save_all_channels', action='store_true', help='If set, save all channels in the forecast video, not just the JPLD channel')
    parser.add_argument('--valid_every_nth_epoch', type=int, default=1, help='Validate every nth epoch')
    parser.add_argument('--cache_dir', type=str, default=None, help='If set, build an on-disk cache for all training batches, to speed up training (WARNING: this will take a lot of disk space, ~terabytes per year)')
    
    args = parser.parse_args()
    args_cache_affecting_keys = {'data_dir', 
                                 'jpld_dir', 
                                 'celestrak_file_name', 
                                 'omniweb_dir', 
                                 'omniweb_columns', 
                                 'set_file_name', 
                                 'date_start', 
                                 'date_end', 
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

            datasets_omniweb_valid = []

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

                    datasets_omniweb_valid.append(OMNIWeb(dataset_omniweb_dir, date_start=exclusion_start, date_end=exclusion_end, column=args.omniweb_columns, return_as_image_size=(180, 360)))

            dataset_omniweb_valid = Union(datasets=datasets_omniweb_valid)

            if args.valid_event_seen_id is None:
                num_seen_events = max(2, len(args.valid_event_id))
                event_catalog_within_training_set = event_catalog.filter(date_start=date_start, date_end=date_end).exclude(date_exclusions=date_exclusions)
                if len(event_catalog_within_training_set) > 0:
                    args.valid_event_seen_id = event_catalog_within_training_set.sample(num_seen_events).ids()
                    print('\nUsing validation events seen during training: {}\n'.format(args.valid_event_seen_id))
                else:
                    print('\nNo validation events seen during training found within the training set. Using empty list.\n')
                    args.valid_event_seen_id = []

            # if args.model_type == 'VAE1':
            #     dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
            #     dataset_train = dataset_jpld_train
            #     dataset_valid = dataset_jpld_valid
            if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_jpld_valid = JPLD(dataset_jpld_dir, date_start=dataset_omniweb_valid.date_start, date_end=dataset_omniweb_valid.date_end)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_omniweb_valid.date_start, date_end=dataset_omniweb_valid.date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_celestrak_train = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end, return_as_image_size=(180, 360))
                dataset_celestrak_valid = CelesTrak(dataset_celestrak_file_name, date_start=dataset_omniweb_valid.date_start, date_end=dataset_omniweb_valid.date_end, return_as_image_size=(180, 360))
                dataset_omniweb_train = OMNIWeb(dataset_omniweb_dir, date_start=date_start, date_end=date_end, column=args.omniweb_columns, return_as_image_size=(180, 360))
                # dataset_omniweb_valid = OMNIWeb(dataset_omniweb_dir, date_start=dataset_omniweb_valid.date_start, date_end=dataset_omniweb_valid.date_end, column=args.omniweb_columns)
                dataset_set_train = SET(dataset_set_file_name, date_start=date_start, date_end=date_end, return_as_image_size=(180, 360))
                dataset_set_valid = SET(dataset_set_file_name, date_start=dataset_omniweb_valid.date_start, date_end=dataset_omniweb_valid.date_end, return_as_image_size=(180, 360))
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train, dataset_celestrak_train, dataset_omniweb_train, dataset_set_train], sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid, dataset_celestrak_valid, dataset_omniweb_valid, dataset_set_valid], sequence_length=training_sequence_length)
            else:
                raise ValueError('Unknown model type: {}'.format(args.model_type))

            print('\nTrain size: {:,}'.format(len(dataset_train)))
            print('Valid size: {:,}'.format(len(dataset_valid)))

            if args.cache_dir:
                # use the hash of the entire args object as the directory suffix for the cached dataset
                train_cache_dir = os.path.join(args.cache_dir, 'train-' + args_cache_affecting_hash)
                train_loader = CachedDataLoader(dataset_train, 
                                                batch_size=args.batch_size, 
                                                cache_dir=train_cache_dir, 
                                                num_workers=args.num_workers, 
                                                shuffle=True,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=4,
                                                name='train_loader')

                valid_cache_dir = os.path.join(args.cache_dir, 'valid-' + args_cache_affecting_hash)
                valid_loader = CachedDataLoader(dataset_valid, 
                                                batch_size=args.batch_size, 
                                                cache_dir=valid_cache_dir, 
                                                num_workers=args.num_workers, 
                                                shuffle=False,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=4,
                                                name='valid_loader')
            else:
                # No on-disk caching
                train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

                if args.max_valid_samples is not None and len(dataset_valid) > args.max_valid_samples:
                    print('Using a random subset of {:,} samples for validation'.format(args.max_valid_samples))
                    indices = random.sample(range(len(dataset_valid)), args.max_valid_samples)
                    sampler = SubsetRandomSampler(indices)
                    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
                else:
                    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

            print()

            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = glob.glob('{}/epoch-*-model.pth'.format(args.target_dir))
            if len(model_files) > 0:
                model_files.sort()
                model_file = model_files[-1]
                print('Resuming training from model file: {}'.format(model_file))
                model, optimizer, epoch, iteration, train_losses, valid_losses, scheduler_state_dict, train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses, best_valid_rmse = load_model(model_file, device)
                epoch_start = epoch + 1
                iteration = iteration + 1
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
                scheduler.load_state_dict(scheduler_state_dict)
                print('Next epoch    : {:,}'.format(epoch_start+1))
                print('Next iteration: {:,}'.format(iteration+1))
            else:
                print('Creating new model')
                total_channels = 58  # JPLD + Sun and Moon geometry + CelesTrak + OMNIWeb + SET
                # if args.model_type == 'VAE1':
                #     model = VAE1(z_dim=512, sigma_vae=False)
                if args.model_type == 'IonCastConvLSTM':
                    model = IonCastConvLSTM(input_channels=total_channels, output_channels=total_channels, context_window=args.context_window, prediction_window=args.prediction_window, dropout=args.dropout)
                elif args.model_type == 'IonCastLSTM':
                    model = IonCastLSTM(input_channels=total_channels, output_channels=total_channels, context_window=args.context_window, dropout=args.dropout)
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

                        # if args.model_type == 'VAE1':
                        #     jpld, _ = batch
                        #     jpld = jpld.to(device)

                        #     loss = model.loss(jpld)
                        if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                            jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            sunmoon_seq = sunmoon_seq.to(device)
                            celestrak_seq = celestrak_seq.to(device)
                            omniweb_seq = omniweb_seq.to(device)
                            set_seq = set_seq.to(device)

                            combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2) # Combine along the channel dimension

                            loss, rmse, jpld_rmse = model.loss(combined_seq, jpld_weight=args.jpld_weight)
                        else:
                            raise ValueError('Unknown model type: {}'.format(args.model_type))
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        iteration += 1

                        loss = loss.detach().item()
                        rmse = rmse.detach().item()
                        jpld_rmse = jpld_rmse.detach().item()

                        train_losses.append((iteration, loss))
                        train_rmse_losses.append((iteration, rmse))
                        train_jpld_rmse_losses.append((iteration, jpld_rmse))
                        pbar.set_description(f'Epoch {epoch + 1}/{args.epochs}, MSE: {loss:.4f}, RMSE: {rmse:.4f}, JPLD RMSE: {jpld_rmse:.4f}')
                        pbar.update(1)

                # Validation
                if (epoch+1) % args.valid_every_nth_epoch == 0:
                    print('*** Validation')
                    model.eval()
                    valid_loss = 0.0
                    valid_rmse_loss = 0.0
                    valid_jpld_rmse_loss = 0.0
                    with torch.no_grad():
                        for batch in tqdm(valid_loader, desc='Validation', leave=False):
                            # if args.model_type == 'VAE1':
                            #     jpld, _ = batch
                            #     jpld = jpld.to(device)
                            #     loss = model.loss(jpld)
                            if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                                jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                                jpld_seq = jpld_seq.to(device)
                                sunmoon_seq = sunmoon_seq.to(device)
                                celestrak_seq = celestrak_seq.to(device)
                                omniweb_seq = omniweb_seq.to(device)
                                set_seq = set_seq.to(device)

                                combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)  # Combine along the channel dimension
                                loss, rmse, jpld_rmse = model.loss(combined_seq, jpld_weight=args.jpld_weight)
                            else:
                                raise ValueError('Unknown model type: {}'.format(args.model_type))
                            valid_loss += loss.item()
                            valid_rmse_loss += rmse.item()
                            valid_jpld_rmse_loss += jpld_rmse.item()
                    valid_loss /= len(valid_loader)
                    valid_rmse_loss /= len(valid_loader)
                    valid_jpld_rmse_loss /= len(valid_loader)
                    valid_losses.append((iteration, valid_loss))
                    valid_rmse_losses.append((iteration, valid_rmse_loss))
                    valid_jpld_rmse_losses.append((iteration, valid_jpld_rmse_loss))
                    print(f'Validation Loss: {valid_loss:.4f}, Validation RMSE: {valid_rmse_loss:.4f}, Validation JPLD RMSE: {valid_jpld_rmse_loss:.4f}')
                    scheduler.step(valid_rmse_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Current learning rate: {current_lr:.6f}')

                    file_name_prefix = f'epoch-{epoch + 1:02d}-'

                    # Save model
                    model_file = os.path.join(args.target_dir, f'{file_name_prefix}model.pth')
                    save_model(model, optimizer, scheduler, epoch, iteration, train_losses, valid_losses, train_rmse_losses, valid_rmse_losses, train_jpld_rmse_losses, valid_jpld_rmse_losses, best_valid_rmse, model_file)
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
                    plot_file = os.path.join(args.target_dir, f'{file_name_prefix}loss.pdf')
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
                    plt.close()

                    # Plot RMSE losses
                    plot_rmse_file = os.path.join(args.target_dir, f'{file_name_prefix}metrics-rmse.pdf')
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
                    plt.close()

                    # Plot model eval results
                    model.eval()
                    with torch.no_grad():
                        num_evals = args.num_evals

                        # if args.model_type == 'VAE1':
                        #     # Set random seed for reproducibility of evaluation samples across epochs
                        #     rng_state = torch.get_rng_state()
                        #     torch.manual_seed(args.seed)

                        #     # Reconstruct a batch from the validation set
                        #     jpld_orig, jpld_orig_dates = next(iter(valid_loader))
                        #     jpld_orig = jpld_orig[:num_evals]
                        #     jpld_orig_dates = jpld_orig_dates[:num_evals]

                        #     jpld_orig = jpld_orig.to(device)
                        #     jpld_recon, _, _ = model.forward(jpld_orig)
                        #     jpld_orig_unnormalized = JPLD.unnormalize(jpld_orig)
                        #     jpld_recon_unnormalized = JPLD.unnormalize(jpld_recon)

                        #     # Sample a batch from the model
                        #     jpld_sample = model.sample(n=num_evals)
                        #     jpld_sample_unnormalized = JPLD.unnormalize(jpld_sample)
                        #     jpld_sample_unnormalized = jpld_sample_unnormalized.clamp(0, 140)
                        #     torch.set_rng_state(rng_state)
                        #     # Resume with the original random state

                        #     # Save plots
                        #     for i in range(num_evals):
                        #         date = jpld_orig_dates[i]
                        #         date_str = datetime.datetime.fromisoformat(date).strftime('%Y-%m-%d %H:%M:%S')

                        #         recon_original_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-original-{i+1:02d}.pdf')
                        #         save_gim_plot(jpld_orig_unnormalized[i][0].cpu().numpy(), recon_original_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str}')

                        #         recon_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-{i+1:02d}.pdf')
                        #         save_gim_plot(jpld_recon_unnormalized[i][0].cpu().numpy(), recon_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str} (Reconstruction)')

                        #         sample_file = os.path.join(args.target_dir, f'{file_name_prefix}sample-{i+1:02d}.pdf')
                        #         save_gim_plot(jpld_sample_unnormalized[i][0].cpu().numpy(), sample_file, vmin=0, vmax=100, title='JPLD GIM TEC (Sampled from model)')


                        if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                            max_videos_to_save = 20

                            metric_event_id = []
                            metric_jpld_rmse = []
                            metric_jpld_mae = []
                            metric_jpld_unnormalized_rmse = []
                            metric_jpld_unnormalized_mae = []
                            metric_jpld_unnormalized_rmse_low_lat = []
                            metric_jpld_unnormalized_rmse_mid_lat = []
                            metric_jpld_unnormalized_rmse_high_lat = []
                            if args.valid_event_id:
                                for i, event_id in enumerate(args.valid_event_id):
                                    save_video = i < max_videos_to_save
                                    jpld_rmse, jpld_mae, jpld_unnormalized_rmse, jpld_unnormalized_mae, jpld_unnormalized_rmse_low_lat, jpld_unnormalized_rmse_mid_lat, jpld_unnormalized_rmse_high_lat = eval_forecast(model, dataset_valid, event_catalog, event_id, file_name_prefix+'valid', save_video, args)
                                    metric_event_id.append(event_id)
                                    metric_jpld_rmse.append(jpld_rmse)
                                    metric_jpld_mae.append(jpld_mae)
                                    metric_jpld_unnormalized_rmse.append(jpld_unnormalized_rmse)
                                    metric_jpld_unnormalized_mae.append(jpld_unnormalized_mae)
                                    metric_jpld_unnormalized_rmse_low_lat.append(jpld_unnormalized_rmse_low_lat)
                                    metric_jpld_unnormalized_rmse_mid_lat.append(jpld_unnormalized_rmse_mid_lat)
                                    metric_jpld_unnormalized_rmse_high_lat.append(jpld_unnormalized_rmse_high_lat)

                            # Save metrics to a CSV file
                            metrics_file_prefix = os.path.join(args.target_dir, f'{file_name_prefix}valid-metrics')
                            save_metrics(metric_event_id, metric_jpld_rmse, metric_jpld_mae, metric_jpld_unnormalized_rmse, metric_jpld_unnormalized_mae, metric_jpld_unnormalized_rmse_low_lat, metric_jpld_unnormalized_rmse_mid_lat, metric_jpld_unnormalized_rmse_high_lat, metrics_file_prefix)

                            metric_seen_event_id = []
                            metric_seen_jpld_rmse = []
                            metric_seen_jpld_mae = []
                            metric_seen_jpld_unnormalized_rmse = []
                            metric_seen_jpld_unnormalized_mae = []
                            metric_seen_jpld_unnormalized_rmse_low_lat = []
                            metric_seen_jpld_unnormalized_rmse_mid_lat = []
                            metric_seen_jpld_unnormalized_rmse_high_lat = []
                            if args.valid_event_seen_id:
                                for i, event_id in enumerate(args.valid_event_seen_id):
                                    # produce forecasts for some events in the training set (for debugging purposes)
                                    save_video = i < max_videos_to_save
                                    jpld_rmse, jpld_mae, jpld_unnormalized_rmse, jpld_unnormalized_mae, jpld_unnormalized_rmse_low_lat, jpld_unnormalized_rmse_mid_lat, jpld_unnormalized_rmse_high_lat = eval_forecast(model, dataset_train, event_catalog, event_id, file_name_prefix+'valid-seen', save_video, args)
                                    metric_seen_event_id.append(event_id)
                                    metric_seen_jpld_rmse.append(jpld_rmse)
                                    metric_seen_jpld_mae.append(jpld_mae)
                                    metric_seen_jpld_unnormalized_rmse.append(jpld_unnormalized_rmse)
                                    metric_seen_jpld_unnormalized_mae.append(jpld_unnormalized_mae)
                                    metric_seen_jpld_unnormalized_rmse_low_lat.append(jpld_unnormalized_rmse_low_lat)
                                    metric_seen_jpld_unnormalized_rmse_mid_lat.append(jpld_unnormalized_rmse_mid_lat)
                                    metric_seen_jpld_unnormalized_rmse_high_lat.append(jpld_unnormalized_rmse_high_lat)

                            # Save metrics to a CSV file
                            seen_metrics_file_prefix = os.path.join(args.target_dir, f'{file_name_prefix}valid-seen-metrics')
                            save_metrics(metric_seen_event_id, metric_seen_jpld_rmse, metric_seen_jpld_mae, metric_seen_jpld_unnormalized_rmse, metric_seen_jpld_unnormalized_mae, metric_seen_jpld_unnormalized_rmse_low_lat, metric_seen_jpld_unnormalized_rmse_mid_lat, metric_seen_jpld_unnormalized_rmse_high_lat, seen_metrics_file_prefix)

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
                            if file.startswith(file_name_prefix) and (file.endswith('.pdf') or file.endswith('.mp4') or file.endswith('.pth') or file.endswith('.csv')):
                                shutil.copyfile(os.path.join(args.target_dir, file), os.path.join(best_model_dir, file))

        elif args.mode == 'test':

            print('*** Testing mode\n')

            model, _, _, _, _, _ = load_model(args.model_file, device)
            model.eval()
            model = model.to(device)

            dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)
            dataset_celestrak_file_name = os.path.join(args.data_dir, args.celestrak_file_name)
            training_sequence_length = args.context_window + args.prediction_window

            with torch.no_grad():
                if args.test_event_id:
                    for event_id in args.test_event_id:
                        if event_id not in event_catalog:
                            raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                        event = event_catalog[event_id]
                        date_start, date_end, max_kp = event['date_start'], event['date_end'], event['max_kp']
                        event_start = datetime.datetime.fromisoformat(date_start)
                        event_end = datetime.datetime.fromisoformat(date_end)

                        print('* Testing event ID: {}'.format(event_id))
                        date_start = event_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                        # date_forecast_start = event_start
                        date_end = event_end

                        dataset_jpld = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end)
                        dataset_sunmoon = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                        dataset_celestrak = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end, return_as_image_size=(180, 360))
                        dataset_omniweb = OMNIWeb(os.path.join(args.data_dir, args.omniweb_dir), date_start=date_start, date_end=date_end, column=args.omniweb_columns, return_as_image_size=(180, 360))
                        dataset_set = SET(os.path.join(args.data_dir, args.set_file_name), date_start=date_start, date_end=date_end, return_as_image_size=(180, 360))
                        dataset = Sequences(datasets=[dataset_jpld, dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)

                        file_name_prefix = os.path.join(args.target_dir, 'test')
    
                        eval_forecast(model, dataset, event_catalog, event_id, file_name_prefix, True, args)

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
