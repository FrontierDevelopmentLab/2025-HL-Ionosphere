import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import imageio
import numpy as np
import datetime
import torch
import os
import csv
import pandas as pd
import seaborn as sns
import random

try:
    import wandb
except ImportError:
    wandb = None

from dataset_jpld import JPLD
from model_convlstm import IonCastConvLSTM
from model_lstm import IonCastLSTM
from model_graphcast import IonCastGNN
from graphcast_utils import stack_features, sunlock_features, get_subsolar_points

matplotlib.use('Agg')

# Two main types of evaluation for an autoregressive model
# Long-horizon forecast: Predicting a sequence of images starting from a given date, using the model's autoregressive capabilities.
# Fixed-lead-time forecast: Predicting a single image at a specific future time, using the model's autoregressive capabilities.


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


def run_forecast(model, dataset, date_start, date_end, date_forecast_start, verbose, args):
    if hasattr(model, "module"):
        model = model.module
        
    if not isinstance(model, (IonCastConvLSTM)) and not isinstance(model, IonCastLSTM) and not isinstance(model, IonCastGNN) and not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError('Model must be an instance of IonCastConvLSTM or IonCastLSTM or IonCastGNN')
    if date_start > date_end:
        raise ValueError('date_start must be before date_end')
    # if hasattr(model, "module"):
    #     if date_forecast_start - datetime.timedelta(minutes=model.module.context_window * args.delta_minutes) < date_start:
    #         raise ValueError('date_forecast_start must be at least context_window * delta_minutes after date_start')
    # else:
    if date_forecast_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes) < date_start:
        raise ValueError('date_forecast_start must be at least context_window * delta_minutes after date_start')
    if date_forecast_start >= date_end:
        raise ValueError('date_forecast_start must be before date_end')
    # date_forecast_start must be an integer multiple of args.delta_minutes from date_start
    if (date_forecast_start - date_start).total_seconds() % (args.delta_minutes * 60) != 0:
        raise ValueError('date_forecast_start must be an integer multiple of args.delta_minutes from date_start')

    if verbose:
        print('Context start date : {}'.format(date_start))
        print('Forecast start date: {}'.format(date_forecast_start))
        print('End date           : {}'.format(date_end))

    if date_end > date_forecast_start + datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes):
        date_end = date_forecast_start + datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes)
        if verbose:
            print('Adjusted end date  : {} (limited by forecast_max_time_steps: {})'.format(date_end, args.forecast_max_time_steps))

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
    if verbose:
        print(f'Sequence length    : {sequence_length} ({sequence_forecast_start_index} context + {sequence_prediction_window} forecast)')

    sequence_data = dataset.get_sequence_data(sequence_dates)

    if isinstance(model, (IonCastConvLSTM)) or isinstance(model, IonCastLSTM):
        device = next(model.parameters()).device
        # The lines up until the concat is handled directly through graphcast_utils.stack_features
        jpld_seq_data = sequence_data[0]  # Original data
        sunmoon_seq_data = sequence_data[1]  # Sun and Moon geometry data
        celestrak_seq_data = sequence_data[2]  # CelesTrak data
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
   
    
    if isinstance(model, IonCastGNN):
        # Stack features will convert the sequence_dataset to output shape (B, T, C, H, W)
        timestamps = sequence_data[-1] # Get the timestamp list
        sequence_data = sequence_data[:-1] # Remove timestamp list from sequence_data
        combined_seq_batch, image_indices = stack_features(sequence_data, batched=False)

        combined_seq_batch = combined_seq_batch.to(model.device)
        combined_seq_batch = combined_seq_batch.float() # Ensure the grid nodes are in float32 

        if args.sunlock_features:
            subsolar_lats, subsolar_lons = get_subsolar_points(combined_seq_batch, timestamps, batched=False)
            subsolar_lats, subsolar_lons = subsolar_lats.to(model.device), subsolar_lons.to(model.device)
            combined_seq_batch = sunlock_features(combined_seq_batch, subsolar_lats, subsolar_lons, image_indices=image_indices, latitude_lock=False)

            combined_seq_batch = combined_seq_batch.to(model.device)
            combined_seq_batch = combined_seq_batch.float() # Ensure the grid nodes are in float32

        # Output context & forecast for all time steps, shape (B, T, C, H, W)
        if hasattr(model, "module"):
            device = next(model.module.parameters()).device
            combined_seq_batch = combined_seq_batch.to(device)
            combined_forecast = model.module.predict(
                combined_seq_batch, # .predict will mask out values not in [:, :sequence_forecast_start_index, :, :, :]
                context_window=sequence_forecast_start_index, # Context window is the number of time steps before the forecast start
                train=False # Use ground truth forcings for t+1
            )
        else:
            device = next(model.parameters()).device
            combined_seq_batch = combined_seq_batch.to(device)
            combined_forecast = model.predict(
                combined_seq_batch, # .predict will mask out values not in [:, :sequence_forecast_start_index, :, :, :]
                context_window=sequence_forecast_start_index, # Context window is the number of time steps before the forecast start
                train=False # Use ground truth forcings for t+1
            )

        combined_seq_data_original = combined_seq_batch[0, sequence_forecast_start_index:, :, :, :]  # Original data for forecast
        combined_seq_data_forecast = combined_forecast[0, sequence_forecast_start_index:, :, :, :]  # Forecast data for forecast
        combined_seq_data = combined_seq_batch[0, :, :, :, :]  # All data for the sequence


    jpld_forecast = combined_seq_data_forecast[:, 0]  # Extract JPLD channels from the forecast
    jpld_original = combined_seq_data_original[:, 0]

    jpld_original_unnormalized = JPLD.unnormalize(jpld_original) # [T, H, W]
    jpld_forecast_unnormalized = JPLD.unnormalize(jpld_forecast).clamp(0, 140)

    return jpld_forecast, jpld_original, jpld_forecast_unnormalized, jpld_original_unnormalized, combined_seq_data_original, combined_seq_data_forecast, sequence_start_date, sequence_forecast_dates, sequence_prediction_window
def plot_scatter_with_hist(gt, pred, file_name, max_points=3000, title=None):
    """
    Create a scatter plot with marginal histograms comparing ground truth vs predicted values.
    
    Parameters:
        gt (np.ndarray): Ground truth values (1D array, flattened across all pixels & frames)
        pred (np.ndarray): Predicted values (1D array, flattened across all pixels & frames)  
        file_name (str): Path to save the plot
        max_points (int): Maximum number of points to plot for performance
        title (str): Title for the plot
    """
    print(f'Saving scatter plot to {file_name}')
    
    # Sample data for performance if needed
    n_total = len(gt)
    if n_total > max_points:
        random.seed(42)  # For reproducible sampling
        sample_indices = random.sample(range(n_total), max_points)
        gt_sampled = gt[sample_indices]
        pred_sampled = pred[sample_indices]
    else:
        gt_sampled = gt
        pred_sampled = pred
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({'Ground Truth (TECU)': gt_sampled, 'Predicted (TECU)': pred_sampled})
    
    # Use seaborn jointplot - much faster and cleaner
    g = sns.jointplot(
        data=df, x='Ground Truth (TECU)', y='Predicted (TECU)',
        kind='scatter', alpha=0.4, height=8, marginal_kws=dict(bins=50, fill=True)
    )
    
    # Add light grid for better readability
    g.ax_joint.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add reference line (perfect prediction) - subtle styling
    max_val = max(np.max(gt_sampled), np.max(pred_sampled))
    min_val = min(np.min(gt_sampled), np.min(pred_sampled))
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 
                   color='black', linestyle='-', linewidth=1.5, alpha=0.7, 
                   label='Perfect Prediction')
    g.ax_joint.legend()
    
    # Add correlation coefficient (compute on sampled data for speed)
    corr = np.corrcoef(gt_sampled, pred_sampled)[0, 1]
    g.ax_joint.text(0.05, 0.95, f'R = {corr:.3f}', transform=g.ax_joint.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add title if provided
    if title:
        # Adjust figure spacing to accommodate title
        g.fig.subplots_adjust(top=0.9)  # Make room for title
        g.fig.suptitle(title, fontsize=12, y=0.95)
    
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    plt.close()

def eval_forecast_long_horizon(model, dataset, event_catalog, event_id, file_name_prefix, save_video, save_numpy, save_scatter, args):
    if event_id not in event_catalog:
        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
    event = event_catalog[event_id]
    event_start, event_end, max_kp, = event['date_start'], event['date_end'], event['max_kp']
    event_start = datetime.datetime.fromisoformat(event_start)
    event_end = datetime.datetime.fromisoformat(event_end)

    print('\n* Forecasting (Long Horizon)')
    print('Event ID           : {}'.format(event_id))
    date_start = event_start - datetime.timedelta(minutes=args.context_window * args.delta_minutes)

    date_forecast_start = event_start
    date_end = event_end
    file_name = f'{file_name_prefix}-long-horizon-event-{event_id}.mp4'
    title = f'Event: {event_id}, Kp={max_kp:.2f}'

    jpld_forecast, jpld_original, jpld_forecast_unnormalized, jpld_original_unnormalized, combined_seq_data_original, combined_seq_data_forecast, sequence_start_date, sequence_forecast_dates, sequence_prediction_window = run_forecast(model, dataset, date_start, date_end, date_forecast_start, True, args)

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
    titles_forecast = [f'JPLD GIM TEC Forecast: {d} (Long-horizon rollout from {sequence_start_date}, {forecast_mins_ahead[i]})' for i, d in enumerate(sequence_forecast_dates)]

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
    
    if save_scatter:
        # Generate scatter plot with marginal histograms
        scatter_file_name = file_name.replace('.mp4', '-scatter.pdf')
        
        # Flatten the unnormalized JPLD data
        gt_flat = jpld_original_unnormalized.cpu().numpy().flatten()
        pred_flat = jpld_forecast_unnormalized.cpu().numpy().flatten()
        
        plot_scatter_with_hist(gt_flat, pred_flat, scatter_file_name, title=fig_title)

    if save_numpy:
        # Save the numpy arrays for the main JPLD channel
        numpy_file_original = file_name.replace('.mp4', '-original.npy')
        numpy_file_forecast = file_name.replace('.mp4', '-forecast.npy')
        
        np.save(numpy_file_original, jpld_original_unnormalized.cpu().numpy())
        np.save(numpy_file_forecast, jpld_forecast_unnormalized.cpu().numpy())
        
        print(f'Saved original frames to {numpy_file_original}')
        print(f'Saved forecast frames to {numpy_file_forecast}')
    return jpld_rmse, jpld_mae, jpld_unnormalized_rmse, jpld_unnormalized_mae, jpld_unnormalized_rmse_low_lat, jpld_unnormalized_rmse_mid_lat, jpld_unnormalized_rmse_high_lat


def plot_lead_time_metrics(metrics, file_name):
    """
    Plots metrics (e.g., RMSE, MAE) as a function of lead time.
    
    Parameters:
        metrics (dict): A dictionary where keys are lead times (int) and values are dicts
                        containing 'mean' and 'std' for each metric.
        file_name (str): The path to save the plot.
    """
    print(f'Saving lead time metrics plot to {file_name}')
    
    lead_times = sorted(metrics.keys())
    
    # Extract mean and std for RMSE and MAE
    rmse_means = [metrics[lt]['rmse']['mean'] for lt in lead_times]
    rmse_stds = [metrics[lt]['rmse']['std'] for lt in lead_times]
    mae_means = [metrics[lt]['mae']['mean'] for lt in lead_times]
    mae_stds = [metrics[lt]['mae']['std'] for lt in lead_times]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- RMSE Plot ---
    ax1.errorbar(lead_times, rmse_means, yerr=rmse_stds, fmt='-o', capsize=5, label='RMSE (TECU)', alpha=0.7)
    ax1.set_ylabel('RMSE (TECU)')
    ax1.set_title('Forecast RMSE vs. Lead Time')
    ax1.grid(True, which='both', linestyle='--')
    ax1.legend()

    # --- MAE Plot ---
    ax2.errorbar(lead_times, mae_means, yerr=mae_stds, fmt='-o', capsize=5, color='tab:green', label='MAE (TECU)', alpha=0.7)
    ax2.set_xlabel('Lead Time (minutes)')
    ax2.set_ylabel('MAE (TECU)')
    ax2.set_title('Forecast MAE vs. Lead Time')
    ax2.grid(True, which='both', linestyle='--')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(file_name)

    # Also save as PNG for W&B upload
    if wandb is not None and wandb.run is not None:
        png_file = file_name.replace('.pdf', '.png')
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plot_name = os.path.splitext(os.path.basename(file_name))[0]
        try:
            wandb.log({f"plots/{plot_name}": wandb.Image(png_file)})
        except Exception as e:
            print(f"Warning: Could not upload plot {plot_name}: {e}")

    plt.close()


def eval_forecast_fixed_lead_time(model, dataset, event_catalog, event_id, lead_times_minutes, file_name_prefix, save_video, save_numpy, save_scatter, args):
    """
    Evaluates an autoregressive model at fixed lead times over a specified event period.
    """
    if event_id not in event_catalog:
        raise ValueError(f'Event ID {event_id} not found in EventCatalog')
    event = event_catalog[event_id]
    event_start = datetime.datetime.fromisoformat(event['date_start'])
    event_end = datetime.datetime.fromisoformat(event['date_end'])
    max_kp = event['max_kp']

    # Limit the evaluation period by forecast_max_time_steps
    max_duration = datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes)
    if event_end > event_start + max_duration:
        original_event_end = event_end
        event_end = event_start + max_duration
        print(f'\nAdjusted evaluation period end to {event_end} (limited by forecast_max_time_steps: {args.forecast_max_time_steps})')
        print(f'Original event end was: {original_event_end}')


    print('\n* Forecasting (Fixed Lead Time)')
    print(f'Event ID           : {event_id}')
    print(f'Evaluation Period  : {event_start} to {event_end}')
    print(f'Lead Times (mins)  : {lead_times_minutes}')

    # Dictionaries to store errors and frames for each lead time
    lead_time_errors = {lt: {'rmse': [], 'mae': []} for lt in lead_times_minutes}
    lead_time_forecast_frames = {lt: [] for lt in lead_times_minutes}
    lead_time_original_frames = {lt: [] for lt in lead_times_minutes}
    lead_time_dates = {lt: [] for lt in lead_times_minutes}

    # Iterate through each 15-minute step in the evaluation period
    current_target_date = event_start
    pbar = tqdm(total=(event_end - event_start).total_seconds() / (args.delta_minutes * 60) + 1, desc="Fixed-Lead-Time Eval")

    while current_target_date <= event_end:
        for lead_time in lead_times_minutes: # this func can take in a list of lead times to test.
            # Determine the forecast window for this specific target and lead time
            forecast_start_date = current_target_date - datetime.timedelta(minutes=lead_time)
            if hasattr(model, "module"):
                context_start_date = forecast_start_date - datetime.timedelta(minutes=model.module.context_window * args.delta_minutes) # [ context window | lead time ]
            else:
                context_start_date = forecast_start_date - datetime.timedelta(minutes=model.context_window * args.delta_minutes) # [ context window | lead time ]

            prediction_steps = lead_time // args.delta_minutes
            if prediction_steps == 0:
                continue
            if prediction_steps > args.forecast_max_time_steps:
                print(f'Skipping lead time {lead_time} minutes: exceeds forecast_max_time_steps ({args.forecast_max_time_steps} minutes)')
                continue

            try:
                _, _, jpld_forecast_unnormalized, jpld_original_unnormalized, _, _, _, _, _ = \
                    run_forecast(model, dataset, context_start_date, current_target_date, forecast_start_date, verbose=False, args=args)

                forecast_frame = jpld_forecast_unnormalized[-1]
                original_frame = jpld_original_unnormalized[-1]

                rmse = torch.nn.functional.mse_loss(forecast_frame, original_frame).sqrt().item()
                mae = torch.nn.functional.l1_loss(forecast_frame, original_frame).item()
                
                lead_time_errors[lead_time]['rmse'].append(rmse)
                lead_time_errors[lead_time]['mae'].append(mae)

                if save_video:
                    lead_time_forecast_frames[lead_time].append(forecast_frame.cpu().numpy())
                    lead_time_original_frames[lead_time].append(original_frame.cpu().numpy())
                    lead_time_dates[lead_time].append(current_target_date)

            except ValueError:
                pass
        
        current_target_date += datetime.timedelta(minutes=args.delta_minutes)
        pbar.update(1)
    
    pbar.close()

    # --- Aggregate, Save, and Plot Metrics ---
    final_metrics = {}
    print("\n--- Fixed-Lead-Time Results ---")
    print(f"{'Lead Time (min)':<20} {'Avg. RMSE (TECU)':<20} {'Std. RMSE (TECU)':<20} {'Avg. MAE (TECU)':<20} {'Std. MAE (TECU)':<20} {'Num. Samples':<15}")
    
    csv_file_name = f'{file_name_prefix}-fixed-lead-time-event-{event_id}-metrics.csv'
    csv_data = []

    for lt in sorted(lead_time_errors.keys()):
        rmse_errors = np.array(lead_time_errors[lt]['rmse'])
        mae_errors = np.array(lead_time_errors[lt]['mae'])
        num_samples = len(rmse_errors)

        if num_samples > 0:
            mean_rmse, std_rmse = np.mean(rmse_errors), np.std(rmse_errors)
            mean_mae, std_mae = np.mean(mae_errors), np.std(mae_errors)
            
            final_metrics[lt] = {
                'rmse': {'mean': mean_rmse, 'std': std_rmse},
                'mae': {'mean': mean_mae, 'std': std_mae}
            }
            
            print(f"{lt:<20} {mean_rmse:<20.4f} {std_rmse:<20.4f} {mean_mae:<20.4f} {std_mae:<20.4f} {num_samples:<15}")
            csv_data.append([lt, mean_rmse, std_rmse, mean_mae, std_mae, num_samples])
        else:
            print(f"{lt:<20} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<20} {0:<15}")

    # CSV file creation disabled for fixed-lead-time evaluation
    # print(f"\nSaving detailed metrics to {csv_file_name}")
    # with open(csv_file_name, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['lead_time_minutes', 'mean_rmse_tecu', 'std_rmse_tecu', 'mean_mae_tecu', 'std_mae_tecu', 'num_samples'])
    #     writer.writerows(csv_data)
    # Per-event plot generation disabled - metrics will be aggregated across all events
    
    # --- Save Videos if Requested ---
    if save_video:
        print("\n--- Generating Fixed-Lead-Time Videos ---")
        
        for lt in sorted(lead_time_forecast_frames.keys()):
            forecast_frames = lead_time_forecast_frames[lt]
            original_frames = lead_time_original_frames[lt]
            dates = lead_time_dates[lt]
            
            if not forecast_frames:
                continue

            video_file_name = f'{file_name_prefix}-fixed-lead-time-event-{event_id}-{lt}min.mp4'

            titles_top = [f'JPLD GIM TEC Ground Truth: {d}' for d in dates]
            titles_bottom = [f'JPLD GIM TEC Forecast: {d} ({lt}-min fixed lead time)' for d in dates]
            
            #fig_title = title + f' - RMSE: {jpld_unnormalized_rmse:.2f} TECU - MAE: {jpld_unnormalized_mae:.2f} TECU'
            mean_rmse = np.mean(lead_time_errors[lt]['rmse'])
            mean_mae = np.mean(lead_time_errors[lt]['mae'])
            fig_title = f"Event: {event_id} - Kp={max_kp} - RMSE: {mean_rmse:.2f} TECU - MAE: {mean_mae:.2f} TECU"

            save_gim_video_comparison(
                gim_sequence_top=np.array(original_frames),
                gim_sequence_bottom=np.array(forecast_frames),
                file_name=video_file_name,
                vmin=0, vmax=120,
                titles_top=titles_top,
                titles_bottom=titles_bottom,
                fig_title=fig_title
            )

            # Generate scatter plot for this lead time if requested
            if save_scatter:
                scatter_file_name = video_file_name.replace('.mp4', '-scatter.pdf')
                
                # Flatten the data
                gt_flat = np.array(original_frames).flatten()
                pred_flat = np.array(forecast_frames).flatten()
                
                plot_scatter_with_hist(gt_flat, pred_flat, scatter_file_name, title=fig_title)

    
    # --- Save NumPy arrays if Requested ---
    if save_numpy:
        print("\n--- Saving Fixed-Lead-Time NumPy Arrays ---")
        
        for lt in sorted(lead_time_forecast_frames.keys()):
            forecast_frames = lead_time_forecast_frames[lt]
            original_frames = lead_time_original_frames[lt]
            
            if not forecast_frames:
                continue

            numpy_file_original = f'{file_name_prefix}-fixed-lead-time-event-{event_id}-{lt}min-original.npy'
            numpy_file_forecast = f'{file_name_prefix}-fixed-lead-time-event-{event_id}-{lt}min-forecast.npy'
            
            np.save(numpy_file_original, np.array(original_frames))
            np.save(numpy_file_forecast, np.array(forecast_frames))
            
            print(f'Saved original frames to {numpy_file_original}')
            print(f'Saved forecast frames to {numpy_file_forecast}')
    
    # Return metrics data for aggregation
    return lead_time_errors, event_id


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

    # Compute xlim for each column (metric) based on all data for that metric
    col_xlims = []
    col_bins = []
    num_bins = 15
    for col, (key, _) in enumerate(metric_names):
        all_data = []
        for prefix in prefixes:
            data = metrics_dict[prefix][key]
            if len(data) > 0:
                all_data.append(data)
        if all_data:
            all_data = np.concatenate(all_data)
            xmin = np.nanmin(all_data)
            xmax = np.nanmax(all_data)
            # Add a small margin
            xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
            xlim = (xmin - xpad, xmax + xpad)
            col_xlims.append(xlim)
            col_bins.append(np.linspace(xlim[0], xlim[1], num_bins + 1))
        else:
            col_xlims.append((0, 1))
            col_bins.append(np.linspace(0, 1, num_bins + 1))

    for row, prefix in enumerate(prefixes):
        metrics = metrics_dict[prefix]
        color = prefix_colors.get(prefix, 'black')
        for col, (key, title) in enumerate(metric_names):
            ax = axes[row, col]
            data = metrics[key]
            if len(data) > 0:
                ax.hist(data, bins=col_bins[col], alpha=0.7, color=color, edgecolor='black')
                ax.axvline(np.mean(data), color='gray', linestyle='dashed', linewidth=1, label='Mean')
                ax.set_xlim(col_xlims[col])
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    if prefix == 'all':
                        ax.set_ylabel('All events')
                    else:
                        ax.set_ylabel(f'{prefix} events')
                ax.legend()
                # Set plain formatting for x-axis, no scientific notation, no offset
                ax.ticklabel_format(style='plain', axis='x', useOffset=False)
                # Optionally, set a fixed number of decimals for readability
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            else:
                # No data: turn off axis
                ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_name_hist)

    # Also save as PNG for W&B upload
    if wandb is not None and wandb.run is not None:
        png_file = file_name_hist.replace('.pdf', '.png')
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plot_name = os.path.splitext(os.path.basename(file_name_hist))[0]
        try:
            wandb.log({f"plots/{plot_name}": wandb.Image(png_file)})
        except Exception as e:
            print(f"Warning: Could not upload plot {plot_name}: {e}")
    
    plt.close()

def aggregate_and_plot_fixed_lead_time_metrics(all_lead_time_errors, all_event_ids, file_name_prefix):
    """
    Aggregate metrics from multiple fixed-lead-time evaluations and create a single plot
    showing mean and std of RMSE and MAE versus lead times for each event category.
    
    Args:
        all_lead_time_errors: List of lead_time_errors dictionaries from eval_forecast_fixed_lead_time calls
        all_event_ids: List of event_ids corresponding to each lead_time_errors
        file_name_prefix: Prefix for output file names
    """
    # save a metrics figure (pdf) with four histograms of all metrics for all events
    file_name_hist = os.path.join(file_name_prefix + '-histograms.pdf')
    print(f'Saving metrics histograms to {file_name_hist}')

    if not all_lead_time_errors:
        print("No fixed-lead-time metrics to aggregate")
        return
    
    # Combine all metrics by event category
    category_metrics = {}
    
    for lead_time_errors, event_id in zip(all_lead_time_errors, all_event_ids):
        # Determine event category from event_id (first two characters: G0, G1, G2, etc.)
        category = event_id[:2] if len(event_id) >= 2 else "Unknown"
        
        if category not in category_metrics:
            category_metrics[category] = {
                'lead_times': [],
                'rmse_values': [],
                'mae_values': []
            }
        
        # Extract metrics for each lead time
        for lead_time_minutes, metrics in lead_time_errors.items():
            # lead_time_minutes is already an integer (e.g., 60, 120, 180)
            # metrics['rmse'] and metrics['mae'] are lists of values for this lead time
            
            rmse_list = metrics['rmse']
            mae_list = metrics['mae']
            
            # Add each individual measurement to the category metrics
            # Both lists should have the same length
            for rmse_val, mae_val in zip(rmse_list, mae_list):
                category_metrics[category]['lead_times'].append(lead_time_minutes)
                category_metrics[category]['rmse_values'].append(rmse_val)
                category_metrics[category]['mae_values'].append(mae_val)
    
    # Convert to numpy arrays and compute statistics for each category
    category_stats = {}
    for category, data in category_metrics.items():
        # Group by lead time to compute mean and std
        lead_times = np.array(data['lead_times'])
        rmse_values = np.array(data['rmse_values'])
        mae_values = np.array(data['mae_values'])
        
        unique_lead_times = np.unique(lead_times)
        rmse_means = []
        rmse_stds = []
        mae_means = []
        mae_stds = []
        
        for lt in unique_lead_times:
            mask = lead_times == lt
            rmse_means.append(np.mean(rmse_values[mask]))
            rmse_stds.append(np.std(rmse_values[mask]))
            mae_means.append(np.mean(mae_values[mask]))
            mae_stds.append(np.std(mae_values[mask]))
        
        category_stats[category] = {
            'lead_times': unique_lead_times,
            'rmse_mean': np.array(rmse_means),
            'rmse_std': np.array(rmse_stds),
            'mae_mean': np.array(mae_means),
            'mae_std': np.array(mae_stds)
        }
    
    # Create the aggregated plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    
    # Plot RMSE
    for category, stats in category_stats.items():
        color = prefix_colors.get(category, 'black')  # Default to black if category not found
        ax1.errorbar(stats['lead_times'], stats['rmse_mean'], yerr=stats['rmse_std'], 
                    label=category, color=color, capsize=5, marker='o')
    
    ax1.set_xlabel('Lead Time (minutes)')
    ax1.set_ylabel('RMSE (TECU)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE
    for category, stats in category_stats.items():
        color = prefix_colors.get(category, 'black')  # Default to black if category not found
        ax2.errorbar(stats['lead_times'], stats['mae_mean'], yerr=stats['mae_std'], 
                    label=category, color=color, capsize=5, marker='o')
    
    ax2.set_xlabel('Lead Time (minutes)')
    ax2.set_ylabel('MAE (TECU)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file_name = file_name_prefix + '-fixed-lead-time-metrics.pdf'
    plt.savefig(plot_file_name, bbox_inches='tight')
    print(f'Fixed-lead-time aggregated metrics plot saved to {plot_file_name}')


    # Also save as PNG for W&B upload
    if wandb is not None and wandb.run is not None:
        png_file = file_name_hist.replace('.pdf', '.png')
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plot_name = os.path.splitext(os.path.basename(file_name_hist))[0]
        try:
            wandb.log({f"plots/{plot_name}": wandb.Image(png_file)})
        except Exception as e:
            print(f"Warning: Could not upload plot {plot_name}: {e}")
    
    plt.close()

    # Save aggregated metrics to CSV file
    csv_file_name = file_name_prefix + '-fixed-lead-time-metrics.csv'
    print(f'Saving aggregated fixed-lead-time metrics to {csv_file_name}')
    
    with open(csv_file_name, 'w', newline='') as csvfile:
        fieldnames = ['event_category', 'lead_time_minutes', 'mean_rmse_tecu', 'std_rmse_tecu', 'mean_mae_tecu', 'std_mae_tecu', 'num_samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data for each category and lead time
        for category, stats in category_stats.items():
            for i, lead_time in enumerate(stats['lead_times']):
                # Count samples for this category and lead time
                lead_times_array = np.array(category_metrics[category]['lead_times'])
                num_samples = np.sum(lead_times_array == lead_time)
                
                writer.writerow({
                    'event_category': category,
                    'lead_time_minutes': lead_time,
                    'mean_rmse_tecu': stats['rmse_mean'][i],
                    'std_rmse_tecu': stats['rmse_std'][i],
                    'mean_mae_tecu': stats['mae_mean'][i],
                    'std_mae_tecu': stats['mae_std'][i],
                    'num_samples': num_samples
                })
        
        # Add summary statistics across all categories
        # writer.writerow({})  # Empty row for separation
        # writer.writerow({'event_category': '--- SUMMARY ACROSS ALL CATEGORIES ---'})
        
        # Compute overall statistics across all categories for each lead time
        all_lead_times = set()
        for stats in category_stats.values():
            all_lead_times.update(stats['lead_times'])
        
        for lead_time in sorted(all_lead_times):
            # Collect all RMSE and MAE values for this lead time across all categories
            all_rmse_for_lt = []
            all_mae_for_lt = []
            total_samples = 0
            
            for category, data in category_metrics.items():
                lead_times_array = np.array(data['lead_times'])
                rmse_array = np.array(data['rmse_values'])
                mae_array = np.array(data['mae_values'])
                
                mask = lead_times_array == lead_time
                all_rmse_for_lt.extend(rmse_array[mask])
                all_mae_for_lt.extend(mae_array[mask])
                total_samples += np.sum(mask)
            
            if total_samples > 0:
                writer.writerow({
                    'event_category': 'all',
                    'lead_time_minutes': lead_time,
                    'mean_rmse_tecu': np.mean(all_rmse_for_lt),
                    'std_rmse_tecu': np.std(all_rmse_for_lt),
                    'mean_mae_tecu': np.mean(all_mae_for_lt),
                    'std_mae_tecu': np.std(all_mae_for_lt),
                    'num_samples': total_samples
                })
    
    print(f"Aggregated fixed-lead-time metrics for {len(category_stats)} event categories: {list(category_stats.keys())}")
