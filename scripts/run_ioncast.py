"""
Monday todos:
- Get wandb logging from Frank
- Integrate quasi-dipole dataset from Halil
- Get new cached dataset implementation from Gunes
- Set large run using these new chang1es
- Merge with main branch (our run.py)
"""

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
import shutil

from util import Tee
from util import set_random_seed
from util import stack_as_channels
# from model_vae import VAE1
from model_convlstm import IonCastConvLSTM
from model_lstm import IonCastLSTM
from model_graphcast import IonCastGNN
from graphcast_utils import stack_features, calc_shapes_for_stack_features
from dataset_jpld import JPLD
from dataset_sequences import Sequences
from dataset_union import Union
from dataset_sunmoongeometry import SunMoonGeometry
from dataset_quasidipole import QuasiDipole
from dataset_celestrak import CelesTrak
from dataset_omniweb import OMNIWeb, omniweb_all_columns
from dataset_set import SET, set_all_columns
from dataset_cached import CachedDataset
from events import EventCatalog
from plot_functions import save_gim_plot, save_gim_video, save_gim_video_comparison

FIXED_CADENCE = 15 # mins

matplotlib.use('Agg')
def run_n_step_prediction(model, ground_truth_sequence, context_window, n_steps=4, return_init_context=False):
    B, T, C, H, W = ground_truth_sequence.shape
    all_preds = torch.zeros_like(ground_truth_sequence)
    if return_init_context:
        # note that the first context_window+n_steps window wont be usable as to get the first t+n_steps pred need 
        # to have a) passed in context_window worth of ground truth, then b) compute n_steps autoregressive steps 
        # to get out your frame, hense the ground truth in-filling up untill [0, context_window+n_steps) (not inclusive)
        all_preds[:, :context_window+n_steps, :, :, :] = ground_truth_sequence[:, :context_window+n_steps, :, :, :] 
    if isinstance(model, IonCastGNN):
        for i in range(T-context_window-n_steps): # There will only be T-context_window-n_steps new frames generated for the same reason as above
            input_grid = ground_truth_sequence[:, i:context_window+n_steps+i, :, :, :] # note that outside of the context window, predict masks 
                                                                                       # even though we pass in the full context_window+n_steps 
                                                                                       # ground truth frames
            output_grid = model.predict(input_grid, context_window, train=False)
            n_step_pred = output_grid[:, -1, :, :] # we only care about the t + n_steps prediction
            all_preds[:, i+context_window+n_steps] = n_step_pred # NOTE: check if this is an off by one error

    else: 
        raise NotImplementedError("currently only IonCastGNN is supported")
    return all_preds
    # pass

# model = MyModel()
# model = torch.nn.DataParallel(model)  # Wrap model
# model = model.to(device)  # usually 'cuda' or 'cuda:0'

def run_forecast(model, dataset, date_start, date_end, date_forecast_start, title, file_name, args): 
    # Checks
    if not isinstance(model, (IonCastConvLSTM, IonCastLSTM, IonCastGNN)):
        raise ValueError('Model must be an instance of IonCastConvLSTM, IonCastLSTM, or IonCastGNN')
    if date_start > date_end:
        raise ValueError('date_start must be before date_end')
    if date_forecast_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes) < date_start:
        raise ValueError('date_forecast_start must be at least context_window * delta_minutes after date_start')
    if date_forecast_start >= date_end:
        raise ValueError('date_forecast_start must be before date_end')
    
    # date_forecast_start must be an integer multiple of args.delta_minutes from date_start
    if (date_forecast_start - date_start).total_seconds() % (args.delta_minutes * 60) != 0:
        raise ValueError('date_forecast_start must be an integer multiple of args.delta_minutes from date_start')

    # Get forecast sequence and prediction window
    print('Context start date : {}'.format(date_start))
    print('Forecast start date: {}'.format(date_forecast_start))
    print('End date           : {}'.format(date_end))

    if date_end > date_forecast_start + datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes):
        date_end = date_forecast_start + datetime.timedelta(minutes=args.forecast_max_time_steps * args.delta_minutes)
        print('Adjusted end date  : {} ({} time steps after forecast start)'.format(date_end, args.forecast_max_time_steps))

    sequence_start = date_start
    sequence_end = date_end
    sequence_length = int((sequence_end - sequence_start).total_seconds() / 60 / args.delta_minutes)
    sequence = [sequence_start + datetime.timedelta(minutes=args.delta_minutes * i) for i in range(sequence_length)]

    # find the index of the date_forecast_start in the list sequence
    if date_forecast_start not in sequence:
        raise ValueError('date_forecast_start must be in the sequence')
    sequence_forecast_start_index = sequence.index(date_forecast_start)
    sequence_prediction_window = sequence_length - (sequence_forecast_start_index) # TODO: should this be sequence_length - (sequence_forecast_start_index + 1)
    sequence_forecast = sequence[sequence_forecast_start_index:]
    print(f'Sequence length    : {sequence_length} ({sequence_forecast_start_index} context + {sequence_prediction_window} forecast)')

    # Get the sequence_data
    if isinstance(dataset, CachedDataset):
        dataset = dataset.dataset
    sequence_data = dataset.get_sequence_data(sequence)

    device = next(model.parameters()).device

    # If IonCastConvLSTM, load data and concatenate along channel dimension
    if isinstance(model, (IonCastConvLSTM, IonCastLSTM)):
        # Get separated datasets
        jpld_seq = sequence_data[0]  # Original data
        sunmoon_seq = sequence_data[1]  # Sun and Moon geometry data
        celestrak_seq = sequence_data[2]  # CelesTrak data
        omniweb_seq = sequence_data[3]  # OMNIWeb data
        set_seq = sequence_data[4]  # SET data

        # Send to device
        jpld_seq = jpld_seq.to(device) # sequence_length, channels, 180, 360
        sunmoon_seq = sunmoon_seq.to(device) # sequence_length, channels, 180, 360
        celestrak_seq = celestrak_seq.to(device) # sequence_length, channels, 180, 360
        omniweb_seq = omniweb_seq.to(device)  # sequence_length, channels, 180, 360
        set_seq = set_seq.to(device)  # sequence_length, channels, 180, 360

        # Expand
        celestrak_seq = celestrak_seq.view(celestrak_seq.shape + (1, 1)).expand(-1, 2, 180, 360)
        omniweb_seq = omniweb_seq.view(omniweb_seq.shape + (1, 1)).expand(-1, 10, 180, 360)
        set_seq = set_seq.view(set_seq.shape + (1, 1)).expand(-1, 9, 180, 360)

        combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=1)  # Combine along the channel dimension
        
        combined_seq_context = combined_seq[:sequence_forecast_start_index]  # Context data for forecast
        combined_seq_original = combined_seq[sequence_forecast_start_index:]  # Original data for forecast
        combined_seq_forecast = model.predict(combined_seq_context.unsqueeze(0), prediction_window=sequence_prediction_window).squeeze(0) # Only forecast (not context + forecast)
    
    # If IonCastGNN, pass sequence data to stack_features
    if isinstance(model, IonCastGNN):
        # Stack features will convert the sequence_dataset to output shape (B, T, C, H, W)
        sequence_data = sequence_data[:-1] # Remove timestamp list from sequence_data
        combined_seq_batch = stack_features(
            sequence_data, 
            image_size=(180, 360),
            batched=False
        ) # [1, T, C, H, W]

        combined_seq_batch = combined_seq_batch.to(device)
        combined_seq_batch = combined_seq_batch.float() # Ensure the grid nodes are in float32 

        # Output context & forecast for all time steps, shape (B, T, C, H, W)
        combined_forecast = model.predict(
            combined_seq_batch, # .predict will mask out values not in [:, :sequence_forecast_start_index, :, :, :]
            context_window=sequence_forecast_start_index, # Context window is the number of time steps before the forecast start
            train=False # Use ground truth forcings for t+1
        )

        combined_seq_original = combined_seq_batch[0, sequence_forecast_start_index:, :, :, :]  # Original data for forecast
        combined_seq_forecast = combined_forecast[0, sequence_forecast_start_index:, :, :, :]  # Forecast data for forecast
        combined_seq = combined_seq_batch[0, :, :, :, :]  # All data for the sequence

        N_STEPS = 4
        n_step_preds = run_n_step_prediction(
            model=model, 
            ground_truth_sequence=combined_seq_batch, 
            context_window=sequence_forecast_start_index, 
            n_steps=N_STEPS # NOTE: HARDCODED FOR NOW
            )

        jpld_n_step_preds = n_step_preds[0, sequence_forecast_start_index+N_STEPS:, 0] # take 1st batch (already single batch) and JPLD channel
        jpld_n_step_preds_unnormalized = JPLD.unnormalize(jpld_n_step_preds).clamp(0, 140)


    # Extract JPLD & unnormalize
    jpld_forecast = combined_seq_forecast[:, 0]  # Extract JPLD channels from the forecast
    jpld_original = combined_seq_original[:, 0]

    jpld_original_unnormalized = JPLD.unnormalize(jpld_original) # Unnormalize
    jpld_forecast_unnormalized = JPLD.unnormalize(jpld_forecast).clamp(0, 140)

    # rmse between original and forecast
    jpld_rmse = torch.nn.functional.mse_loss(jpld_forecast_unnormalized, jpld_original_unnormalized, reduction='mean').sqrt().item()
    print('\033[92mRMSE (TECU)        : {}\033[0m'.format(jpld_rmse))
    jpld_mae = torch.nn.functional.l1_loss(jpld_forecast_unnormalized, jpld_original_unnormalized, reduction='mean').item()
    print('\033[96mMAE (TECU)         : {}\033[0m'.format(jpld_mae))

    # Create title for the video
    fig_title = title + f' - RMSE: {jpld_rmse:.2f} TECU - MAE: {jpld_mae:.2f} TECU'
    forecast_mins_ahead = ['{} mins'.format((j + 1) * 15) for j in range(sequence_prediction_window)]
    titles_original = [f'JPLD GIM TEC Ground Truth: {d}' for d in sequence_forecast]
    titles_forecast = [f'JPLD GIM TEC Forecast: {d} - Autoregressive rollout from {sequence_start} ({forecast_mins_ahead[i]})' for i, d in enumerate(sequence_forecast)]

    # Create JPLD video comparison
    save_gim_video_comparison(
        gim_sequence_top=jpld_original_unnormalized.cpu().numpy().reshape(-1, 180, 360),
        gim_sequence_bottom=jpld_forecast_unnormalized.cpu().numpy().reshape(-1, 180, 360),
        file_name=file_name,
        vmin=0, vmax=120,
        titles_top=titles_original,
        titles_bottom=titles_forecast,
        fig_title=fig_title
    )

    if isinstance(model, IonCastGNN): # hacky solution, maybe reformat run_forecast a bit to deal with this a bit better

        print(f"n_step_preds.shape: {n_step_preds.shape}")
        print(f"combined_seq_batch.shape: {combined_seq_batch.shape}")
        # print(f"combined_seq_batch.shape: {combined_seq_batch.shape}")
        print(f"jpld_original_unnormalized.shape: {jpld_original_unnormalized.shape}")
        print(f"jpld_n_step_preds.shape: {jpld_n_step_preds.shape}")

        # Create title for the video
        # fig_title = title + f' - RMSE: {jpld_rmse:.2f} TECU - MAE: {jpld_mae:.2f} TECU' 
        fig_title_n_step = title # NOTE: will need to recalculate RMSE TECU MAE etc for n_step preds
        titles_n_step = [f'JPLD GIM TEC Forecast: {d} - Autoregressive rollout from {sequence_start} ({N_STEPS * 15} mins)' for i, d in enumerate(sequence_forecast)]

        file_name_no_ext, file_ext = os.path.splitext(file_name)       # e.g., "example.txt"

        save_gim_video_comparison(
            gim_sequence_top=jpld_original_unnormalized[N_STEPS:].cpu().numpy().reshape(-1, 180, 360),
            gim_sequence_bottom=jpld_n_step_preds_unnormalized.cpu().numpy().reshape(-1, 180, 360),
            file_name=file_name_no_ext + f"_{N_STEPS}_step" + file_ext,
            vmin=0, vmax=120,
            titles_top=titles_original,
            titles_bottom=titles_n_step,
            fig_title=fig_title_n_step
        )

    # If save_all_channels is True, save a video comparison for each channel
    if args.save_all_channels:
        num_channels = combined_seq.shape[1]
        for i in range(num_channels):
            channel_original = combined_seq_original[:, i]
            channel_forecast = combined_seq_forecast[:, i]
            channel_original_unnormalized = channel_original
            channel_forecast_unnormalized = channel_forecast

            titles_channel_original = [f'Channel {i} Original: {d} - {title}' for d in sequence_forecast]
            titles_channel_forecast = [f'Channel {i} Forecast: {d} ({forecast_mins_ahead[i]}) - {title}' for i, d in enumerate(sequence_forecast)]

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
    elif isinstance(model, IonCastGNN):
        checkpoint = {
            'model': 'IonCastGNN',
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
            'input_dim_grid_nodes': model.input_dim_grid_nodes,  # Number of features per grid node
            'output_dim_grid_nodes': model.output_dim_grid_nodes,  # Number of features to predict per grid node
            'hidden_dim': model.hidden_dim,
            'hidden_layers': model.hidden_layers,
            'processor_layers': model.processor_layers,
            'mesh_level': model.mesh_level,
            'processor_type': model.processor_type,
            'num_attention_heads': model.num_attention_heads,
            'khop_neighbors': model.khop_neighbors,
            'input_dim_mesh_nodes': model.input_dim_mesh_nodes,  # Number of features per mesh node
            'input_dim_edges': model.input_dim_edges,  # Number of features per edge
            'aggregation': model.aggregation,
            'activation_fn': model.activation_fn,
            'norm_type': model.norm_type,
            'input_res': model.input_res,  # Input resolution (height, width)
            'context_window': model.context_window,
            'forcing_channels': model.forcing_channels,  # List of forcing channels
        }
    else:
        raise ValueError('Unknown model type: {}'.format(model))
    torch.save(checkpoint, file_name)


def load_model(file_name, device):
    # old_checkpoint = torch.load("/home/jupyter/halil_debug/ioncastgnn-train-may-2015-sanity-check-new-runpy/epoch-33-model.pth") Temporary to get the chekpoint file for the hacked in 726 epoch model to work nicely with the new run code, note this is ofcourse going to lead to a messed up scheduler and other stuff but was meant more so for testing the n_step videos.
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
    elif checkpoint["model"] == "IonCastGNN": 
        pprint.pprint(checkpoint.keys())
        mesh_level = checkpoint["mesh_level"]
        input_dim_grid_nodes = checkpoint["input_dim_grid_nodes"]
        output_dim_grid_nodes = checkpoint["output_dim_grid_nodes"] 
        processor_type = checkpoint["processor_type"]
        processor_layers = checkpoint["processor_layers"]
        hidden_layers = checkpoint["hidden_layers"]
        context_window = checkpoint["context_window"]
        hidden_dim = checkpoint["hidden_dim"]
        forcing_channels = checkpoint["forcing_channels"] if "forcing_channels" in checkpoint else None
        num_attention_heads = checkpoint.get("num_attention_heads", 4)  # Default to 4 if not specified
        khop_neighbors = checkpoint.get("khop_neighbors", 32)  # Default to 32 if not specified
        input_dim_mesh_nodes = checkpoint.get("input_dim_mesh_nodes", 3)  # Default to 3 if not specified
        input_dim_edges = checkpoint.get("input_dim_edges", 4)  # Default to 4 if not specified
        input_res = checkpoint.get("input_res", (180, 360))  # Default to (180, 360) if not specified
        aggregation = checkpoint.get("aggregation", "sum")  # Default to "sum" if not specified
        activation_fn = checkpoint.get("activation_fn", "silu")  # Default to "sum" if not specified
        norm_type = checkpoint.get("norm_type", "LayerNorm")  # Default to "LayerNorm" if not specified

        model = IonCastGNN(
            mesh_level = mesh_level,
            input_res = input_res,
            input_dim_grid_nodes = input_dim_grid_nodes, # IMPORTANT! Based on how many features are stacked in the input.
            output_dim_grid_nodes = output_dim_grid_nodes, 
            input_dim_mesh_nodes = input_dim_mesh_nodes, # GraphCast used 3: cos(lat), sin(lon), cos(lon)
            input_dim_edges = input_dim_edges, # GraphCast used 4: length(edge), vector diff b/w 3D positions of sender and receiver nodes in coordinate system of the reciever
            processor_type = processor_type, # Options: "MessagePassing" or "GraphTransformer", i.e. GraphCast vs. GenCast
            khop_neighbors = khop_neighbors,
            num_attention_heads = num_attention_heads,
            processor_layers = processor_layers,
            hidden_layers = hidden_layers,
            hidden_dim = hidden_dim,
            aggregation = aggregation,
            activation_fn = activation_fn,
            norm_type = norm_type,
            context_window=context_window,
            device=device,
            forcing_channels=forcing_channels,  # List of forcing channels to predict
        )
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
    # scheduler_state_dict = checkpoint['scheduler_state_dict']
    # scheduler_state_dict = old_checkpoint['scheduler_state_dict'] 
    # train_rmse_losses = old_checkpoint['train_rmse_losses']
    # valid_rmse_losses = old_checkpoint['valid_rmse_losses']
    # train_jpld_rmse_losses = old_checkpoint['train_jpld_rmse_losses']
    # valid_jpld_rmse_losses = old_checkpoint['valid_jpld_rmse_losses']
    # best_valid_rmse = old_checkpoint['best_valid_rmse']
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
    parser.add_argument('--quasidipole_dir', type=str, default='quasi_dipole', help='QuasiDipole dataset directory')
    parser.add_argument('--set_file_name', type=str, default='set/karman-2025_data_sw_data_set_sw.csv', help='SET dataset file name')
    parser.add_argument('--aux_datasets', nargs='+', choices=["sunmoon", "omni", "celestrak", "set", "quasidipole"], default=["sunmoon", "omni", "celestrak", "set", "quasidipole"], help="additional datasets to include on top of TEC maps")
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    # parser.add_argument('--date_start', type=str, default='2010-05-13T00:00:00', help='Start date')
    # parser.add_argument('--date_end', type=str, default='2024-08-01T00:00:00', help='End date')
    parser.add_argument('--date_start', type=str, default='2020-04-19T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-04-22T00:00:00', help='End date')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time step in minutes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--model_type', type=str, choices=['IonCastConvLSTM', 'IonCastLSTM', 'IonCastGNN'], default='IonCastLSTM', help='Type of model to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_evals', type=int, default=4, help='Number of samples for evaluation')
    parser.add_argument('--context_window', type=int, default=4, help='Context window size for the model')
    parser.add_argument('--prediction_window', type=int, default=1, help='Evaluation window size for the model')
    parser.add_argument('--valid_event_id', nargs='*', default=['G2H3-202303230900', 'G1H9-202302261800', 'G1H3-202302261800', 'G0H9-202302160900'], help='Validation event IDs to use for evaluation at the end of each epoch')
    parser.add_argument('--valid_event_seen_id', nargs='*', default=['G0H3-202404192100'], help='Event IDs to use for evaluation at the end of each epoch, where the event was a part of the training set')
    parser.add_argument('--test_event_id', nargs='*', default=['G2H3-202303230900', 'G1H9-202302261800', 'G1H3-202302261800', 'G0H9-202302160900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--forecast_max_time_steps', type=int, default=48, help='Maximum number of time steps to evaluate for each test event')
    parser.add_argument('--model_file', type=str, help='Path to the model file to load for testing')
    parser.add_argument('--sun_moon_extra_time_steps', type=int, default=0, help='Number of extra time steps ahead to include in the dataset for Sun and Moon geometry')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate for the model')
    parser.add_argument('--jpld_weight', type=float, default=20.0, help='Weight for the JPLD loss in the total loss calculation')
    parser.add_argument('--save_all_models', action='store_true', help='If set, save all models during training, not just the last one')
    parser.add_argument('--save_all_channels', action='store_true', help='If set, save all channels in the forecast video, not just the JPLD channel')
    parser.add_argument('--cache_datasets', action='store_true', help='If set, pre-load and cache datasets in memory to speed up training')
    parser.add_argument('--valid_every_nth_epoch', type=int, default=1, help='Validate every nth epoch')
    parser.add_argument('--mesh_level', type=int, default=6, help='Mesh level for IonCastGNN model')
    parser.add_argument('--processor_type', type=str, choices=['MessagePassing', 'GraphTransformer'], default='MessagePassing', help='Processor type for IonCastGNN model')
    parser.add_argument('--ioncast_hidden_dim', type=int, default=512, help='Hidden dimension for IonCastGNN model')
    parser.add_argument('--ioncast_hidden_layers', type=int, default=1, help='Number of hidden layers for IonCastGNN model')
    parser.add_argument('--ioncast_processor_layers', type=int, default=6, help='Number of processor layers for IonCastGNN model')
    parser.add_argument('--train_on_predicted_forcings', action='store_true', help='Train on predicted forcings for IonCastGNN model')

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(args.target_dir, f'log_{current_time}.txt')

    set_random_seed(args.seed)
    device = torch.device(args.device)
    print('Using device:', device)

    dataset_constructors = {
            'sunmoon': lambda date_start_=None, date_end_=None, date_exclusions_=None, column_=None: SunMoonGeometry(date_start=date_start_, date_end=date_end_, normalize=True, extra_time_steps=args.sun_moon_extra_time_steps), # Note: no date_exclusions and also extra_time_steps should be 1 for IonCastGNN
            'omni': lambda date_start_=None, date_end_=None, date_exclusions_=None, column_=omniweb_all_columns: OMNIWeb(data_dir=dataset_omniweb_dir, date_start=date_start_, date_end=date_end_, normalize=True, date_exclusions=date_exclusions_, delta_minutes=15, column=column_),
            'celestrak': lambda date_start_=None, date_end_=None, date_exclusions_=None, column_=['Kp', 'Ap']: CelesTrak(file_name=dataset_celestrak_file_name, date_start=date_start_, date_end=date_end_, normalize=True, date_exclusions=date_exclusions_, delta_minutes=15, column=column_),
            'set': lambda date_start_=None, date_end_=None, date_exclusions_=None, column_=set_all_columns: SET(file_name=dataset_set_file_name, date_start=date_start_, date_end=date_end_, normalize=True, date_exclusions=date_exclusions_, delta_minutes=15,column=column_),
            'quasidipole': lambda date_start_=None, date_end_=None, date_exclusions_=None: QuasiDipole(data_dir=dataset_qd_dir, date_start=date_start_, date_end=date_end_, delta_minutes=FIXED_CADENCE),
        }
    
    # Set up the log file
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

            # Checks
            if args.batch_size < args.num_evals:
                print(f'Warning: Batch size {args.batch_size} is less than num_evals {args.num_evals}. Using the batch size for num_evals.')
                args.num_evals = args.batch_size

            if (args.model_type == 'IonCastGNN') & (args.batch_size != 1):
                raise ValueError(f'model type {args.model_type} requires batch size {args.batch_size} (IonCastGNN requires batch size 1)')

            date_start = datetime.datetime.fromisoformat(args.date_start)
            date_end = datetime.datetime.fromisoformat(args.date_end)
            training_sequence_length = args.context_window + args.prediction_window
            print(f'Training sequence length {training_sequence_length} = context_window {args.context_window} + prediction_window {args.prediction_window})')

            # Preparing data paths and constructors
            dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)
            dataset_celestrak_file_name = os.path.join(args.data_dir, args.celestrak_file_name)
            dataset_omniweb_dir = os.path.join(args.data_dir, args.omniweb_dir)
            dataset_qd_dir = os.path.join(args.data_dir, args.quasidipole_dir)
            dataset_set_file_name = os.path.join(args.data_dir, args.set_file_name)
            

            datasets_jpld_valid = []
            date_exclusions = []
            aux_datasets_valid_dict = {}

            print('Processing excluded dates')
            if args.valid_event_id:
                for event_id in args.valid_event_id:
                    print('Excluding event ID: {}'.format(event_id))

                    if event_id not in EventCatalog:
                        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                    # EventCatalog[event_id] is a dict with keys:
                    # 'date_start': date_start,
                    # 'date_end': date_end,
                    # 'duration': duration,
                    # 'max_kp': max_kp,
                    # 'time_steps': time_steps
                    event = EventCatalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(event['date_start']) - datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                    exclusion_end = datetime.datetime.fromisoformat(event['date_end'])
                    date_exclusions.append((exclusion_start, exclusion_end))
                    print('Exclusion start: {}, end: {}'.format(exclusion_start, exclusion_end))

                    datasets_jpld_valid.append(JPLD(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))

                    # datasets_jpld_valid.append(JPLD(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))
                    for name in args.aux_datasets:
                        if aux_datasets_valid_dict.get(name) is None:
                            aux_datasets_valid_dict[name] = []
                        aux_datasets_valid_dict[name].append(dataset_constructors[name](date_start_=exclusion_start, date_end_=exclusion_end, date_exclusions_ = None))

                aux_datasets_valid = []
                dataset_jpld_valid = Union(datasets=datasets_jpld_valid)
                for name, dataset_list in aux_datasets_valid_dict.items():
                    aux_datasets_valid.append(Union(datasets=dataset_list)) # NOTE: the union datasets no longer have the same start dates.
                    print("\nStart and end dates: ", aux_datasets_valid[-1].date_start, aux_datasets_valid[-1].date_end)

            # Set up datasets for VAE
            # if args.model_type == 'VAE1':
            #     dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
            #     dataset_train = dataset_jpld_train
            #     dataset_valid = dataset_jpld_valid
            #     aux_datasets_train = [dataset_constructors[name](date_start_=date_start, date_end_=date_end, date_exclusions_=date_exclusions) for name in args.aux_datasets]

            # Set up datasets for IonCastConvLSTM
            if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                dataset_celestrak_train = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end)
                dataset_celestrak_valid = CelesTrak(dataset_celestrak_file_name, date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end)
                dataset_omniweb_train = OMNIWeb(dataset_omniweb_dir, date_start=date_start, date_end=date_end, column=args.omniweb_columns)
                dataset_omniweb_valid = OMNIWeb(dataset_omniweb_dir, date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end, column=args.omniweb_columns)
                dataset_set_train = SET(dataset_set_file_name, date_start=date_start, date_end=date_end)
                dataset_set_valid = SET(dataset_set_file_name, date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end)
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train, dataset_celestrak_train, dataset_omniweb_train, dataset_set_train], sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid, dataset_celestrak_valid, dataset_omniweb_valid, dataset_set_valid], sequence_length=training_sequence_length)

            # Set up datasets for IonCastGNN
            elif args.model_type == 'IonCastGNN':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)

                if 'sunmoon' in args.aux_datasets and args.sun_moon_extra_time_steps > 0:
                    raise ValueError(f'SunMoonGeometry dataset argument sun_moon_extra_time_steps={args.sun_moon_extra_time_steps} is not compatible with IonCastGNN model. Set sun_moon_extra_time_steps=0 for IonCastGNN.')

                aux_datasets_train = [dataset_constructors[name](date_start_=date_start, date_end_=date_end, date_exclusions_=date_exclusions) for name in args.aux_datasets]

                print('Training sequence: ')
                dataset_train = Sequences([dataset_jpld_train] + aux_datasets_train, delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                print('Validation sequence: ')
                dataset_valid = Sequences([dataset_jpld_valid] + aux_datasets_valid, delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)

            else:
                raise ValueError('Unknown model type: {}'.format(args.model_type))

            if args.cache_datasets:
                print('Caching datasets in memory')
                dataset_train = CachedDataset(dataset_train)
                dataset_valid = CachedDataset(dataset_valid)

            print('\nTrain size: {:,}'.format(len(dataset_train)))
            print('Valid size: {:,}'.format(len(dataset_valid)))

            # Set up the DataLoaders
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

            if args.model_type == 'IonCastGNN':
                # Calculate n_features, C, and forcing_channels given a batch of data
                seq_dataset_batch = next(iter(train_loader))
                n_feats, C, forcing_channels = calc_shapes_for_stack_features(seq_dataset_batch, ["jpld"] + args.aux_datasets, args.context_window, batched=True)

            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = glob.glob(f'{args.target_dir}/epoch-*-model.pth')
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

            else: # Otherwise, create a new model
                print('Creating new model')
                # if args.model_type == 'VAE1':
                #     model = VAE1(z_dim=512, sigma_vae=False)

                if args.model_type == 'IonCastConvLSTM':
                    total_channels = 58  # JPLD + Sun and Moon geometry + CelesTrak + OMNIWeb + SET
                    model = IonCastConvLSTM(input_channels=total_channels, output_channels=total_channels, context_window=args.context_window, prediction_window=args.prediction_window, dropout=args.dropout)
                
                elif args.model_type == 'IonCastLSTM':
                    total_channels = 58  # JPLD + Sun and Moon geometry + CelesTrak + OMNIWeb + SET
                    model = IonCastLSTM(input_channels=total_channels, output_channels=total_channels, context_window=args.context_window, dropout=args.dropout)

                elif args.model_type == 'IonCastGNN':
                    # Note: there are many more features that can be included in IonCastGNN; see iio
                    model = IonCastGNN(
                        mesh_level = args.mesh_level,
                        input_res = (180, 360),
                        input_dim_grid_nodes = n_feats, # IMPORTANT! Based on how many features are stacked in the input.
                        output_dim_grid_nodes = C, 
                        input_dim_mesh_nodes = 3, # GraphCast used 3: cos(lat), sin(lon), cos(lon)
                        input_dim_edges = 4, # GraphCast used 4: length(edge), vector diff b/w 3D positions of sender and receiver nodes in coordinate system of the reciever
                        processor_type = args.processor_type, # Options: "MessagePassing" or "GraphTransformer", i.e. GraphCast vs. GenCast
                        khop_neighbors = 32,
                        num_attention_heads = 4,
                        processor_layers = args.ioncast_processor_layers,
                        hidden_layers = args.ioncast_hidden_layers,
                        hidden_dim = args.ioncast_hidden_dim,
                        aggregation = "sum",
                        activation_fn = "silu",
                        norm_type = "LayerNorm",
                        context_window=args.context_window,
                        forcing_channels=forcing_channels, # Forcing channels to use in the model
                        device=device
                    )

                else:
                    raise ValueError('Unknown model type: {}'.format(args.model_type))

                # Set up optimizer and initialize loss
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
            
            # Training loop
            for epoch in range(epoch_start, args.epochs):
                print('\n*** Epoch {:,}/{:,} started'.format(epoch+1, args.epochs))
                print('*** Training')

                # Training
                model.train()
                with tqdm(total=len(train_loader)) as pbar:
                    for i, batch in enumerate(train_loader):
                        optimizer.zero_grad()

                        # if args.model_type == 'VAE1':
                        #     jpld, _ = batch
                        #     jpld = jpld.to(device)
                        #     loss = model.loss(jpld)

                        if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                            jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch

                            # Send to device
                            jpld_seq = jpld_seq.to(device)
                            sunmoon_seq = sunmoon_seq.to(device)
                            celestrak_seq = celestrak_seq.to(device)
                            omniweb_seq = omniweb_seq.to(device)
                            set_seq = set_seq.to(device)

                            # Expand
                            celestrak_seq = celestrak_seq.view(celestrak_seq.shape + (1, 1)).expand(-1, -1, 2, 180, 360)
                            omniweb_seq = omniweb_seq.view(omniweb_seq.shape + (1, 1)).expand(-1, -1, 10, 180, 360)
                            set_seq = set_seq.view(set_seq.shape + (1, 1)).expand(-1, -1, 9, 180, 360)

                            combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2) # Combine along the channel dimension         
                            loss, rmse, jpld_rmse = model.loss(combined_seq, jpld_weight=args.jpld_weight)

                        elif args.model_type == "IonCastGNN":
                            # Stack features will output shape (B, T, C, H, W)                          
                            batch_notimestamps = batch[:-1] # Remove timestamp list from batch  
                            grid_nodes = stack_features(
                                batch_notimestamps, 
                                image_size=(180, 360),
                                batched=True
                            ) 
                            
                            grid_nodes = grid_nodes.to(device)
                            grid_nodes = grid_nodes.float() # Ensure the grid nodes are in float32                        
 
                            loss, rmse, jpld_rmse = model.loss(
                                grid_nodes, 
                                prediction_window=args.prediction_window,
                                jpld_weight=args.jpld_weight,
                                train_on_predicted_forcings=args.train_on_predicted_forcings, 
                            )

                        else:
                            raise ValueError('Unknown model type: {}'.format(args.model_type))
                        
                        # Backpropagation
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        iteration += 1

                        # Append training loss
                        loss = loss.detach().item()
                        rmse = rmse.detach().item()
                        jpld_rmse = jpld_rmse.detach().item()

                        train_losses.append((iteration, loss))
                        train_rmse_losses.append((iteration, rmse))
                        train_jpld_rmse_losses.append((iteration, jpld_rmse))
                        pbar.set_description(f'Epoch {epoch + 1}/{args.epochs}, MSE: {loss:.4f}, RMSE: {rmse:.4f}, JPLD RMSE: {jpld_rmse:.4f}')
                        pbar.update(1)

                # Validation loop
                if (epoch+1) % args.valid_every_nth_epoch == 0:
                    print('*** Validation')
                    model.eval()
                    valid_loss = 0.0
                    valid_rmse_loss = 0.0
                    valid_jpld_rmse_loss = 0.0
                    with torch.no_grad():
                        for batch in valid_loader:
                            # if args.model_type == 'VAE1':
                            #     jpld, _ = batch
                            #     jpld = jpld.to(device)
                            #     loss = model.loss(jpld)
                            if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                                jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq, _ = batch
                                jpld_seq = jpld_seq.to(device)
                                sunmoon_seq = sunmoon_seq.to(device)
                                celestrak_seq = celestrak_seq.to(device)
                                celestrak_seq = celestrak_seq.view(celestrak_seq.shape + (1, 1)).expand(-1, -1, 2, 180, 360)
                                omniweb_seq = omniweb_seq.to(device)
                                omniweb_seq = omniweb_seq.view(omniweb_seq.shape + (1, 1)).expand(-1, -1, 10, 180, 360)
                                set_seq = set_seq.to(device)
                                set_seq = set_seq.view(set_seq.shape + (1, 1)).expand(-1, -1, 9, 180, 360)

                                combined_seq = torch.cat((jpld_seq, sunmoon_seq, celestrak_seq, omniweb_seq, set_seq), dim=2)  # Combine along the channel dimension
                                loss, rmse, jpld_rmse = model.loss(combined_seq, jpld_weight=args.jpld_weight)

                            elif args.model_type == "IonCastGNN":
                                # Stack features will output shape (B, T, C, H, W)
                                batch_notimestamps = batch[:-1] # Remove timestamp list from batch
                                grid_nodes = stack_features(
                                    batch_notimestamps, 
                                    image_size=(180, 360),
                                    batched=True
                                )
                                
                                grid_nodes = grid_nodes.to(device)
                                grid_nodes = grid_nodes.float() # Ensure the grid nodes are in float32     

                                loss, rmse, jpld_rmse = model.loss(
                                    grid_nodes, 
                                    prediction_window=args.prediction_window,
                                    jpld_weight=args.jpld_weight,
                                    train_on_predicted_forcings=args.train_on_predicted_forcings 
                                )
                                
                            else:
                                raise ValueError('Unknown model type: {}'.format(args.model_type))
                            
                            # Increase validation loss
                            valid_loss += loss.item()
                            valid_rmse_loss += rmse.item()
                            valid_jpld_rmse_loss += jpld_rmse.item()
                        
                    # Append validation loss
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
                    plot_rmse_file = os.path.join(args.target_dir, f'{file_name_prefix}metric-rmse.pdf')
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

                        if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM' or args.model_type == 'IonCastGNN':
                            # Run forecast for validation events
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

                            # Run forecast for seen validation events
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

                                    # Check if the event is in the training dataset range
                                    if date_start < dataset_train.date_start or date_end > dataset_train.date_end:
                                        print(f'Event {event_id} is not in the training dataset range ({dataset_train.date_start} - {dataset_train.date_end}), got instead ({date_start}) - ({date_end}). skipping.')
                                        continue

                                    file_name = os.path.join(args.target_dir, f'{file_name_prefix}valid-event-seen-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                    title = f'Event: {event_id}, Kp={max_kp}'
                                    run_forecast(model, dataset_train, date_start, date_end, date_forecast_start, title, file_name, args)

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
                            if file.startswith(file_name_prefix) and (file.endswith('.pdf') or file.endswith('.mp4') or file.endswith('.pth')):
                                shutil.copyfile(os.path.join(args.target_dir, file), os.path.join(best_model_dir, file))

        elif args.mode == 'test':

            print('*** Testing mode\n')
            model, _, _, _, _, _, _, _, _, _, _, _ = load_model(args.model_file, device)
            model.eval()
            model = model.to(device) #.float()

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

                # Set up dataset paths
                dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)
                dataset_celestrak_file_name = os.path.join(args.data_dir, args.celestrak_file_name)
                dataset_omniweb_dir = os.path.join(args.data_dir, args.omniweb_dir)
                dataset_set_file_name = os.path.join(args.data_dir, args.set_file_name)
                training_sequence_length = args.context_window + args.prediction_window

                print('Running tests:')
                for i, (date_start, date_end, date_forecast_start, title, file_name) in enumerate(tests_to_run):
                    print(f'\n\n* Testing event {i+1}/{len(tests_to_run)}: {title}')
                    # Create dataset for each test individually with date filtering
                    if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
                        dataset_jpld = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end)
                        dataset_sunmoon = SunMoonGeometry(date_start=date_start, date_end=date_end, extra_time_steps=args.sun_moon_extra_time_steps)
                        dataset_celestrak = CelesTrak(dataset_celestrak_file_name, date_start=date_start, date_end=date_end)
                        dataset_omniweb = OMNIWeb(dataset_omniweb_dir, date_start=date_start, date_end=date_end, column=args.omniweb_columns)
                        dataset_set = SET(dataset_set_file_name, date_start=date_start, date_end=date_end)

                        print('Testing sequence: ')
                        dataset = Sequences(datasets=[dataset_jpld, dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)

                    # Set up datasets for IonCastGNN same as training, but with date filtering
                    elif args.model_type == 'IonCastGNN':
                        dataset_jpld = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end)
                        if 'sunmoon' in args.aux_datasets and args.sun_moon_extra_time_steps > 0:
                            raise ValueError(f'SunMoonGeometry dataset argument sun_moon_extra_time_steps={args.sun_moon_extra_time_steps} is not compatible with IonCastGNN model. Set sun_moon_extra_time_steps=0 for IonCastGNN.')
                        aux_datasets = [dataset_constructors[name](date_start_=date_start, date_end_=date_end) for name in args.aux_datasets]

                        print('Testing sequence: ')
                        dataset = Sequences([dataset_jpld] + aux_datasets, delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                    
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

# GraphCast example
# Train
# python run_ioncast.py --data_dir /home/jupyter/data --aux_dataset sunmoon celestrak omni set --mode train --target_dir /home/jupyter/linnea_results/ioncastgnn-train-july-2015-2016 --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 1000 --learning_rate 3e-3 --weight_decay 0.0 --context_window 5 --prediction_window 2 --num_evals 1 --jpld_weight 2.0 --date_start 2015-07-01T00:00:00 --date_end 2016-07-01T00:00:00 --mesh_level 5 --device cuda:0 --valid_event_seen_id G2H3-201509110600 --valid_event_id G1H3-201610261500 --valid_every_nth_epoch 1 --save_all_models

# Test on easy events
# python run_ioncast.py --data_dir /home/jupyter/data --aux_dataset sunmoon celestrak omni set --mode test --target_dir /home/jupyter/linnea_results/ioncastgnn-train-july-2015-2016 --model_file /home/jupyter/linnea_results/ioncastgnn-train-july-2015-2016/epoch-01-model.pth --num_workers 12 --batch_size 1 --model_type IonCastGNN --context_window 5 --prediction_window 2 --device cuda:1 --test_event_id G0H3-201804202100 G0H3-201808272100 G0H3-201905110300 G0H3-202311220900 G0H3-201610140300 G0H3-201506251500 G0H3-201509100000 G0H3-202305100600 G0H3-201604080000 G0H3-202104162100

# Test on hard events
# python run_ioncast.py --data_dir /home/jupyter/data --aux_dataset sunmoon celestrak omni set --mode test --target_dir /home/jupyter/linnea_results/ioncastgnn-train-july-2015-2016 --model_file /home/jupyter/linnea_results/ioncastgnn-train-july-2015-2016/epoch-01-model.pth --num_workers 12 --batch_size 1 --model_type IonCastGNN --context_window 5 --prediction_window 2 --device cuda:1 --test_event_id G2H3-201503170300 G1H3-201510070300 G2H9-202405101500 G2H9-201709072100 G1H3-202302261800

# For Halil! It is set up to run on cuda:1 (the other is running on cuda:0), save to your directory, and also validate on every event from Gunes' list (aka exclude them and plot each time)
# python run_ioncast.py --data_dir /home/jupyter/data --aux_dataset sunmoon celestrak omni set --mode train --target_dir /home/jupyter/halil_debug/ioncastgnn-train-july-2015-2016-dipole --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 1000 --learning_rate 3e-3 --weight_decay 0.0 --context_window 5 --prediction_window 2 --num_evals 1 --jpld_weight 2.0 --date_start 2015-07-01T00:00:00 --date_end 2016-07-01T00:00:00 --mesh_level 5 --device cuda:1 --valid_every_nth_epoch 1 --save_all_models  --valid_event_seen_id G2H3-201509110600 --valid_event_id G1H3-201610261500 G0H3-201804202100 G0H3-201808272100 G0H3-201905110300 G0H3-202311220900 G0H3-201610140300 G0H3-201506251500 G0H3-201509100000 G0H3-202305100600 G0H3-201604080000 G0H3-202104162100 G2H3-201503170300 G1H3-201510070300 G2H9-202405101500 G2H9-201709072100 G1H3-202302261800
