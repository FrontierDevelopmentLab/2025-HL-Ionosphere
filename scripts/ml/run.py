from ioncast import * 
# from model_convlstm import IonCastConvLSTM
# from model_graphcast import IonCastGNN
# from graphcast_utils import stack_features
# from dataset_jpld import JPLD as JPLDGIMDataset # NOTE: hacky solution for now, need to rename either the class or all calls to JPLDGIMDataset in this file
# from dataset_sunmoon import SunMoonGeometry
# from dataset_omni import OMNIDataset
# from dataset_celestrak import CelestrakDataset
# from dataset_solar_indices import SolarIndexDataset
# from dataset_union import UnionDataset, Union
# from dataset_sequences import Sequences
# from event_catalog import EventCatalog



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
import imageio
import glob

matplotlib.use('Agg')

def run_forecast(model, dataset, date_start, date_end, date_forecast_start, title, file_name, args): 
    if not isinstance(model, (IonCastConvLSTM, IonCastGNN)):
        raise ValueError('Model must be an instance of IonCastConvLSTM or IonCastGNN')
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
    print('Evaluation from {} to {}'.format(date_start, date_end))
    print('Forecast start date: {}'.format(date_forecast_start))
    sequence_start = date_start
    sequence_end = date_end
    sequence_length = int((sequence_end - sequence_start).total_seconds() / 60 / args.delta_minutes)
    print('Sequence length: {}'.format(sequence_length))
    sequence = [sequence_start + datetime.timedelta(minutes=args.delta_minutes * i) for i in range(sequence_length)]
    # find the index of the date_forecast_start in the list sequence
    if date_forecast_start not in sequence:
        raise ValueError('date_forecast_start must be in the sequence')
    sequence_forecast_start_index = sequence.index(date_forecast_start)
    sequence_prediction_window = sequence_length - (sequence_forecast_start_index)

    # Get the sequence
    sequence_data = dataset.get_sequence_data(sequence)
    
    # Get JPLD original data
    device = next(model.parameters()).device
    jpld_original = sequence_data[0]      # JPLD GIM TEC data
    jpld_original = jpld_original.to(device) # sequence_length, channels, 180, 360

    # If IonCastConvLSTM, load data and concatenate along channel dimension
    if isinstance(model, IonCastConvLSTM):
        # Separate datasets from the sequence and stack along the channel dimension
        sunmoon_original = sequence_data[1]  # Sun and Moon geometry data
        sunmoon_original = sunmoon_original.to(device) # sequence_length, channels, 180, 360
        combined_original = torch.cat((jpld_original, sunmoon_original), dim=1)  # Combine along the channel dimension

        # Predict by passing in the context data and prediction window
        combined_forecast_context = combined_original[:sequence_forecast_start_index]  # Context data for forecast
        print(combined_forecast_context.unsqueeze(0))
        combined_forecast = model.predict(combined_forecast_context.unsqueeze(0), prediction_window=sequence_prediction_window).squeeze(0)

        # Get JPLD prediction and context channels, and stack the forecast with the context
        jpld_forecast_context = combined_forecast_context[:, 0]  # Extract JPLD channels from the context
        jpld_forecast = combined_forecast[:, 0]  # Extract JPLD channels from the forecast
        jpld_forecast_with_context = torch.cat((jpld_forecast_context, jpld_forecast), dim=0)
    
    # If IonCastGNN, pass sequence data to stack_features
    if isinstance(model, IonCastGNN):
        # Calculate n_features, C, forcing_channels, and n_img_datasets given sequence_data
        _, _, _, n_img_datasets = calc_shapes_for_stack_features(sequence_data, args.aux_datasets, args.context_window, batched=False)


        # Stack features will output shape (B, T, C, H, W)
        grid_nodes_original = stack_features(
            sequence_data, 
            n_img_datasets=n_img_datasets # Note: this is hardcoded to 1
        )

        grid_nodes_original = grid_nodes_original.to(device)
        grid_nodes_original = grid_nodes_original.float() # Ensure the grid nodes are in float32 

        # Output context & forecast for all time steps, shape (B, T, C, H, W)
        combined_forecast = model.predict(
            grid_nodes_original, # .predict will mask out values not in [:, :sequence_forecast_start_index, :, :, :]
            context_window=sequence_forecast_start_index, # Context window is the number of time steps before the forecast start
            train=False
        )

        # Extract JPLD channels from the forecast and removing the batch dimension (batchsize = 1 as we previously 
        # added the batch dimension as a processing step before stacking the features.
        jpld_forecast_with_context = combined_forecast[0, :, 0, :, :] 
        
    # Unnormalize the JPLD data (reference and context/forecast)
    jpld_original_unnormalized = JPLDGIMDataset.unnormalize(jpld_original)
    jpld_forecast_with_context_unnormalized = JPLDGIMDataset.unnormalize(jpld_forecast_with_context)
    jpld_forecast_with_context_unnormalized = jpld_forecast_with_context_unnormalized.clamp(0, 100) # Clamp to valid range for comparison

    forecast_mins_ahead = ['{} mins'.format((j + 1) * 15) for j in range(sequence_prediction_window)]
    titles_original = [f'JPLD GIM TEC Original: {d} - {title}' for d in sequence]
    titles_forecast = []
    for i in range(sequence_length):
        if i < sequence_forecast_start_index:
            titles_forecast.append(f'JPLD GIM TEC Context : {sequence[i]} - {title}')
        else:
            titles_forecast.append(f'JPLD GIM TEC Forecast: {sequence[i]} ({forecast_mins_ahead[i - sequence_forecast_start_index]}) - {title}')

    save_gim_video_comparison(
        gim_sequence_top=jpld_original_unnormalized.cpu().numpy().reshape(-1, 180, 360),
        gim_sequence_bottom=jpld_forecast_with_context_unnormalized.cpu().numpy().reshape(-1, 180, 360),
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
    elif isinstance(model, IonCastGNN):
        checkpoint = {
            'model': 'IonCastGNN',
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
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

    elif checkpoint["model"] == "IonCastGNN":
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
    return model, optimizer, epoch, iteration, train_losses, valid_losses

def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--jpld_dir', type=str, default='jpld/webdataset', help='JPLD GIM dataset directory')
    parser.add_argument('--omni_dir', type=str, default='omniweb/cleaned/', help='OMNIWeb dataset directory')
    parser.add_argument('--celestrak_file', type=str, default='celestrak/kp_ap_processed_timeseries.csv', help='Celestrak dataset csv file')
    parser.add_argument('--solar_index_file', type=str, default='solar_env_tech_indices/Indices_F10_processed.csv', help='Solar indices dataset csv file')
    parser.add_argument('--aux_datasets', nargs='+', choices=["sunmoon", "omni", "celestrak", "solar_inds"], default=["sunmoon", "omni", "celestrak", "solar_inds"], help="additional datasets to include on top of TEC maps")
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
    parser.add_argument('--model_type', type=str, choices=['VAE1', 'IonCastConvLSTM', "IonCastGNN"], default='VAE1', help='Type of model to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_evals', type=int, default=4, help='Number of samples for evaluation')
    parser.add_argument('--context_window', type=int, default=4, help='Context window size for the model')
    parser.add_argument('--prediction_window', type=int, default=4, help='Evaluation window size for the model')
    # parser.add_argument('--test_event_id', nargs='+', default=['G2H9-202311050900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--test_event_id', nargs='*', default=['G2H9-202406280900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--test_event_seen_id', nargs='*', default=['G1H9-202404190600'], help='Test event IDs that the model has seen during training') 
    parser.add_argument('--model_file', type=str, help='Path to the model file to load for testing')
    parser.add_argument('--mesh_level', type=int, default=6, help='Mesh level for IonCastGNN model')
    parser.add_argument('--processor_type', type=str, choices=['MessagePassing', 'GraphTransformer'], default='MessagePassing', help='Processor type for IonCastGNN model')
    parser.add_argument('--ioncast_hidden_dim', type=int, default=512, help='Hidden dimension for IonCastGNN model')
    parser.add_argument('--ioncast_hidden_layers', type=int, default=1, help='Number of hidden layers for IonCastGNN model')
    parser.add_argument('--ioncast_processor_layers', type=int, default=6, help='Number of processor layers for IonCastGNN model')
    parser.add_argument('--sunmoon_extra_time_steps', type=int, default=0, help='Extra time steps for SunMoonGeometry dataset')
    parser.add_argument('--train_on_predicted_forcings', action='store_true', help='Train on predicted forcings for IonCastGNN model')

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    log_file = os.path.join(args.target_dir, 'log.txt')

    set_random_seed(args.seed)
    device = torch.device(args.device)
    print('Using device:', device)

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
            # gim_webdataset = os.path.join(args.data_dir, args.jpld_dir)
            omni_dir = os.path.join(args.data_dir, args.omni_dir)
            celestrak_file = os.path.join(args.data_dir, args.celestrak_file)
            solar_index_file = os.path.join(args.data_dir, args.solar_index_file)
            
            # 'jpld': lambda date_exclusion: JPLD(gim_webdataset, date_start=date_start, date_end=date_end, normalize=True, date_exclusions=date_exclusion)
            dataset_constructors = {
                'sunmoon': lambda date_start_, date_end_, date_exclusions_: SunMoonGeometry(date_start=date_start_, date_end=date_end_, normalize=True, extra_time_steps=args.sunmoon_extra_time_steps), # Note: no date_exclusions and also extra_time_steps should be 1 for IonCastGNN
                'omni': lambda date_start_, date_end_, date_exclusions_: OMNIDataset(file_dir=omni_dir, delta_minutes=15, date_start=date_start_, date_end=date_end_, normalize=True, date_exclusions=date_exclusions_),
                'celestrak': lambda date_start_, date_end_, date_exclusions_: CelestrakDataset(file_name=celestrak_file, delta_minutes=15, date_start=date_start_, date_end=date_end_, normalize=True, date_exclusions=date_exclusions_),
                'solar_inds': lambda date_start_, date_end_, date_exclusions_: SolarIndexDataset(file_name=solar_index_file, delta_minutes=15, date_start=date_start_, date_end=date_end_, normalize=True, date_exclusions=date_exclusions_),
            }


            datasets_jpld_valid = []
            date_exclusions = []
            aux_datasets_valid_dict = {}

            # Process excluded dates
            print('Processing excluded dates')
            if args.test_event_id:
                for event_id in args.test_event_id:
                    print('Excluding event ID: {}'.format(event_id))
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                    _, _, exclusion_start, exclusion_end, _, _, _ = EventCatalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(exclusion_start)
                    exclusion_end = datetime.datetime.fromisoformat(exclusion_end)
                    date_exclusions.append((exclusion_start, exclusion_end))

                    datasets_jpld_valid.append(JPLDGIMDataset(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))

                    # datasets_jpld_valid.append(JPLD(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))
                    for name in args.aux_datasets:
                        if aux_datasets_valid_dict.get(name) is None:
                            aux_datasets_valid_dict[name] = []
                        aux_datasets_valid_dict[name].append(dataset_constructors[name](date_start_=exclusion_start, date_end_=exclusion_end, date_exclusions_ = None))


                aux_datasets_valid = []
                dataset_jpld_valid = Union(datasets=datasets_jpld_valid)
                for name, dataset_list in aux_datasets_valid_dict.items():
                    aux_datasets_valid.append(UnionDataset(datasets=dataset_list)) # NOTE: the union datasets no longer have the same start dates.
                    print("\nStart and end dates: ", aux_datasets_valid[-1].date_start, aux_datasets_valid[-1].date_end)

            # Set up datasets for VAE
            if args.model_type == 'VAE1':
                dataset_jpld_train = JPLDGIMDataset(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = dataset_jpld_train
                dataset_valid = dataset_jpld_valid
                aux_datasets_train = [dataset_constructors[name](date_start_=date_start, date_end_=date_end, date_exclusions_=date_exclusions) for name in args.aux_datasets]

            # Set up datasets for IonCastConvLSTM
            elif args.model_type == 'IonCastConvLSTM':
                dataset_jpld_train = JPLDGIMDataset(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = Sequences(datasets=[dataset_jpld_train], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                dataset_sunmoon_train = SunMoonGeometry(date_start=date_start, date_end=date_end)
                dataset_sunmoon_valid = SunMoonGeometry(date_start=dataset_jpld_valid.date_start, date_end=dataset_jpld_valid.date_end)
                dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_sunmoon_train], sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_sunmoon_valid], sequence_length=training_sequence_length)

            # Set up datasets for IonCastGNN
            elif args.model_type == 'IonCastGNN':
                dataset_jpld_train = JPLDGIMDataset(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)

                if 'sunmoon' in args.aux_datasets and args.sunmoon_extra_time_steps > 0:
                    raise ValueError(f'SunMoonGeometry dataset argument sunmoon_extra_time_steps={args.sunmoon_extra_time_steps} is not compatible with IonCastGNN model. Set sunmoon_extra_time_steps=0 for IonCastGNN.')

                aux_datasets_train = [dataset_constructors[name](date_start_=date_start, date_end_=date_end, date_exclusions_=date_exclusions) for name in args.aux_datasets]

                print('Training sequence: ')
                dataset_train = Sequences([dataset_jpld_train] + aux_datasets_train, delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                print('Validation sequence: ')
                dataset_valid = Sequences([dataset_jpld_valid] + aux_datasets_valid, delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)

            else:
                raise ValueError('Unknown model type: {}'.format(args.model_type))

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
                # Calculate n_features, C, forcing_channels, and n_img_datasets given a batch of data
                seq_dataset_batch = next(iter(train_loader))
                n_feats, C, forcing_channels, n_img_datasets = calc_shapes_for_stack_features(seq_dataset_batch, args.aux_datasets, args.context_window, batched=True)

            # check if a previous training run exists in the target directory, if so, find the latest model file saved, resume training from there by loading the model instead of creating a new one
            model_files = glob.glob(f'{args.target_dir}/epoch-*-model.pth')
            if len(model_files) > 0:
                model_files.sort()
                model_file = model_files[-1]
                print('Resuming training from model file: {}'.format(model_file))
                model, optimizer, epoch, iteration, train_losses, valid_losses = load_model(model_file, device)
                epoch_start = epoch + 1
                iteration = iteration + 1
                print('Next epoch    : {:,}'.format(epoch_start+1))
                print('Next iteration: {:,}'.format(iteration+1))

            else: # Otherwise, create a new model
                print('Creating new model')
                if args.model_type == 'VAE1':
                    model = VAE1(z_dim=512, sigma_vae=False)

                elif args.model_type == 'IonCastConvLSTM':
                    model = IonCastConvLSTM(input_channels=19, output_channels=19, context_window=args.context_window, prediction_window=args.prediction_window)

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
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # Note: GraphCast used AdamW and FusedAdam
                iteration = 0
                epoch_start = 0
                train_losses = []
                valid_losses = []
                
                model = model.to(device)

                for name, param in model.named_parameters():
                    if param.dtype != torch.float32:
                        print(f"[param] {name} has dtype {param.dtype}")

                for name, buffer in model.named_buffers():
                    if buffer.dtype != torch.float32:
                        print(f"[buffer] {name} has dtype {buffer.dtype}")
                

            
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

                        if args.model_type == 'VAE1':
                            jpld, _ = batch
                            jpld = jpld.to(device)

                            loss = model.loss(jpld)

                        elif args.model_type == 'IonCastConvLSTM':
                            jpld_seq, sunmoon_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            sunmoon_seq = sunmoon_seq.to(device)

                            combined_seq = torch.cat((jpld_seq, sunmoon_seq), dim=2) # Combine along the channel dimension

                            loss = model.loss(combined_seq, context_window=args.context_window)

                        elif args.model_type == "IonCastGNN":
                            # jpld_dataset, omni_dataset, celestrak_dataset, solar_index_dataset
                            # Stack features will output shape (B, T, C, H, W)                            

                            grid_nodes = stack_features(
                                batch, 
                                n_img_datasets=n_img_datasets,
                            ) 
                            
                            grid_nodes = grid_nodes.to(device)
                            grid_nodes = grid_nodes.float() # Ensure the grid nodes are in float32                        
 
                            loss = model.loss(
                                grid_nodes, 
                                prediction_window=args.prediction_window, # Starts at 1, but eventually during training this should increase to args.prediction_window
                                train_on_predicted_forcings=args.train_on_predicted_forcings 
                            )

                        else:
                            raise ValueError('Unknown model type: {}'.format(args.model_type))
                        
                        # Backpropagation
                        loss.backward()
                        optimizer.step()
                        iteration += 1

                        # Append training loss
                        train_losses.append((iteration, float(loss)))
                        pbar.set_description(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}')
                        pbar.update(1)

                # Validation loop
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
                            jpld_seq, sunmoon_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            sunmoon_seq = sunmoon_seq.to(device)
                            combined_seq = torch.cat((jpld_seq, sunmoon_seq), dim=2)  # Combine along the channel dimension
                            loss = model.loss(combined_seq, context_window=args.context_window)

                        elif args.model_type == "IonCastGNN":
                            grid_nodes = stack_features(
                                batch, 
                                n_img_datasets=n_img_datasets, 
                            )
                            
                            grid_nodes = grid_nodes.to(device)
                            grid_nodes = grid_nodes.float() # Ensure the grid nodes are in float32     

                            loss = model.loss(
                                grid_nodes, 
                                prediction_window=1, # Starts at 1, but eventually during training this should increase to args.prediction_window
                                train_on_predicted_forcings=args.train_on_predicted_forcings 
                            )
                            
                        else:
                            raise ValueError('Unknown model type: {}'.format(args.model_type))
                        
                        # Increase validation loss
                        valid_loss += loss.item()
                    
                # Append validation loss
                valid_loss /= len(valid_loader)
                valid_losses.append((iteration, valid_loss))
                print(f'Validation Loss: {valid_loss:.4f}')

                # Save model
                file_name_prefix = f'epoch-{epoch + 1:02d}-'
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
                model.eval()
                with torch.no_grad():
                    num_evals = args.num_evals

                    if args.model_type == 'VAE1':
                        # Set random seed for reproducibility of evaluation samples across epochs
                        rng_state = torch.get_rng_state()
                        torch.manual_seed(args.seed)
                    if args.model_type == 'VAE1':
                        # Set random seed for reproducibility of evaluation samples across epochs
                        rng_state = torch.get_rng_state()
                        torch.manual_seed(args.seed)

                        # Reconstruct a batch from the validation set
                        jpld_orig, jpld_orig_dates = next(iter(valid_loader))
                        jpld_orig = jpld_orig[:num_evals]
                        jpld_orig_dates = jpld_orig_dates[:num_evals]
                        # Reconstruct a batch from the validation set
                        jpld_orig, jpld_orig_dates = next(iter(valid_loader))
                        jpld_orig = jpld_orig[:num_evals]
                        jpld_orig_dates = jpld_orig_dates[:num_evals]

                        jpld_orig = jpld_orig.to(device)
                        jpld_recon, _, _ = model.forward(jpld_orig)
                        jpld_orig_unnormalized = JPLDGIMDataset.unnormalize(jpld_orig)
                        jpld_recon_unnormalized = JPLDGIMDataset.unnormalize(jpld_recon)
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
                        # Sample a batch from the model
                        jpld_sample = model.sample(n=num_evals)
                        jpld_sample_unnormalized = JPLDGIMDataset.unnormalize(jpld_sample)
                        jpld_sample_unnormalized = jpld_sample_unnormalized.clamp(0, 100)
                        torch.set_rng_state(rng_state)
                        # Resume with the original random state

                        # Save plots
                        for i in range(num_evals):
                            date = jpld_orig_dates[i]
                            date_str = datetime.datetime.fromisoformat(date).strftime('%Y-%m-%d %H:%M:%S')
                        # Save plots
                        for i in range(num_evals):
                            date = jpld_orig_dates[i]
                            date_str = datetime.datetime.fromisoformat(date).strftime('%Y-%m-%d %H:%M:%S')

                            recon_original_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-original-{i+1:02d}.pdf')
                            save_gim_plot(jpld_orig_unnormalized[i][0].cpu().numpy(), recon_original_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str}')
                            recon_original_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-original-{i+1:02d}.pdf')
                            save_gim_plot(jpld_orig_unnormalized[i][0].cpu().numpy(), recon_original_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str}')

                            recon_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-{i+1:02d}.pdf')
                            save_gim_plot(jpld_recon_unnormalized[i][0].cpu().numpy(), recon_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str} (Reconstruction)')
                            recon_file = os.path.join(args.target_dir, f'{file_name_prefix}reconstruction-{i+1:02d}.pdf')
                            save_gim_plot(jpld_recon_unnormalized[i][0].cpu().numpy(), recon_file, vmin=0, vmax=100, title=f'JPLD GIM TEC, {date_str} (Reconstruction)')

                            sample_file = os.path.join(args.target_dir, f'{file_name_prefix}sample-{i+1:02d}.pdf')
                            save_gim_plot(jpld_sample_unnormalized[i][0].cpu().numpy(), sample_file, vmin=0, vmax=100, title='JPLD GIM TEC (Sampled from model)')
                            sample_file = os.path.join(args.target_dir, f'{file_name_prefix}sample-{i+1:02d}.pdf')
                            save_gim_plot(jpld_sample_unnormalized[i][0].cpu().numpy(), sample_file, vmin=0, vmax=100, title='JPLD GIM TEC (Sampled from model)')

                    elif args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastGNN':
                        # Run forecast for test events
                        if args.test_event_id:
                            for event_id in args.test_event_id:
                                if event_id not in EventCatalog:
                                    raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                                event = EventCatalog[event_id]
                                _, _, date_start, date_end, _, max_kp, _ = event
                                print('* Testing event ID: {}'.format(event_id))
                                date_end = datetime.datetime.fromisoformat(date_end)
                                date_forecast_start = date_start + datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                                file_name = os.path.join(args.target_dir, f'{file_name_prefix}test-event-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                title = f'Event: {event_id}, Kp={max_kp}'
                                run_forecast(model, dataset_valid, date_start, date_end, date_forecast_start, title, file_name, args)

                        # Run forecast for seen test events
                        if args.test_event_seen_id:
                            for event_id in args.test_event_seen_id:
                                if event_id not in EventCatalog:
                                    raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                                event = EventCatalog[event_id]
                                _, _, date_start, date_end, _, max_kp, _ = event
                                print('* Testing seen event ID: {}'.format(event_id))
                                date_start = datetime.datetime.fromisoformat(date_start)
                                date_end = datetime.datetime.fromisoformat(date_end)

                                # Check if the event is in the training dataset range
                                if date_start < dataset_train.date_start or date_end > dataset_train.date_end:
                                    print(f'Event {event_id} is not in the training dataset range ({dataset_train.date_start} - {dataset_train.date_end}), skipping.')
                                    continue

                                date_forecast_start = date_start + datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                                file_name = os.path.join(args.target_dir, f'{file_name_prefix}test-event-seen-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                title = f'Event: {event_id}, Kp={max_kp}'
                                run_forecast(model, dataset_train, date_start, date_end, date_forecast_start, title, file_name, args)

        elif args.mode == 'test':

            print('*** Testing mode\n')

            model, _, _, _, _, _, = load_model(args.model_file, device)
            model.eval()
            model = model.to(device).float()

            with torch.no_grad():
                tests_to_run = []
                if args.test_event_id:
                    for event_id in args.test_event_id:
                        if event_id not in EventCatalog:
                            raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                        event = EventCatalog[event_id]
                        _, _, date_start, date_end, _, max_kp, _ = event
                        print('* Testing event ID: {}'.format(event_id))
                        date_start = datetime.datetime.fromisoformat(date_start)
                        date_end = datetime.datetime.fromisoformat(date_end)
                        date_forecast_start = date_start + datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                        file_name = os.path.join(args.target_dir, f'test-event-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                        title = f'Event: {event_id}, Kp={max_kp}'
                        tests_to_run.append((date_start, date_end, date_forecast_start, title, file_name))
                else:
                    print('No test events specified, will use date_start and date_end arguments')
                    date_start = datetime.datetime.fromisoformat(args.date_start)
                    date_end = datetime.datetime.fromisoformat(args.date_end)
                    date_forecast_start = date_start + datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                    file_name = os.path.join(args.target_dir, f'test-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                    title = f'Test from {date_start.strftime("%Y-%m-%d %H:%M:%S")} to {date_end.strftime("%Y-%m-%d %H:%M:%S")}'
                    tests_to_run.append((date_start, date_end, date_forecast_start, title, file_name))

                dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)
                
                print('Running tests:')
                for date_start, date_end, date_forecast_start, title, file_name in tests_to_run:
                    # Create dataset for each test individually with date filtering
                    # Add some buffer time for context window
                    dataset_start = date_start - datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                    dataset_jpld = JPLD(dataset_jpld_dir, date_start=dataset_start, date_end=date_end)

                    # TODO: make this compatible with IonCastGNN, as it doesn't have a prediction window
                    dataset = Sequences(datasets=[dataset_jpld], delta_minutes=args.delta_minutes, 
                                    sequence_length=model.context_window + model.prediction_window)
                    
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
# With more aux datasets
# python run.py --data_dir /home/jupyter/data --aux_dataset sunmoon celestrak --mode train --target_dir /home/jupyter/linnea_results/ioncastgnn-train-sunmoon-celestrak --num_workers 4 --batch_size 1 --model_type IonCastGNN --epochs 1 --learning_rate 1e-3 --weight_decay 0.0 --context_window 2 --prediction_window 1 --num_evals 1 --date_start 2023-07-01T00:00:00 --date_end 2023-08-01T00:00:00 --mesh_level 4 --device cuda:0

# Baseline without auxiliary datasets
# python run.py --data_dir /home/jupyter/data --aux_dataset sunmoon celestrak --mode train --target_dir /home/jupyter/halil_debug/ioncastgnn-train-debug-1 --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 100 --learning_rate 1e-3 --weight_decay 0.0 --context_window 5 --prediction_window 1 --num_evals 1 --date_start 2022-07-01T00:00:00 --date_end 2022-07-01T05:00:00 --mesh_level 4 --device cuda:0 --train_on_predicted_forcings
