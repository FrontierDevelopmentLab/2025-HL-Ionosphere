import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import glob
import shutil
import random

from util import Tee
try:
    import wandb
except ImportError:
    wandb = None
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
from eval import eval_forecast_long_horizon, save_metrics, eval_forecast_fixed_lead_time

event_catalog = EventCatalog(events_csv_file_name='../data/events.csv')

matplotlib.use('Agg')


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
            'model_base_channels': model.base_channels,
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
        model_base_channels = checkpoint['model_base_channels']
        model_lstm_dim = checkpoint['model_lstm_dim']
        model_num_layers = checkpoint['model_num_layers']
        model_context_window = checkpoint['model_context_window']
        model_dropout = checkpoint['model_dropout']
        model = IonCastLSTM(input_channels=model_input_channels, output_channels=model_output_channels,
                            base_channels=model_base_channels, lstm_dim=model_lstm_dim, num_layers=model_num_layers,
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
    parser.add_argument('--eval_mode', type=str, choices=['long_horizon', 'fixed_lead_time', 'all'], default='all', help='Type of evaluation to run in test mode.')
    parser.add_argument('--lead_times', nargs='+', type=int, default=[15, 30, 45, 60], help='A list of lead times in minutes for fixed-lead-time evaluation.')
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
    
    # Weights & Biases options
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb_project', type=str, default='Ionosphere')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_notes', type=str, default=None)
    parser.add_argument('--wandb_tags', nargs='*', default=None)
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable W&B (same as --wandb_mode disabled)')

    args = parser.parse_args()

    # --- W&B setup ---
    if args.wandb_disabled:
        args.wandb_mode = 'disabled'
    wandb_config = vars(args).copy()
    
    # Initialize wandb
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
                date_start_plus_context = date_start + datetime.timedelta(minutes=args.context_window * args.delta_minutes)
                event_catalog_within_training_set = event_catalog.filter(date_start=date_start_plus_context, date_end=date_end).exclude(date_exclusions=date_exclusions)
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
                        # W&B train metrics
                        if wandb is not None and args.wandb_mode != 'disabled':
                            wandb.log({
                                'train/loss': float(loss),
                                'train/rmse': float(rmse),
                                'train/jpld_rmse': float(jpld_rmse),
                                'train/epoch': epoch + 1,
                                'train/iteration': iteration,
                                'lr': optimizer.param_groups[0]['lr'],
                            }, step=iteration)
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
                        if args.model_type == 'IonCastConvLSTM' or args.model_type == 'IonCastLSTM':
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
                            
                            if args.valid_event_id:
                                for i, event_id in enumerate(args.valid_event_id):
                                    print(f'\n--- Evaluating validation event: {event_id} ---')
                                    event_category = event_id.split('-')[0]
                                    save_video = False
                                    if event_category not in saved_video_categories:
                                        save_video = True
                                        saved_video_categories.add(event_category)

                                    # --- Long Horizon Evaluation ---
                                    if args.eval_mode in ['long_horizon', 'all']:
                                        
                                        jpld_rmse, jpld_mae, jpld_unnormalized_rmse_val, jpld_unnormalized_mae_val, jpld_unnormalized_rmse_low_lat_val, jpld_unnormalized_rmse_mid_lat_val, jpld_unnormalized_rmse_high_lat_val = eval_forecast_long_horizon(model, dataset_valid, event_catalog, event_id, file_name_prefix+'valid', save_video, args)
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
                                        eval_forecast_fixed_lead_time(model, dataset_valid, event_catalog, event_id, args.lead_times, file_name_prefix+'valid', save_video, args)

                            # Save metrics from long-horizon eval
                            if metric_event_id:
                                metrics_file_prefix = os.path.join(args.target_dir, f'{file_name_prefix}valid-long-horizon-metrics')
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

                            # --- EVALUATION ON SEEN VALIDATION EVENTS ---
                            saved_video_categories_seen = set()                            
                            if args.valid_event_seen_id:
                                for i, event_id in enumerate(args.valid_event_seen_id):
                                    event_category = event_id.split('-')[0]
                                    save_video = False
                                    if event_category not in saved_video_categories_seen:
                                        save_video = True
                                        saved_video_categories_seen.add(event_category)                                    
                                    print(f'\n--- Evaluating seen validation event: {event_id} ---')
                                    # --- Long Horizon Evaluation (Seen) ---
                                    if args.eval_mode in ['long_horizon', 'all']:
                                        # Note: We don't save metrics for 'seen' events to avoid clutter, just the video.
                                        eval_forecast_long_horizon(model, dataset_train, event_catalog, event_id, file_name_prefix+'valid-seen', save_video, args)
                                    
                                    # --- Fixed Lead Time Evaluation (Seen) ---
                                    if args.eval_mode in ['fixed_lead_time', 'all']:
                                        eval_forecast_fixed_lead_time(model, dataset_train, event_catalog, event_id, args.lead_times, file_name_prefix+'valid-seen', save_video, args)

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
                            if file.startswith(file_name_prefix) and (file.endswith('.pdf') or file.endswith('.png') or file.endswith('.mp4') or file.endswith('.pth') or file.endswith('.csv')):
                                shutil.copyfile(os.path.join(args.target_dir, file), os.path.join(best_model_dir, file))

        elif args.mode == 'test':
            print('*** Testing mode\n')

            if not args.model_file:
                raise ValueError("A --model_file must be specified for testing mode.")
            
            print(f'Loading model from {args.model_file}')
            model, optimizer, _, _, _, _, _, _, _, _, _, _ = load_model(args.model_file, device)
            model.eval()
            model = model.to(device)
            if not args.test_event_id:
                print("No --test_event_id provided. Exiting test mode.")
                return

            with torch.no_grad():
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
                    dataset_jpld = JPLD(os.path.join(args.data_dir, args.jpld_dir), date_start=buffer_start, date_end=event_end)
                    dataset_sunmoon = SunMoonGeometry(date_start=buffer_start, date_end=event_end, extra_time_steps=args.sun_moon_extra_time_steps)
                    dataset_celestrak = CelesTrak(os.path.join(args.data_dir, args.celestrak_file_name), date_start=buffer_start, date_end=event_end, return_as_image_size=(180, 360))
                    dataset_omniweb = OMNIWeb(os.path.join(args.data_dir, args.omniweb_dir), date_start=buffer_start, date_end=event_end, column=args.omniweb_columns, return_as_image_size=(180, 360))
                    dataset_set = SET(os.path.join(args.data_dir, args.set_file_name), date_start=buffer_start, date_end=event_end, return_as_image_size=(180, 360))
                    
                    dataset = Sequences(datasets=[dataset_jpld, dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set], delta_minutes=args.delta_minutes, sequence_length=1) # sequence_length doesn't matter here

                    file_name_prefix = os.path.join(args.target_dir, 'test')

                    if args.eval_mode in ['long_horizon', 'all']:
                        eval_forecast_long_horizon(model, dataset, event_catalog, event_id, file_name_prefix, True, args)

                    if args.eval_mode in ['fixed_lead_time', 'all']:
                        eval_forecast_fixed_lead_time(model, dataset, event_catalog, event_id, args.lead_times, file_name_prefix, args)

                    # Force cleanup
                    del dataset_jpld, dataset_sunmoon, dataset_celestrak, dataset_omniweb, dataset_set, dataset
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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

