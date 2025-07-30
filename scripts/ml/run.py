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
import imageio

# from util import Tee
# from util import set_random_seed
# from models import VAE1, IonCastConvLSTM
# from datasets import JPLD, Sequences, UnionDataset
# from events import EventCatalog
# from plot_functions import save_gim_video_comparison, save_gim_video, save_gim_plot, plot_global_ionosphere_map
from ioncast import * # TODO: check this as replacement of import commented above works

matplotlib.use('Agg')

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
    sequence_prediction_window = sequence_length - (sequence_forecast_start_index) # TODO: should this be sequence_length - (sequence_forecast_start_index + 1)

    sequence_data = dataset.get_sequence_data(sequence)
    jpld_original = sequence_data[0]  # Original data
    device = next(model.parameters()).device
    jpld_original = jpld_original.to(device)
    jpld_forecast_context = jpld_original[:sequence_forecast_start_index]  # Context data for forecast
    jpld_forecast = model.predict(jpld_forecast_context.unsqueeze(0), prediction_window=sequence_prediction_window).squeeze(0)

    # print(jpld_original.shape)
    # print(jpld_forecast_context.shape)
    # print(jpld_forecast.shape)

    jpld_forecast_with_context = torch.cat((jpld_forecast_context, jpld_forecast), dim=0)

    jpld_original_unnormalized = JPLD.unnormalize(jpld_original)
    jpld_forecast_with_context_unnormalized = JPLD.unnormalize(jpld_forecast_with_context)
    jpld_forecast_with_context_unnormalized = jpld_forecast_with_context_unnormalized.clamp(0, 100)

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
        model_context_window = checkpoint['model_context_window']
        model_prediction_window = checkpoint['model_prediction_window']
        model = IonCastConvLSTM(context_window=model_context_window, prediction_window=model_prediction_window)
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
    parser.add_argument('--context_window', type=int, default=4, help='Context window size for the model')
    parser.add_argument('--prediction_window', type=int, default=4, help='Evaluation window size for the model')
    # parser.add_argument('--test_event_id', nargs='+', default=['G2H9-202311050900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--test_event_id', nargs='+', default=['G2H9-202405101500', 'G2H9-202406280900'], help='Test event IDs to use for evaluation')
    parser.add_argument('--test_event_seen_id', nargs='+', default=['G1H9-202404190600'], help='Test event IDs that the model has seen during training')

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

            training_sequence_length = args.context_window + args.prediction_window

            dataset_jpld_dir = os.path.join(args.data_dir, args.jpld_dir)

            print('Processing excluded dates')

            datasets_jpld_valid = []

            date_exclusions = []
            if args.test_event_id:
                for event_id in args.test_event_id:
                    print('Excluding event ID: {}'.format(event_id))
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                    _, _, exclusion_start, exclusion_end, _, _ = EventCatalog[event_id]
                    exclusion_start = datetime.datetime.fromisoformat(exclusion_start)
                    exclusion_end = datetime.datetime.fromisoformat(exclusion_end)
                    date_exclusions.append((exclusion_start, exclusion_end))

                    datasets_jpld_valid.append(JPLD(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))

            dataset_jpld_valid = UnionDataset(datasets=datasets_jpld_valid)


            if args.model_type == 'VAE1':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = dataset_jpld_train
                dataset_valid = dataset_jpld_valid
            elif args.model_type == 'GraphCast_reconstruct':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = dataset_jpld_train
                dataset_valid = dataset_jpld_valid
            elif args.model_type == 'IonCastConvLSTM':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = Sequences(datasets=[dataset_jpld_train], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
            elif args.model_type == 'GraphCast_forecast':
                dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
                dataset_train = Sequences(datasets=[dataset_jpld_train], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
                dataset_valid = Sequences(datasets=[dataset_jpld_valid], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)
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
                    model = IonCastConvLSTM(input_channels=1, output_channels=1, context_window=args.context_window, prediction_window=args.prediction_window)
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
                            jpld_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            
                            loss = model.loss(jpld_seq, context_window=args.context_window)
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
                            jpld_seq, _ = batch
                            jpld_seq = jpld_seq.to(device)
                            loss = model.loss(jpld_seq, context_window=args.context_window)
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
                        jpld_sample_unnormalized = jpld_sample_unnormalized.clamp(0, 100)
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
                        # jpld_seq, dates_seq = next(iter(valid_loader))
                        # jpld_seq = jpld_seq[:num_evals]
                        # dates_seq = [dates_seq[t][:num_evals] for t in range(args.context_window + args.prediction_window)]
                        # jpld_seq = jpld_seq.to(device)

                        # jpld_contexts = jpld_seq[:, :args.context_window, :, :]
                        # jpld_forecasts_originals = jpld_seq[:, args.context_window:, :, :]

                        # # Forecasts
                        # jpld_forecasts = model.predict(jpld_contexts, prediction_window=args.prediction_window)

                        # jpld_contexts_unnormalized = JPLD.unnormalize(jpld_contexts)
                        # jpld_forecasts_unnormalized = JPLD.unnormalize(jpld_forecasts)
                        # jpld_forecasts_originals_unnormalized = JPLD.unnormalize(jpld_forecasts_originals)

                        # # save forecasts

                        # # # jpld_seq_dates is a nested list of dates with shape (context_window + prediction_window, batch_size)
                        # # for b in range(args.batch_size):
                        # #     print(f'Batch {b+1}/{args.batch_size} dates:')
                        # #     for t in range(args.context_window + args.prediction_window):
                        # #         print(dates_seq[t][b])
                        # for i in range(num_evals):
                        #     dates = [dates_seq[t][i] for t in range(args.context_window + args.prediction_window)]
                        #     dates_context = [datetime.datetime.fromisoformat(d).strftime('%Y-%m-%d %H:%M:%S') for d in dates[:args.context_window]]
                        #     dates_forecast = [datetime.datetime.fromisoformat(d).strftime('%Y-%m-%d %H:%M:%S') for d in dates[args.context_window:args.context_window + args.prediction_window]]
                        #     dates_forecast_ahead = ['{} mins'.format((j + 1) * 15) for j in range(args.prediction_window)]
                        #     # save videos of the forecasts
                        #     forecast_video_file = os.path.join(args.target_dir, f'{file_name_prefix}forecast-{i+1:02d}.mp4')
                        #     save_gim_video(
                        #         jpld_forecasts_unnormalized.cpu().numpy()[i].reshape(args.prediction_window, 180, 360),
                        #         forecast_video_file,
                        #         vmin=0, vmax=100,
                        #         titles=[f'JPLD GIM TEC Forecast: {d} ({mins_ahead})' for d, mins_ahead in zip(dates_forecast, dates_forecast_ahead)]
                        #     )

                        #     # save comparison video (original vs forecast)
                        #     comparison_video_file = os.path.join(args.target_dir, f'{file_name_prefix}forecast-comparison-{i+1:02d}.mp4')
                        #     save_gim_video_comparison(
                        #         jpld_forecasts_originals_unnormalized.cpu().numpy()[i].reshape(args.prediction_window, 180, 360),  # top (original)
                        #         jpld_forecasts_unnormalized.cpu().numpy()[i].reshape(args.prediction_window, 180, 360),  # bottom (forecast)
                        #         comparison_video_file,
                        #         vmin=0, vmax=100,
                        #         titles_top=[f'JPLD GIM TEC Original: {d}' for d in dates_forecast],
                        #         titles_bottom=[f'JPLD GIM TEC Forecast: {d} ({mins_ahead})' for d, mins_ahead in zip(dates_forecast, dates_forecast_ahead)]
                        #     )

                        #     if epoch == 0:
                        #         # save videos of the forecasts originals
                        #         forecast_original_video_file = os.path.join(args.target_dir, f'{file_name_prefix}forecast-original-{i+1:02d}.mp4')
                        #         save_gim_video(
                        #             jpld_forecasts_originals_unnormalized.cpu().numpy()[i].reshape(args.prediction_window, 180, 360),
                        #             forecast_original_video_file,
                        #             vmin=0, vmax=100,
                        #             titles=[f'JPLD GIM TEC: {d}' for d in dates_forecast]
                        #         )

                        #         # save videos of the contexts
                        #         context_video_file = os.path.join(args.target_dir, f'{file_name_prefix}context-{i+1:02d}.mp4')
                        #         save_gim_video(
                        #             jpld_contexts_unnormalized.cpu().numpy()[i].reshape(args.context_window, 180, 360),
                        #             context_video_file,
                        #             vmin=0, vmax=100,
                        #             titles=[f'JPLD GIM TEC: {d}' for d in dates_context]
                        #         )

                        if args.test_event_id:
                            for event_id in args.test_event_id:
                                if event_id not in EventCatalog:
                                    raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                                event = EventCatalog[event_id]
                                _, _, date_start, date_end, _, max_kp = event
                                print('* Testing event ID: {}'.format(event_id))
                                date_start = datetime.datetime.fromisoformat(date_start)
                                date_end = datetime.datetime.fromisoformat(date_end)
                                date_forecast_start = date_start + datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                                file_name = os.path.join(args.target_dir, f'{file_name_prefix}test-event-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                title = f'Event: {event_id}, Kp={max_kp}'
                                run_forecast(model, dataset_valid, date_start, date_end, date_forecast_start, title, file_name, args)

                        if args.test_event_seen_id:
                            for event_id in args.test_event_seen_id:
                                if event_id not in EventCatalog:
                                    raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
                                event = EventCatalog[event_id]
                                _, _, date_start, date_end, _, max_kp = event
                                print('* Testing seen event ID: {}'.format(event_id))
                                date_start = datetime.datetime.fromisoformat(date_start)
                                date_end = datetime.datetime.fromisoformat(date_end)
                                date_forecast_start = date_start + datetime.timedelta(minutes=model.context_window * args.delta_minutes)
                                file_name = os.path.join(args.target_dir, f'{file_name_prefix}test-event-seen-{event_id}-kp{max_kp}-{date_start.strftime("%Y%m%d%H%M")}-{date_end.strftime("%Y%m%d%H%M")}.mp4')
                                title = f'Event: {event_id}, Kp={max_kp}'
                                run_forecast(model, dataset_train, date_start, date_end, date_forecast_start, title, file_name, args)


        elif args.mode == 'test':
            raise NotImplementedError("Testing mode is not implemented yet.")

        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()


# Example
# python run.py --data_dir /disk2-ssd-8tb/data/2025-hl-ionosphere --mode train --target_dir ./train-1 --num_workers 4 --batch_size 4 --model_type IonCastConvLSTM --epochs 2 --learning_rate 1e-3 --weight_decay 0.0 --context_window 4 --prediction_window 4 --num_evals 4 --date_start 2023-07-01T00:00:00 --date_end 2023-08-01T00:00:00
