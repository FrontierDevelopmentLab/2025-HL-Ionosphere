import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append("../")
import ionopy
from ionopy import MadrigalDatasetTimeSeries, weight_init, mae_loss

import pandas as pd
import torch
from omegaconf import OmegaConf
from tft_torch import tft
import tft_torch.loss as tft_loss
import torch.nn.init as init
import numpy as np
import wandb
from pyfiglet import Figlet
from termcolor import colored
from tqdm import tqdm
import argparse
import pprint
import time
from torch import optim
from torch.utils.data import RandomSampler, SequentialSampler
import random

def train():
    print('Ionopy Model Training -> Forecasting the ionosphere vTEC using Madrigal data as ground truth')
    f = Figlet(font='big')
    print(colored(f.renderText('Ionopy 0.0.1dev'), 'red'))
    f = Figlet(font='digital')
    print(colored(f.renderText("Training Forecasting Model"), 'blue'))
    #print(colored(f'Version {ionopy.__version__}\n','blue'))
    print(colored(f'Version {ionopy.__version__}\n','blue'))

    parser = argparse.ArgumentParser(description='HL-25 Ionopy Model Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='', help='Device to use for training')
    parser.add_argument('--torch_type', type=str, default='float32', help='Torch type to use for training')
    parser.add_argument('--subset_type', type=int, default=5,  choices=[5, 10, 20, 30, 40], help='Which Madrigal data to use, possible choices are: 5, 10, 20, 30, 40 million points')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default='', help='Path to the model to load. If None, a new model is created')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate for the optimizer')
    parser.add_argument('--run_name', default='', help='Run name to be stored in wandb')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for the dataloader')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--lag_days_proxies', type=float, default=144, help='Lag in days for the SET and Celestrack proxies')
    parser.add_argument('--proxies_resolution', type=int, default=1, help='Resolution in days for the SET and Celestrack proxies')
    parser.add_argument('--lag_minutes_omni', type=int, default=8640, help='Lag in minutes for the OMNIweb data (indices, magnetic field, solar wind)')
    parser.add_argument('--omni_resolution', type=int, default=60, help='Resolution in minutes for the OMNIweb data (indices, magnetic field, solar wind)')
    parser.add_argument('--lag_minutes_jpld', type=int, default=8640, help='Lag in minutes for the JPLD data')
    parser.add_argument('--jpld_resolution', type=int, default=60, help='Resolution in minutes for the JPLD data')
    # parser.add_argument('--min_date', type=str, default='2000-07-29 00:59:47', help='Min date to consider for the dataset')
    # parser.add_argument('--max_date', type=str, default='2024-05-31 23:59:32', help='Max date to consider for the dataset')
    parser.add_argument('--model_type', type=str, default='tft', choices=['tft'],help='Time series model to be used')
    parser.add_argument('--bucket_dir', type=str, default='/home/ga00693/gcs-bucket', help='Path to the directory where the ionosphere-data bucket is mounted')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for the TFT model')
    parser.add_argument('--state_size', type=int, default=64, help='State size for the TFT model or the LSTM model, depending which one is chosen as model_type')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers of the TFT or the LSTM model, depending which one is chosen as model_type')
    parser.add_argument('--attention_heads', type=int, default=4, help='Number of attention heads for the TFT')
    parser.add_argument('--wandb_inactive', action='store_true', help='Flag to activate/deactivate weights and biases')
    opt = parser.parse_args()

    timestamp_training = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if opt.wandb_inactive is False:
        wandb.init(
            project="Ionosphere",
            entity="Ionosphere",
            config=vars(opt),
            name=f"{opt.model_type}_{opt.subset_type}mln_{timestamp_training}",
        )
        print("W&B is active")
    
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=1)
    print()
    if opt.torch_type=='float32':
        torch_type=torch.float32
    elif opt.torch_type=='float64':
        torch_type=torch.float64
    else:
        raise ValueError('Invalid torch type. Only float32 and float64 are supported')
    torch.set_default_dtype(torch_type)
    if opt.device=='':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)
    print("Using: ", device)

    config={'madrigal_path': f'{opt.bucket_dir}/madrigal_data/processed/gps_data_tarr/csv_subsets/subset_tec_{opt.subset_type}mln.csv',
            'set_sw_path': f'{opt.bucket_dir}/karman-2025/data/sw_data/set_sw.csv',
            'celestrack_path': f'{opt.bucket_dir}/karman-2025/data/sw_data/celestrack_sw.csv',
            'omni_indices_path': f'{opt.bucket_dir}/karman-2025/data/omniweb_data/merged_omni_indices.csv',
            'omni_magnetic_field_path': f'{opt.bucket_dir}/karman-2025/data/omniweb_data/merged_omni_magnetic_field.csv',
            'omni_solar_wind_path': f'{opt.bucket_dir}/karman-2025/data/omniweb_data/merged_omni_solar_wind.csv',
            'jpld_path': f'{opt.bucket_dir}/jpld/subset_lat_lon/jpld_vtec_15min.csv',
            'use_celestrack': True,
            'use_set_sw': True,
            'use_jpld': True,
            'use_omni_indices': True,
            'use_omni_magnetic_field': True,
            'use_omni_solar_wind': True,
            'lag_days_proxies':opt.lag_days_proxies, # 81 days
            'proxies_resolution':opt.proxies_resolution,  # 1 day
            'lag_minutes_omni':opt.lag_minutes_omni,  # 2880 minutes (2 days)
            'omni_resolution':opt.omni_resolution,  # 1 minute
            'lag_minutes_jpld':opt.lag_minutes_jpld,  # 2880 minutes (2 days)
            'jpld_resolution':opt.jpld_resolution,  # 1 minute
    }
    madrigal_dataset = MadrigalDatasetTimeSeries(config,
                                                torch_type=torch_type)
    
    # set configuration
    num_historical_numeric=0

    if madrigal_dataset.config['omni_indices_path'] is not None:
        num_historical_numeric+=madrigal_dataset[0]['omni_indices'].shape[1]
    if madrigal_dataset.config['omni_magnetic_field_path'] is not None:
        num_historical_numeric+=madrigal_dataset[0]['omni_magnetic_field'].shape[1]
    if madrigal_dataset.config['omni_solar_wind_path'] is not None:
        num_historical_numeric+=madrigal_dataset[0]['omni_solar_wind'].shape[1]
    if madrigal_dataset.config['celestrack_path'] is not None:
        num_historical_numeric+=madrigal_dataset[0]['celestrack'].shape[1]
    if madrigal_dataset.config['set_sw_path'] is not None:
        num_historical_numeric+=madrigal_dataset[0]['set_sw'].shape[1]
    if madrigal_dataset.config['jpld_path'] is not None:
        num_historical_numeric+=madrigal_dataset[0]['jpld'].shape[1]

    print(f"Historical input features of the model: {num_historical_numeric}")

    input_dimension=len(madrigal_dataset[0]['inputs'])
    print(f"Static features of the model: {input_dimension}")

    if num_historical_numeric==0:
        raise ValueError('No historical numeric data found in the dataset')
    if opt.model_type=='tft':
        data_props = {'num_historical_numeric': num_historical_numeric,
                    'num_static_numeric': input_dimension,
                    'num_future_numeric': 1,
                    }

        configuration = {
                        'model':
                            {
                                'dropout': opt.dropout,
                                'state_size': opt.state_size,
                                'output_quantiles': [0.5], #[0.1, 0.5, 0.9],
                                'lstm_layers': opt.lstm_layers,
                                'attention_heads': opt.attention_heads,
                            },
                        'task_type': 'regression',
                        'target_window_start': None,
                        'data_props': data_props,
                        }
        # initialize TFT model 
        ts_ionopy_model = tft.TemporalFusionTransformer(OmegaConf.create(configuration))
        # weight init
        ts_ionopy_model.apply(weight_init)
    else:
        raise ValueError('Invalid model type. Only tft is supported')


    ts_ionopy_model.to(device)
    if opt.model_path != '':
        print(f"Loading Ionopy model from {opt.model_path}")
        ts_ionopy_model.load_state_dict(torch.load(opt.model_path))

    num_params=sum(p.numel() for p in ts_ionopy_model.parameters() if p.requires_grad)
    print(f'Ionopy model num parameters: {num_params}')

    idx_test_fold=2
    test_month_idx = 2 * (idx_test_fold - 1)
    validation_month_idx = test_month_idx + 2
    print(test_month_idx,validation_month_idx)
    madrigal_dataset._set_indices(test_month_idx=[test_month_idx], validation_month_idx=[validation_month_idx],custom={ 2012: {"validation":8, "test":9},
                                                                                                                        2013: {"validation":4, "test":5},
                                                                                                                        2015: {"validation":2, "test":3},#geomag storm
                                                                                                                        2019: {"validation":6, "test":10},#quiet period
                                                                                                                        2022: {"validation":0, "test":1},
                                                                                                                        2024: {"validation":4,"test":5}})
    train_dataset = madrigal_dataset.train_dataset()
    validation_dataset = madrigal_dataset.validation_dataset()
    test_dataset = madrigal_dataset.test_dataset()
    print(f'Training dataset example: {train_dataset[0].items()}')

    train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset))
    validation_sampler = RandomSampler(validation_dataset, num_samples=len(validation_dataset))
    test_sampler = SequentialSampler(test_dataset)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, list(ts_ionopy_model.parameters())),
        lr=opt.lr,
        amsgrad=True,
    )

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100,125,150,175,200,225,230,240,250,260,270], gamma=0.8, verbose=False)
    criterion=torch.nn.MSELoss()

    # And the dataloader
    #seed them
    g = torch.Generator()
    g.manual_seed(0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        sampler=train_sampler,
        drop_last=True,
        generator=g
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        sampler=validation_sampler,
        drop_last=True,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        sampler=test_sampler,
        drop_last=True,
        generator=g
    )
    criterion=torch.nn.MSELoss()
    quantiles_tensor= torch.tensor([0.5], dtype=torch_type, device=device)  # For TFT, we use the median quantile
    best_val_loss = np.inf
    if opt.wandb_inactive is False:
        wandb.config.update({
            'num_historical_numeric': num_historical_numeric,
            'num_static_numeric': input_dimension,
            'num_future_numeric': 1,
            'model_type': opt.model_type,
            'subset_type': opt.subset_type,
            'batch_size': opt.batch_size,
            'learning_rate': opt.lr,
            'epochs': opt.epochs,
            'dropout': opt.dropout,
            'state_size': opt.state_size,
            'lstm_layers': opt.lstm_layers,
            'attention_heads': opt.attention_heads
        })
        table = wandb.Table(columns=["Epoch", "Validation Loss", "Training Loss", "Validation RMSE", "Training RMSE", "Validation MAE", "Training MAE"])
    # CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
    for epoch in range(opt.epochs):
        ts_ionopy_model.train()
        count=0
        train_loss = 0.
        train_rmse_loss_unnormalized = 0.
        train_mae_unnormalized = 0.
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.epochs}"):
            historical_ts_numeric = []
            for key in batch:
                if key not in {'date', 'inputs', 'tec', 'dtec'}:    
                    if key in batch:
                        historical_ts_numeric.append(batch[key][:, :-1, :])
            if historical_ts_numeric:  # Only stack if not empty
                historical_ts_numeric = torch.cat(historical_ts_numeric, dim=2).to(device)
            future_ts_numeric=batch['jpld'][:,-1,:].unsqueeze(1).to(device)
            minibatch = {
                    'static_feats_numeric': batch['inputs'].to(device),
                    'historical_ts_numeric': historical_ts_numeric,
                    'future_ts_numeric':  future_ts_numeric,#batch size x future steps x num features
                    'target': batch['tec'].to(device)
                    }
                        
            #let's store the normalized and unnormalized target density:
            tec_log1p = minibatch['target'] * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
            tec_madrigal = torch.expm1(tec_log1p) 
            if opt.model_type=='tft':
                batch_out=ts_ionopy_model(minibatch)                    
                #now the quantiles:
                predicted_quantiles = batch_out['predicted_quantiles']#it's of shape batch_size x future_steps x num_quantiles
                target_nn_median=predicted_quantiles[:, :, 0].squeeze()
                q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_quantiles,
                                                                                targets=minibatch['target'],
                                                                                desired_quantiles=quantiles_tensor)
            else:
                raise ValueError('Invalid model type. Only tft is supported')
            loss_nn = criterion(target_nn_median, minibatch['target'])

            optimizer.zero_grad(set_to_none=True)
            loss_nn.backward()
            optimizer.step()

            #let's also track the non-normalized loss, without actually computing any gradients:
            target_nn_median_unnormalized = target_nn_median * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
            target_nn_median_unnormalized = torch.expm1(target_nn_median_unnormalized)
            loss_nn_unnormalized = criterion(target_nn_median_unnormalized.detach(), tec_madrigal.detach())
            #let's also record it to keep track of the average:
            train_loss += loss_nn.item()
            mae_value=mae_loss(target_nn_median_unnormalized.detach(), tec_madrigal.detach()).item()
            train_mae_unnormalized += mae_value
            train_rmse_loss_unnormalized += np.sqrt(loss_nn_unnormalized.item())
            if opt.wandb_inactive is False:
                wandb.log({
                    'train_loss': loss_nn.item(),
                    'train_loss_unnormalized': loss_nn_unnormalized.item(),
                    'train_rmse_unnormalized': np.sqrt(loss_nn_unnormalized.item()),
                    'train_train_mae_unnormalized': mae_value,
                    'train_q_loss': q_loss.item(),
                    'train_q_risk': q_risk.item(),
                    'train_minibatch': count
                })
            #every 100 epochs, print the losses:
            if (count+1) % 100 == 0:
                print(f"Epoch {count}, Train Loss: {loss_nn.item():.6f}, RMSE Loss: {np.sqrt(loss_nn_unnormalized.item())}, Train Loss Unnormalized: {loss_nn_unnormalized.item():.6f}, Q Loss: {q_loss.item():.6f}, Q Risk: {q_risk.item():.6f}")
            count+=1
        train_loss /= len(train_loader)
        train_rmse_loss_unnormalized /= len(train_loader)
        train_mae_unnormalized /= len(train_loader)
        scheduler.step()
        curr_lr=scheduler.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}, Average Train Loss: {train_loss:.8f}, Average Train RMSE Loss: {train_rmse_loss_unnormalized:.8f}")
        if opt.wandb_inactive is False:
            wandb.log({'train_loss_epoch': train_loss, 
                       'train_rmse_epoch': train_rmse_loss_unnormalized,
                       'train_mae_epoch': train_mae_unnormalized,
                       'epoch': epoch,
                       'learning_rate': curr_lr})
        #scheduler.step()
        #validation
        ts_ionopy_model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            validation_rmse_loss_unnormalized = 0.0
            validation_mae_unnormalized = 0.0
            count=0
            val_rmses=[]
            for batch in tqdm(validation_loader, desc=f"Validation Epoch {epoch+1}/{opt.epochs}"):
                historical_ts_numeric = []
                for key in batch:
                    if key not in {'date', 'inputs', 'tec', 'dtec'}:    
                        if key in batch:
                            historical_ts_numeric.append(batch[key][:, :-1, :])
                if historical_ts_numeric:
                    historical_ts_numeric = torch.cat(historical_ts_numeric, dim=2).to(device)
                future_ts_numeric=batch['jpld'][:,-1,:].unsqueeze(1).to(device)
                minibatch = {
                        'static_feats_numeric': batch['inputs'].to(device),
                        'historical_ts_numeric': historical_ts_numeric,
                        'future_ts_numeric':  future_ts_numeric,#batch size x future steps x num features
                        'target': batch['tec'].to(device)
                        }
                #let's store the normalized and unnormalized target density:
                tec_log1p = minibatch['target'] * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
                tec_madrigal = torch.expm1(tec_log1p)
                if opt.model_type=='tft':
                    batch_out=ts_ionopy_model(minibatch)
                    #now the quantiles:
                    predicted_quantiles = batch_out['predicted_quantiles']
                    target_nn_median=predicted_quantiles[:, :, 0].squeeze()
                    q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=target_nn_median,
                                                                                targets=minibatch['target'],
                                                                                desired_quantiles=quantiles_tensor)
                else:
                    raise ValueError('Invalid model type. Only tft is supported')
                loss_nn = criterion(target_nn_median, minibatch['target'])
                #let's also track the non-normalized loss, without actually computing any gradients:
                target_nn_median_unnormalized = target_nn_median * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
                target_nn_median_unnormalized = torch.expm1(target_nn_median_unnormalized)
                loss_nn_unnormalized = criterion(target_nn_median_unnormalized.detach(), tec_madrigal.detach())
                #let's also record it to keep track of the average:
                validation_loss += loss_nn.item()
                validation_rmse_loss_unnormalized += np.sqrt(loss_nn_unnormalized.item())
                mae_value=mae_loss(target_nn_median_unnormalized.detach(), tec_madrigal.detach()).item()
                validation_mae_unnormalized += mae_value
                val_rmses.append(validation_rmse_loss_unnormalized)
                if opt.wandb_inactive is False:
                    wandb.log({
                        'validation_loss': loss_nn.item(),
                        'validation_loss_unnormalized': loss_nn_unnormalized.item(),
                        'validation_rmse_unnormalized': np.sqrt(loss_nn_unnormalized.item()),
                        'validation_validation_mae_unnormalized': mae_value,
                        'validation_q_loss': q_loss.item(),
                        'validation_q_risk': q_risk.item(),
                        'validation_minibatch': count
                    })
                #every 100 epochs, print the losses:
                if (count+1) % 100 == 0:
                    print(f"minibatch {count}, Validation Loss: {loss_nn.item():.6f}, Validation RMSE Loss: {np.sqrt(loss_nn_unnormalized.item())}, Validation Loss Unnormalized: {loss_nn_unnormalized.item():.6f}, Q Loss: {q_loss.item():.6f}, Q Risk: {q_risk.item():.6f}")
                count+=1
        #let's create a W&B histogram on the val_losses:
        validation_loss /= len(validation_loader)
        validation_rmse_loss_unnormalized /= len(validation_loader)
        validation_mae_unnormalized /= len(validation_loader)
        print(f"Epoch {epoch+1}, Average Validation Loss: {validation_loss:.8f}, Average Validation RMSE Loss: {validation_rmse_loss_unnormalized:.8f}")
        if opt.wandb_inactive is False:
            try:
                wandb.log({f'validation_rmse_histogram_{epoch+1}': wandb.Histogram(val_rmses, num_bins=100, title='Validation RMSE')})
            except Exception as e:
                print(f"Error logging histogram: {e}")
            wandb.log({'validation_loss_epoch': validation_loss, 
                        'validation_rmse_epoch': validation_rmse_loss_unnormalized,
                        'validation_mae_epoch': validation_mae_unnormalized})
            try:
                wandb.log({f'validation_rmse_histogram_{epoch+1}': wandb.Histogram(val_rmses, num_bins=100, title='Validation RMSE')})
                table.add_data(epoch+1, 
                               validation_loss, 
                               train_loss, 
                               validation_rmse_loss_unnormalized, 
                               train_rmse_loss_unnormalized, 
                               validation_mae_unnormalized, 
                               train_mae_unnormalized)
            except Exception as e:
                print(f"Error logging: {e}")


        #save the model if the validation loss is lower than the best validation loss
        if validation_loss < best_val_loss:
            old_best = best_val_loss
            best_val_loss = validation_loss
            print(f"Validation loss improved from {old_best:.8f} to {validation_loss:.8f}, saving model")
            if opt.model_path != '':
                torch.save(ts_ionopy_model.state_dict(), opt.model_path)
            else:
                torch.save(ts_ionopy_model.state_dict(), f"{opt.model_type}_{opt.subset_type}mln_{timestamp_training}.pth")

if __name__ == "__main__":
    time_start = time.time()
    train()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
