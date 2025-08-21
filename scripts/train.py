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
torch.set_float32_matmul_precision('medium')  # or 'high'

def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_dataloaders(dataset, opt, seed=0):
    """Create train/validation/test loaders with proper seeding."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_sampler = RandomSampler(dataset.train_dataset(), generator=g)
    val_sampler = SequentialSampler(dataset.validation_dataset())
    test_sampler = SequentialSampler(dataset.test_dataset())

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset(),
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        sampler=train_sampler,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        dataset.validation_dataset(),
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        sampler=val_sampler,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset(),
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
        sampler=test_sampler,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    return train_loader, val_loader, test_loader

def run_epoch(ts_ionopy_model, dataloader, device, opt, scheduler, optimizer, epoch, mse_loss, quantiles_tensor, table, train=True):
    if train:    
        ts_ionopy_model.train()
        prefix='train'
    else:
        ts_ionopy_model.eval()
        prefix='val'
    
    count=0
    loss = 0.

    mean_rmse_loss_mean=0.
    mean_rmse_loss_std=0.
    mean_mae_loss_mean=0.
    mean_mae_loss_std=0.

    rmses_mean = []
    rmses_std = []
    #if it's validation we need torch no grad:
    if not train:
        torch.set_grad_enabled(False)
    else:
        torch.set_grad_enabled(True)
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs}"):
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

        dtec_log1p = (np.log1p(10.)-0.)* (batch['dtec'].to(device) + 1) / 2
        dtec_madrigal = torch.expm1(dtec_log1p)  # Unnormalize dTEC

        if opt.model_type=='tft':
            batch_out=ts_ionopy_model(minibatch)                    
            #now the quantiles:
            predicted_quantiles = batch_out['predicted_quantiles']#it's of shape batch_size x future_steps x num_quantiles
            target_nn_median = predicted_quantiles[:, :, 0]
            #target_nn_std is always in -1,1:
            target_nn_std = torch.tanh(predicted_quantiles[:, :, 1])

            #make the output always positive:

            #let's also track the non-normalized predictions
            #target_nn_median_unnormalized = target_nn_median * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
            #target_nn_median_unnormalized = torch.expm1(target_nn_median_unnormalized)
            target_nn_median_unnormalized = torch.expm1(target_nn_median * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P)
            #let's undo the min/max normalization of the std as well:
            target_nn_std_unnormalized = torch.expm1((np.log1p(10.) - 0.) * (target_nn_std + 1) / 2)

            #target_nn_median_unnormalized = predicted_quantiles_unnormalized[:, :, 1].squeeze()
            #q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=target_nn_median.detach(),
            #                                                                targets=minibatch['target'],
            #                                                                desired_quantiles=quantiles_tensor)
        else:
            raise ValueError('Invalid model type. Only tft is supported')
        if opt.loss_type == 'mse':    
            mse_loss_mean = mse_loss(target_nn_median.squeeze(), minibatch['target'])
            mse_loss_std = mse_loss(target_nn_std.squeeze(), batch['dtec'].to(device))
            loss_nn = mse_loss_mean + mse_loss_std
        else:
            raise ValueError('Invalid loss type. Only mse and kl are supported')
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss_nn.backward()
            optimizer.step()
        loss += loss_nn.item()
        # print(f"Shapes for mean: {target_nn_median_unnormalized.shape}, {tec_madrigal.shape}")
        # print(f"Shapes for std: {target_nn_std_unnormalized.shape}, {tec_madrigal.shape}")

        rmse_loss_mean_unnormalized = torch.sqrt(mse_loss(target_nn_median_unnormalized.detach().squeeze(), tec_madrigal.detach()))
        rmse_loss_std_unnormalized = torch.sqrt(mse_loss(target_nn_std_unnormalized.detach().squeeze(), dtec_madrigal.detach()))
        rmses_mean.append(rmse_loss_mean_unnormalized.item())
        rmses_std.append(rmse_loss_std_unnormalized.item())
        mean_rmse_loss_mean+= rmse_loss_mean_unnormalized.item()
        mean_rmse_loss_std+= rmse_loss_std_unnormalized.item()

        mae_loss_mean_unnormalized = mae_loss(target_nn_median_unnormalized.detach().squeeze(), tec_madrigal.detach()).item()
        mae_loss_std_unnormalized = mae_loss(target_nn_std_unnormalized.detach().squeeze(), dtec_madrigal.detach()).item()
        mean_mae_loss_mean+= mae_loss_mean_unnormalized
        mean_mae_loss_std+= mae_loss_std_unnormalized


        if opt.wandb_inactive is False:
            wandb.log({
                'loss': loss_nn.item(),
                f'{prefix}_rmse_loss_mean_unnormalized': rmse_loss_mean_unnormalized.item(),
                f'{prefix}_rmse_loss_std_unnormalized': rmse_loss_std_unnormalized.item(),
                f'{prefix}_mae_loss_mean_unnormalized': mae_loss_mean_unnormalized,
                f'{prefix}_mae_loss_std_unnormalized': mae_loss_std_unnormalized,
                'minibatch': count,
            })
        #every 100 epochs, print the losses:
        if (count+1) % 100 == 0:
            print(f"Epoch {epoch+1}, Minibatch {count+1}, {prefix} Loss: {loss_nn.item():.8f}, "
                    f"{prefix} RMSE Loss Mean Unnormalized: {rmse_loss_mean_unnormalized.item():.8f}, "
                    f"{prefix} RMSE Loss Std Unnormalized: {rmse_loss_std_unnormalized.item():.8f}, "
                    f"{prefix} MAE Loss Mean Unnormalized: {mae_loss_mean_unnormalized:.8f}, "
                    f"{prefix} MAE Loss Std Unnormalized: {mae_loss_std_unnormalized:.8f}")
        count+=1
    loss /= len(dataloader)
    mean_rmse_loss_mean /= len(dataloader)
    mean_rmse_loss_std /= len(dataloader)
    mean_mae_loss_mean /= len(dataloader)
    mean_mae_loss_std /= len(dataloader)

    scheduler.step()
    curr_lr=scheduler.optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}, Average {prefix} Loss: {loss:.8f}, Average {prefix} RMSE Loss for mean & std: {mean_rmse_loss_mean:.8f}, {mean_rmse_loss_std:.8f}")
    if opt.wandb_inactive is False:
        wandb.log({f'{prefix}_loss_epoch': loss, 
                    f'{prefix}_rmse_loss_mean_unnormalized': mean_rmse_loss_mean,
                    f'{prefix}_rmse_loss_std_unnormalized': mean_rmse_loss_std,
                    f'{prefix}_mae_loss_mean_unnormalized': mean_mae_loss_mean,
                    f'{prefix}_mae_loss_std_unnormalized': mean_mae_loss_std,
                    'epoch': epoch,
                    'learning_rate': curr_lr})
    # Histogram of RMSEs
    wandb.log({f"{prefix}_rmse_mean_hist_ep{epoch+1}": wandb.Histogram(rmses_mean)})
    wandb.log({f"{prefix}_rmse_std_hist_ep{epoch+1}": wandb.Histogram(rmses_std)})

    # Add to table
    if table is not None:
        table.add_data(
            epoch + 1, 
            loss, 
            mean_rmse_loss_mean,
            mean_mae_loss_mean, 
            mean_rmse_loss_std,
            mean_mae_loss_std
        )
    return loss

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
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
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
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse'],help='Loss to be used (MS is currently supported)')
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
            project="ionopy",
            entity="ionocast",
            config=vars(opt),
            name=f"{opt.model_type}_{opt.subset_type}mln_{timestamp_training}_Batch{opt.batch_size}_LSTM{opt.lstm_layers}_Att{opt.attention_heads}_SS{opt.state_size}",
        )
        print("W&B is active")
    
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=1)
    print()
    device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = torch.float32 if opt.torch_type == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    print(f"Using device: {device}, dtype: {dtype}")

    # Seed everything
    set_seed(opt.seed)

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
    madrigal_dataset = MadrigalDatasetTimeSeries(config, torch_type=dtype)

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
                                'output_quantiles': [0.5, 0.75], #[0.1, 0.5, 0.9],
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
        if torch.__version__ >= '2.0':
            print("Compiling model with torch.compile()...")
            ts_ionopy_model = torch.compile(ts_ionopy_model, backend='inductor')

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
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, list(ts_ionopy_model.parameters())),
        lr=opt.lr,
        amsgrad=True,
    )

    # And the dataloader
    #seed them
    train_loader, validation_loader, test_loader = prepare_dataloaders(madrigal_dataset, opt, seed=opt.seed)

    if opt.wandb_inactive is False:
        wandb.config.update({
            'num_historical_numeric': num_historical_numeric,
            'num_static_numeric': input_dimension,
            'num_future_numeric': madrigal_dataset[0]['jpld'].shape[1],
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
        table_training = wandb.Table(columns=["Epoch", "Training Loss", "Training Mean RMSE", "Training Mean MAE", "Training Std RMSE", "Training Std MAE"])
        table_validation = wandb.Table(columns=["Epoch", "Validation Loss", "Validation Mean RMSE", "Validation Mean MAE",  "Validation Std RMSE", "Validation Std MAE"])
    # CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)

    mse_loss=torch.nn.MSELoss()
    quantiles_tensor= torch.tensor([0.5, 0.75], device=device)  # For TFT, we use the median quantile
    best_val_loss = np.inf

    for epoch in range(opt.epochs):
        #first we run training:
        _ = run_epoch(ts_ionopy_model, train_loader, device, opt, scheduler, optimizer, epoch, mse_loss, quantiles_tensor, table=table_training, train=True)

        #then validation:
        validation_loss = run_epoch(ts_ionopy_model, validation_loader, device, opt, scheduler, optimizer, epoch, mse_loss, quantiles_tensor, table=table_validation, train=False)
        # Save best model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            save_path = opt.model_path or f"{opt.model_type}_{opt.subset_type}mln_{timestamp_training}_Batch{opt.batch_size}_LSTM{opt.lstm_layers}_Att{opt.attention_heads}_SS{opt.state_size}.pth"
            torch.save(ts_ionopy_model.state_dict(), save_path)
            print(f"New best model saved: {save_path} (Val Loss: {validation_loss:.6f})")

if __name__ == "__main__":
    time_start = time.time()
    train()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)
