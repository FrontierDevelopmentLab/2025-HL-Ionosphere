import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
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

from torch import optim
from torch.utils.data import RandomSampler, SequentialSampler
import random

import re


model_paths=[
            # 'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-25-36_Batch256_LSTM2_Att2_SS32.pth',
            # 'FalseTrueFalseTrueFalseFalseFalse_tft_10mln_2025-08-25_12-32-35_Batch256_LSTM2_Att2_SS32.pth',
            # 'TrueFalseFalseTrueTrueTrueTrue_tft_10mln_2025-08-21_23-13-17_Batch256_LSTM2_Att2_SS64.pth',
            # 'TrueFalseTrueTrueTrueTrueFalse_tft_10mln_2025-08-21_23-14-31_Batch256_LSTM2_Att2_SS64.pth',
            # 'TrueFalseTrueFalseTrueTrueTrue_tft_10mln_2025-08-21_23-13-38_Batch256_LSTM2_Att2_SS64.pth',
            # 'FalseFalseFalseFalseTrueTrueTrue_tft_10mln_2025-08-21_23-32-04_Batch256_LSTM2_Att2_SS64.pth',
            # 'FalseFalseFalseTrueTrueTrueTrue_tft_10mln_2025-08-21_23-25-08_Batch256_LSTM2_Att2_SS64.pth',
            # 'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-33-49_Batch256_LSTM2_Att2_SS32.pth',
            # 'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-28-30_Batch256_LSTM2_Att2_SS32.pth',
            'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-34-57_Batch256_LSTM2_Att2_SS32.pth',
            'FalseTrueTrueTrueTrueTrueTrue_tft_10mln_2025-08-25_09-45-31_Batch256_LSTM2_Att2_SS32.pth',
            'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-19-13_Batch256_LSTM2_Att2_SS32.pth',
            'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-29-24_Batch256_LSTM2_Att2_SS32.pth',
            'TrueFalseTrueTrueTrueTrueTrue_tft_10mln_2025-08-21_23-15-50_Batch256_LSTM2_Att2_SS64.pth',
            'FalseFalseTrueTrueTrueTrueTrue_tft_10mln_2025-08-21_23-15-56_Batch256_LSTM2_Att2_SS64.pth',
            'FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_09-45-58_Batch256_LSTM2_Att2_SS32.pth',
            'TrueFalseTrueTrueFalseTrueTrue_tft_10mln_2025-08-21_23-14-44_Batch256_LSTM2_Att2_SS64.pth',
            'TrueFalseTrueTrueTrueFalseTrue_tft_10mln_2025-08-21_23-14-41_Batch256_LSTM2_Att2_SS64.pth'
]
#add the base dir to the path:
model_paths = [os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models', p) for p in model_paths]

lags_res_dict={}
#we store them as: lag_days, res_days, lag_minutes, res_minutes
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-25-36_Batch256_LSTM2_Att2_SS32.pth')]=[27,1,1620,60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueFalseFalseFalse_tft_10mln_2025-08-25_12-32-35_Batch256_LSTM2_Att2_SS32.pth')]=[81, 1, 8640, 60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','TrueFalseFalseTrueTrueTrueTrue_tft_10mln_2025-08-21_23-13-17_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','TrueFalseTrueTrueTrueTrueFalse_tft_10mln_2025-08-21_23-14-31_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','TrueFalseTrueFalseTrueTrueTrue_tft_10mln_2025-08-21_23-13-38_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseFalseFalseFalseTrueTrueTrue_tft_10mln_2025-08-21_23-32-04_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseFalseFalseTrueTrueTrueTrue_tft_10mln_2025-08-21_23-25-08_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-33-49_Batch256_LSTM2_Att2_SS32.pth')]=[108,1,6480,60]
# lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-28-30_Batch256_LSTM2_Att2_SS32.pth')]=[54, 1, 3240, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-34-57_Batch256_LSTM2_Att2_SS32.pth')]=[135, 1, 8100, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueTrueTrueTrueTrueTrue_tft_10mln_2025-08-25_09-45-31_Batch256_LSTM2_Att2_SS32.pth')]=[144, 1, 8640, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-19-13_Batch256_LSTM2_Att2_SS32.pth')]=[72, 1, 4320, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_12-29-24_Batch256_LSTM2_Att2_SS32.pth')]=[81, 1, 4860, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','TrueFalseTrueTrueTrueTrueTrue_tft_10mln_2025-08-21_23-15-50_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseFalseTrueTrueTrueTrueTrue_tft_10mln_2025-08-21_23-15-56_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','FalseTrueFalseTrueTrueTrueTrue_tft_10mln_2025-08-25_09-45-58_Batch256_LSTM2_Att2_SS32.pth')]=[144, 1, 8640, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','TrueFalseTrueTrueFalseTrueTrue_tft_10mln_2025-08-21_23-14-44_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]
lags_res_dict[os.path.join('/home/ga00693/Develop/2025-HL-Ionosphere/models','TrueFalseTrueTrueTrueFalseTrue_tft_10mln_2025-08-21_23-14-41_Batch256_LSTM2_Att2_SS64.pth')]=[144, 1, 8640, 60]



for model_path in model_paths:
    results={}
    # Define the flag names in the correct order
    flag_names = [
        "use_jpld",
        "use_timed",
        "use_set_sw",
        "use_celestrack",
        "use_omni_indices",
        "use_omni_magnetic_field",
        "use_omni_solar_wind"
    ]

    # --- Extract the flag string ---
    match = re.search(r'/([FTa-z]+)_tft', model_path)
    if match:
        flag_str = match.group(1)
    else:
        raise ValueError("Could not find flag string in path")

    # Find all True/False
    flags = re.findall(r'True|False', flag_str)
    if len(flags) != len(flag_names):
        raise ValueError("Number of flags does not match expected variables")

    # Map to dictionary
    flag_dict = {name: f == "True" for name, f in zip(flag_names, flags)}

    # --- Extract state size (SS##) ---
    state_match = re.search(r'_SS(\d+)', model_path)
    if state_match:
        state_size = int(state_match.group(1))
    else:
        raise ValueError("Could not find state size in path")

    print(flag_dict)
    print("state_size =", state_size)

    #from the model path name extract the flags for the use_* etc.:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using: ", device)
    torch_type = torch.float32# if torch_type == 'float32' else torch.float64
    torch.set_default_dtype(torch_type)

    bucket_dir='/home/ga00693/gcs-bucket'
    config={'madrigal_path': f'{bucket_dir}/madrigal_data/processed/gps_data_tarr/csv_subsets/subset_tec_10mln.csv',
            'set_sw_path': f'{bucket_dir}/karman-2025/data/sw_data/set_sw.csv',
            'celestrack_path': f'{bucket_dir}/karman-2025/data/sw_data/celestrack_sw.csv',
            'omni_indices_path': f'{bucket_dir}/karman-2025/data/omniweb_data/merged_omni_indices.csv',
            'omni_magnetic_field_path': f'{bucket_dir}/karman-2025/data/omniweb_data/merged_omni_magnetic_field.csv',
            'omni_solar_wind_path': f'{bucket_dir}/karman-2025/data/omniweb_data/merged_omni_solar_wind.csv',
            'jpld_path': f'{bucket_dir}/jpld/subset_lat_lon/jpld_vtec_15min.csv',
            'timed_path': f'{bucket_dir}/karman-2025/data/timed_see_level3_data/timed_see_level3.csv',
            'use_celestrack': flag_dict['use_celestrack'],
            'use_set_sw': flag_dict['use_set_sw'],
            'use_jpld': flag_dict['use_jpld'],
            'use_timed': flag_dict['use_timed'],
            'use_omni_indices': flag_dict['use_omni_indices'],
            'use_omni_magnetic_field': flag_dict['use_omni_magnetic_field'],
            'use_omni_solar_wind': flag_dict['use_omni_solar_wind'],
            'lag_days_proxies':lags_res_dict[model_path][0], # 81 days
            'proxies_resolution':lags_res_dict[model_path][1],  # 1 day
            'lag_minutes_omni':lags_res_dict[model_path][2],  # 4860 minutes (3 days)
            'omni_resolution':lags_res_dict[model_path][3],  # 1 minute
            'lag_minutes_jpld':lags_res_dict[model_path][2],  # 8640 minutes (6 days)
            'jpld_resolution':lags_res_dict[model_path][3],  # 1 minute
            'timed_resolution':lags_res_dict[model_path][1],  # 1 day
            'lag_days_timed':lags_res_dict[model_path][0],  # 81 days
    }
    madrigal_dataset = MadrigalDatasetTimeSeries(config,
                                                torch_type=torch_type)

    #we also create one where the config is identical but all the use_ flags are True:
    config_all_true = config.copy()
    for key in config_all_true.keys():
        if key.startswith('use_'):
            config_all_true[key] = True
    madrigal_dataset_all_true = MadrigalDatasetTimeSeries(config_all_true,
                                                torch_type=torch_type)


    # set configuration
    num_historical_numeric=0

    if madrigal_dataset.config['use_omni_indices'] is True:
        num_historical_numeric+=madrigal_dataset[0]['omni_indices'].shape[1]
    if madrigal_dataset.config['use_omni_magnetic_field'] is True:
        num_historical_numeric+=madrigal_dataset[0]['omni_magnetic_field'].shape[1]
    if madrigal_dataset.config['use_omni_solar_wind'] is True:
        num_historical_numeric+=madrigal_dataset[0]['omni_solar_wind'].shape[1]
    if madrigal_dataset.config['use_celestrack'] is True:
        num_historical_numeric+=madrigal_dataset[0]['celestrack'].shape[1]
    if madrigal_dataset.config['use_set_sw'] is True:
        num_historical_numeric+=madrigal_dataset[0]['set_sw'].shape[1]
    if madrigal_dataset.config['use_jpld'] is True:
        num_historical_numeric+=madrigal_dataset[0]['jpld'].shape[1]
        num_future_numeric=madrigal_dataset[0]['jpld'].shape[1]
    else:
        num_future_numeric=1
    if madrigal_dataset.config['use_timed'] is True:
        num_historical_numeric+=madrigal_dataset[0]['timed'].shape[1]

    print(f"Historical input features of the model: {num_historical_numeric}")

    input_dimension=len(madrigal_dataset[0]['inputs'])
    print(f"Static features of the model: {input_dimension}")

    if num_historical_numeric==0:
        raise ValueError('No historical numeric data found in the dataset')

    data_props = {'num_historical_numeric': num_historical_numeric,
                'num_static_numeric': input_dimension,
                'num_future_numeric': num_future_numeric,
                }


    configuration = {
                    'model':
                        {
                            'dropout': 0.1,
                            'state_size': state_size,
                            'output_quantiles': [0.5,0.75], #[0.1, 0.5, 0.9],
                            'lstm_layers': 2,
                            'attention_heads': 2,
                        },
                    'task_type': 'regression',
                    'target_window_start': None,
                    'data_props': data_props,
                    }
    # initialize TFT model 
    ts_ionopy_model = tft.TemporalFusionTransformer(OmegaConf.create(configuration))
    # weight init
    ts_ionopy_model.apply(weight_init)
    
    #print(madrigal_dataset[0]['omni_indices'].shape, madrigal_dataset[0]['celestrack'].shape, madrigal_dataset[0]['omni_magnetic_field'].shape, madrigal_dataset[0]['omni_solar_wind'].shape)

    state_dict=torch.load(model_path,map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    missing, unexpected = ts_ionopy_model.load_state_dict(new_state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    ts_ionopy_model.to(device)
    print(f"Loading Ionopy model from {model_path}")
    ts_ionopy_model.load_state_dict(new_state_dict)
    #compile it
    #ts_ionopy_model = torch.compile(ts_ionopy_model)
    ts_ionopy_model.eval()

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
    madrigal_dataset_all_true._set_indices(test_month_idx=[test_month_idx], validation_month_idx=[validation_month_idx],custom={ 2012: {"validation":8, "test":9},
                                                                                                                        2013: {"validation":4, "test":5},
                                                                                                                        2015: {"validation":2, "test":3},#geomag storm
                                                                                                                        2019: {"validation":6, "test":10},#quiet period
                                                                                                                        2022: {"validation":0, "test":1},
                                                                                                                        2024: {"validation":4,"test":5}})
    train_dataset = madrigal_dataset.train_dataset()
    train_dataset_all_true = madrigal_dataset_all_true.train_dataset()
    validation_dataset = madrigal_dataset.validation_dataset()
    validation_dataset_all_true = madrigal_dataset_all_true.validation_dataset()
    test_dataset = madrigal_dataset.test_dataset()
    test_dataset_all_true = madrigal_dataset_all_true.test_dataset()

    train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset))
    train_sampler_all_true = RandomSampler(train_dataset_all_true, num_samples=len(train_dataset_all_true))
    validation_sampler = SequentialSampler(validation_dataset)
    validation_sampler_all_true = SequentialSampler(validation_dataset_all_true)
    test_sampler = SequentialSampler(test_dataset)
    test_sampler_all_true = SequentialSampler(test_dataset_all_true)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100,125,150,175,200,225,230,240,250,260,270], gamma=0.8, verbose=False)
    criterion=torch.nn.MSELoss()

    seed=0
    batch_size=256
    num_workers=24

    # And the dataloader
    #seed them
    g = torch.Generator()
    g.manual_seed(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=True,
        generator=g
    )
    train_loader_all_true = torch.utils.data.DataLoader(
        train_dataset_all_true,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=train_sampler_all_true,
        drop_last=True,
        generator=g
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=validation_sampler,
        drop_last=False,
        generator=g
    )
    validation_loader_all_true = torch.utils.data.DataLoader(
        validation_dataset_all_true,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=validation_sampler_all_true,
        drop_last=False,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=test_sampler,
        drop_last=False,
        generator=g
    )
    test_loader_all_true = torch.utils.data.DataLoader(
        test_dataset_all_true,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=test_sampler_all_true,
        drop_last=False,
        generator=g
    )

    import torch
    from tqdm import tqdm

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    ts_ionopy_model.eval()

    # ===========================
    # Define known categorical classes
    # ===========================
    latitude_classes = ['High latitude (>60 deg)', 'Low latitude (<30 deg)', 'Mid latitude (30-60 deg)']
    storm_classes = ['G0', 'G1', 'G2', 'G3', 'G4']
    solar_activity_classes = [
        'F10.7: 0-70 (low)',
        'F10.7: 70-150 (moderate)',
        'F10.7: 150-200 (moderate-high)',
        'F10.7: 200 (high)'
    ]

    # ===========================
    # Metric helpers
    # ===========================
    def mape_loss(y_pred, y_true, eps=1e-8):
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + eps)))

    def init_metrics_dict(class_list):
        return {cls: {'rmse_mean': 0., 'rmse_std': 0.,
                    'mae_mean': 0., 'mae_std': 0.,
                    'mape_mean': 0., 'mape_std': 0.,
                    'count': 0} for cls in class_list}

    metrics_lat   = init_metrics_dict(latitude_classes)
    metrics_storm = init_metrics_dict(storm_classes)
    metrics_solar = init_metrics_dict(solar_activity_classes)

    # ===========================
    # Helper function
    # ===========================
    def accumulate_metrics(class_labels, class_dict, pred_mean, pred_std, target_mean, target_std):
        """Accumulate metrics per class using boolean masking"""
        if class_labels is None:
            return
        unique_classes = list(class_dict.keys())
        mapping = {c: i for i, c in enumerate(unique_classes)}
        indices = torch.tensor([mapping[c] for c in class_labels], device=pred_mean.device)

        for i, cls in enumerate(unique_classes):
            mask = indices == i
            if mask.sum() == 0:
                continue

            rmse_mean = torch.sqrt(mse_loss(pred_mean[mask].squeeze(), target_mean[mask].squeeze())).item()
            rmse_std  = torch.sqrt(mse_loss(pred_std[mask].squeeze(), target_std[mask].squeeze())).item()
            mae_mean_val = mae_loss(pred_mean[mask].squeeze(), target_mean[mask].squeeze()).item()
            mae_std_val  = mae_loss(pred_std[mask].squeeze(), target_std[mask].squeeze()).item()
            mape_mean_val = mape_loss(pred_mean[mask].squeeze(), target_mean[mask].squeeze()).item()
            mape_std_val  = mape_loss(pred_std[mask].squeeze(), target_std[mask].squeeze()).item()

            n = mask.sum().item()
            class_dict[cls]['rmse_mean'] += rmse_mean * n
            class_dict[cls]['rmse_std']  += rmse_std  * n
            class_dict[cls]['mae_mean']  += mae_mean_val * n
            class_dict[cls]['mae_std']   += mae_std_val  * n
            class_dict[cls]['mape_mean'] += mape_mean_val * n
            class_dict[cls]['mape_std']  += mape_std_val * n
            class_dict[cls]['count']     += n

    def finalize_metrics(metrics_dict):
        results = {}
        for cls, vals in metrics_dict.items():
            if vals['count'] > 0:
                results[cls] = {
                    'RMSE Mean': vals['rmse_mean'] / vals['count'],
                    'RMSE Std':  vals['rmse_std']  / vals['count'],
                    'MAE Mean':  vals['mae_mean']  / vals['count'],
                    'MAE Std':   vals['mae_std']   / vals['count'],
                    'MAPE Mean': vals['mape_mean'] / vals['count'],
                    'MAPE Std':  vals['mape_std']  / vals['count'],
                }
        return results

    # ===========================
    # Global metric trackers
    # ===========================
    count=0
    loss = 0.
    mean_rmse_loss_mean=0.
    mean_rmse_loss_std=0.
    mean_mae_loss_mean=0.
    mean_mae_loss_std=0.
    mean_mape_loss_mean=0.
    mean_mape_loss_std=0.

    torch.set_grad_enabled(False)

    # ===========================
    # Evaluation loop
    # ===========================
    for batch, batch_all_true in tqdm(zip(test_loader, test_loader_all_true)):
        historical_ts_numeric = []
        future_ts_numeric = None
        for key in batch:
            if key not in {'date', 'inputs', 'tec', 'dtec', 'storm_classification',
                        'latitude_classification', 'solar_activity_classification'}:    
                if key in batch:
                    historical_ts_numeric.append(batch[key][:, :-1, :])
            if key == 'jpld':
                future_ts_numeric = batch['jpld'][:, -1, :].unsqueeze(1).to(device)

        if historical_ts_numeric:
            historical_ts_numeric = torch.cat(historical_ts_numeric, dim=2).to(device)

        minibatch = {
            'static_feats_numeric': batch['inputs'].to(device),
            'historical_ts_numeric': historical_ts_numeric,
            'target': batch['tec'].to(device)
        }
        if future_ts_numeric is not None:
            minibatch['future_ts_numeric'] = future_ts_numeric.to(device)
        else:
            minibatch['future_ts_numeric'] = torch.zeros(
                historical_ts_numeric.shape[0], 1, historical_ts_numeric.shape[2], device=device
            )

        # ===========================
        # Targets (unnormalized)
        # ===========================
        tec_log1p = minibatch['target'] * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
        tec_madrigal = torch.expm1(tec_log1p) 

        dtec_log1p = (np.log1p(10.)-0.) * (batch['dtec'].to(device) + 1) / 2
        dtec_madrigal = torch.expm1(dtec_log1p)

        # ===========================
        # Forward pass
        # ===========================
        batch_out=ts_ionopy_model(minibatch)                    
        predicted_quantiles = batch_out['predicted_quantiles']
        target_nn_median = predicted_quantiles[:, :, 0]
        target_nn_std = torch.tanh(predicted_quantiles[:, :, 1])

        target_nn_median_unnormalized = torch.expm1(
            target_nn_median * ionopy.dataset.TEC_STD_LOG1P + ionopy.dataset.TEC_MEAN_LOG1P
        )
        target_nn_std_unnormalized = torch.expm1(
            (np.log1p(10.) - 0.) * (target_nn_std + 1) / 2
        )

        # ===========================
        # Losses (normalized space)
        # ===========================
        mse_loss_mean = mse_loss(target_nn_median.squeeze(), minibatch['target'])
        mse_loss_std = mse_loss(target_nn_std.squeeze(), batch['dtec'].to(device))
        loss_nn = mse_loss_mean + mse_loss_std
        loss += loss_nn.item()

        # ===========================
        # Global metrics (unnormalized)
        # ===========================
        rmse_loss_mean_unnormalized = torch.sqrt(mse_loss(
            target_nn_median_unnormalized.detach().squeeze(), tec_madrigal.detach()
        ))
        rmse_loss_std_unnormalized = torch.sqrt(mse_loss(
            target_nn_std_unnormalized.detach().squeeze(), dtec_madrigal.detach()
        ))

        mean_rmse_loss_mean+= rmse_loss_mean_unnormalized.item()
        mean_rmse_loss_std+= rmse_loss_std_unnormalized.item()

        mae_loss_mean_unnormalized = mae_loss(
            target_nn_median_unnormalized.detach().squeeze(), tec_madrigal.detach()
        ).item()
        mae_loss_std_unnormalized = mae_loss(
            target_nn_std_unnormalized.detach().squeeze(), dtec_madrigal.detach()
        ).item()

        mean_mae_loss_mean+= mae_loss_mean_unnormalized
        mean_mae_loss_std+= mae_loss_std_unnormalized

        mape_loss_mean_unnormalized = mape_loss(
            target_nn_median_unnormalized.detach().squeeze(), tec_madrigal.detach()
        ).item()
        mape_loss_std_unnormalized = mape_loss(
            target_nn_std_unnormalized.detach().squeeze(), dtec_madrigal.detach()
        ).item()

        mean_mape_loss_mean+= mape_loss_mean_unnormalized
        mean_mape_loss_std+= mape_loss_std_unnormalized

        # ===========================
        # Per-class metrics
        # ===========================
        accumulate_metrics(batch['latitude_classification'],
                        metrics_lat,
                        target_nn_median_unnormalized, target_nn_std_unnormalized,
                        tec_madrigal, dtec_madrigal)

        accumulate_metrics(batch_all_true['storm_classification'],
                            metrics_storm,
                            target_nn_median_unnormalized, target_nn_std_unnormalized,
                            tec_madrigal, dtec_madrigal)

        accumulate_metrics(batch_all_true['solar_activity_classification'],
                            metrics_solar,
                            target_nn_median_unnormalized, target_nn_std_unnormalized,
                            tec_madrigal, dtec_madrigal)

        # ===========================
        # Debug print
        # ===========================
        if (count+1) % 100 == 0:
            print(f"Loss: {loss_nn.item():.8f}, "
                f"RMSE Mean: {rmse_loss_mean_unnormalized.item():.8f}, "
                f"RMSE Std: {rmse_loss_std_unnormalized.item():.8f}, "
                f"MAE Mean: {mae_loss_mean_unnormalized:.8f}, "
                f"MAE Std: {mae_loss_std_unnormalized:.8f}, "
                f"MAPE Mean: {mape_loss_mean_unnormalized:.8f}, "
                f"MAPE Std: {mape_loss_std_unnormalized:.8f}")
        count+=1

    # ===========================
    # Averages
    # ===========================
    loss /= len(test_loader)
    mean_rmse_loss_mean /= len(test_loader)
    mean_rmse_loss_std /= len(test_loader)
    mean_mae_loss_mean /= len(test_loader)
    mean_mae_loss_std /= len(test_loader)
    mean_mape_loss_mean /= len(test_loader)
    mean_mape_loss_std /= len(test_loader)

    results_lat   = finalize_metrics(metrics_lat)
    results_storm = finalize_metrics(metrics_storm)
    results_solar = finalize_metrics(metrics_solar)

    results['latitude metrics'] = results_lat
    results['storm metrics'] = results_storm
    results['solar activity metrics'] = results_solar
    results['global metrics'] = {
        'MSE Loss': loss,
        'Avg RMSE Mean': mean_rmse_loss_mean,
        'Avg RMSE Std': mean_rmse_loss_std,
        'Avg MAE Mean': mean_mae_loss_mean,
        'Avg MAE Std': mean_mae_loss_std,
        'Avg MAPE Mean': mean_mape_loss_mean,
        'Avg MAPE Std': mean_mape_loss_std
    }

    print(f"\nGlobal Loss: {loss:.8f}, "
        f"Avg RMSE Mean: {mean_rmse_loss_mean:.8f}, "
        f"Avg RMSE Std: {mean_rmse_loss_std:.8f}, "
        f"Avg MAE Mean: {mean_mae_loss_mean:.8f}, "
        f"Avg MAE Std: {mean_mae_loss_std:.8f}, "
        f"Avg MAPE Mean: {mean_mape_loss_mean:.8f}, "
        f"Avg MAPE Std: {mean_mape_loss_std:.8f}\n")

    print("Latitude metrics:", results_lat)
    print("Storm metrics:", results_storm)
    print("Solar activity metrics:", results_solar)
    #save results to a json file with the name of the model_path
    with open(model_path.replace('.pth', '_test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
