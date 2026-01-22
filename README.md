# NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments

This repository contains the code and resources for the Ionosphere-Thermosphere Twin project, part of the NASA Heliolab 2025 Frontier Development Lab. The project focuses on using machine learning to analyze and predict ionospheric conditions.

__Team:__ Linnea Wolniewicz, Halil Kelebek, Simone Mestici, Michael Vergalla

__Faculty:__ Giacomo Acciarini, Atilim Gunes Baydin, Tom Berger, Frank Soboczencki, James Walsh, Bala Poduval, Umaa Rebbapragada, Olga Verkhoglyadova

## How to install

### Pre-requisites
- Install Docker for your platform: [Get Docker](https://docs.docker.com/get-started/get-docker/)
- Install NVIDIA Container Toolkit for GPU support: [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Clone the repository:
  ```bash
  git clone git@github.com:FrontierDevelopmentLab/2025-HL-Ionosphere.git
  cd 2025-HL-Ionosphere
  ```
- Build the Docker image:
  ```bash
  docker build -t ioncast .
  ```
  This command builds a Docker image with the name `ioncast:latest`.

## Usage

This Branch is for Running the NVIDIA Inspired SFNO Model. 
There are custom run.py additions that vary from the main branch. 

Below is an example run command for the Training mode. 

CUDA_LAUNCH_BLOCKING=1 python run.py   --mode train   --model_type SphericalFourierNeuralOperatorModel   --device cuda:0   --data_dir /mnt/disks/disk-main-data-1   --target_dir /mnt/disks/disk-main-data-1/letsgo2   --date_start "2010-05-13 00:00:00"   --date_end   "2024-08-01 00:00:00"   --celestrak_file_name celestrak/kp_ap_processed_timeseries.csv   --set_file_name set/karman-2025_data_sw_data_set_sw.csv   --delta_minutes 15   --date_dilation 256   --epochs 800   --batch_size 30  --learning_rate 0.00013   --weight_decay 1e-6   --num_workers 2   --context_window 6   --prediction_window 2   --spectral_backend sht   --jpld_weight 1.0   --aux_weight 0.3   --eval_mode all   --valid_every_nth_epoch 20   --n_harmonics 4   --sfno_depth 8   --sfno_modes_lat 48   --sfno_modes_lon 64   --sfno_width 64   --n_sunlocked_heads 360   --dropout 0.1   --head_blend_sigma 2.0   --head_smooth_reg 1e-4   --lon_tv_reg 1e-5   --lon_highfreq_reg 5e-7   --lon_highfreq_kmin 72   --lon_blur_sigma_deg 6.0   --log_every 200   --amp   --channels_last   --area_weighted_loss   --wandb_mode online   --wandb_run_name "final2"


Below is an example run command for running Test mode. 

CUDA_LAUNCH_BLOCKING=1 python run.py \
  --mode test \
  --device cuda:0 \
  --data_dir /mnt/disks/disk-main-data-1 \
  --target_dir /mnt/disks/disk-main-data-1/letsgo2/test_evalmax \
  --model_file /mnt/disks/disk-main-data-1/letsgo2/best_model/epoch-340-model.pth \
  --context_window 6 \
  --prediction_window 2 \
  --date_dilation 256 \
  --delta_minutes 15 \
  --n_harmonics 4 \
  --head_blend_sigma 2.0 \
  --lon_blur_sigma_deg 6.0 \
  --eval_mode all \
  --lead_times 30 60 90 120 150 180 210 240 270 300 330 360 390 420 450 480 510 540 570 600 630 660 690 720 \
  --test_event_id G0H12-201804202100 G2H12-201509071500 G4H12-201506221500

NOTES for the FUTURE:
-its possible my latband ensemble is not active in loss and predict, but is in forward.. 
Missed this. 

Suggestion for future. Create models per lat band.. And make full model ensemble these.

### Training example

The following command runs the training script using Docker, mounting the current directory as `/mnt` and a data directory as `/mnt/data` inside the container.

```bash
docker run --rm -it \
    --ipc=host \
    --gpus all \
    -v $PWD:/mnt \
    -v /disk2-ssd-8tb/data/2025-hl-ionosphere:/mnt/data \
    ioncast python run.py \
        --data_dir /mnt/data \
        --mode train \
        --device cuda:0 \
        --target_dir /mnt/experiment-1 \
        --num_workers 12 \
        --batch_size 4 \
        --epochs 10
```

### Help

To see all available options and configurations, run:
```bash
docker run --rm -it ioncast python run.py --help
```

## Acknowledgements

This work is the research product of FDL-X Heliolab a public/private partnership between NASA, Trillium Technologies Inc (trillium.tech) and commercial AI partners Google Cloud, NVIDIA and Pasteur Labs & ISI, developing open science for all Humankind.
