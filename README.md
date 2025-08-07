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

## Usage

TO DO: Add better instructions.

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