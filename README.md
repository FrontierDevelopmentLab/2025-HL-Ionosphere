# NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments

This repository contains the code and resources for the Ionosphere-Thermosphere Twin project, part of the NASA Heliolab 2025 Frontier Development Lab. The project focuses on using machine learning to analyze and predict ionospheric conditions.

__Team:__ Linnea Wolniewicz, Halil Kelebek, Simone Mestici, Michael Vergalla

__Faculty:__ Giacomo Acciarini, Atilim Gunes Baydin, Tom Berger, Frank Soboczencki, James Walsh, Bala Poduval, Umaa Rebbapragada, Olga Verkhoglyadova

# Branch descriptions
main: LSTM training and models

graph-experiment: GNN training and models

SFNOpostFDL: Spherical FNO training and models

# Directories
/scripts contains all the dataset files (dataset_....py), event files (events.csv), model files (model_....py), train (run_....py) and evaluation (eval_....py) files, and utility files (util_....py).

/ionopy contains ionopy files (for the most up-to-date code refer to https://github.com/spaceml-org/ionopy).

/notebooks contains visualization and experimental notebooks.

/tests contains testing and experimental code.

/data contains events.csv.

### Pre-requisites
- Set up environment using the branch's environment.yml and requirements.txt
- Have the data available (public link incoming)

## Usage
In the graph-experiment branch, you can run the IonoPy and the IonCast GNN models.

The necessary environment can be created from environment.yml.

IonoPy:
**Code for this model may be out of date, for the most up-to-date code refer to https://github.com/spaceml-org/ionopy)**
- Code for creating the model is contained in the ionopy/ folder
- To train the model, run scripts/train_ionopy.py

IonCast GNN:
- All relevant datasets, models, events, and utility code are contained in the scripts/ folder
- To run the model, run scripts/run_ioncast.py
- To evaluate the model on test data, run scripts/run_ioncast.py in test mode.

### Training example

The following command runs the training script, assuming the current directory is `/scripts` and the data is stored in the directory `/mnt/data`. Results will be saved to the `/mnt/experiment-1' directory.

```bash
python run_ioncast.py --data_dir /mnt/data --aux_dataset sunmoon quasidipole celestrak omni set --mode train --target_dir /mnt/experiment-1 --num_workers 12 --batch_size 1 --model_type IonCastGNN --epochs 1000 --learning_rate 3e-4 --weight_decay 0.0 --context_window 8 --prediction_window 1 --num_evals 1 --jpld_weight 2.0 --date_start 2010-05-13T00:00:00 --date_end 2024-08-01T00:00:00 --mesh_level 6 --valid_every_nth_epoch 1 --save_all_models --residual_target --wandb_run_name IonCastGNN --date_dilation 256 --partition_size 1 --device cuda:1 --valid_event_id validation_events_1
```

# Acknowledgements
This work is the research product of FDL-X Heliolab a public/private partnership between NASA, Trillium Technologies Inc (trillium.tech) and commercial AI partners Google Cloud, NVIDIA and Pasteur Labs & ISI, developing open science for all Humankind.

# Licence
This project is licenced under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


Copyright 2025-2026 NASA  
Copyright 2026-2026 Trillium Technologies Inc
