# NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments

This repository contains the code and resources for the Ionosphere-Thermosphere Twin project, part of the NASA Heliolab 2025 Frontier Development Lab. The project focuses on using machine learning to analyze and predict ionospheric conditions.

__Team:__ Linnea Wolniewicz, Halil Kelebek, Simone Mestici, Michael Vergalla

__Faculty:__ Giacomo Acciarini, Atilim Gunes Baydin, Tom Berger, Frank Soboczencki, James Walsh, Bala Poduval, Umaa Rebbapragada, Olga Verkhoglyadova

# Branch descriptions
main: LSTM training and models

ioncast-gnn: GNN training and models

ioncast-sfno: Spherical FNO training and models

# Directories
/scripts contains all the dataset files (dataset_....py), event files (events.csv), model files (model_....py), train (run_....py) and evaluation (eval_....py) files, and utility files (util_....py).

/ionopy contains ionopy files (for the most up-to-date code refer to https://github.com/spaceml-org/ionopy).

/notebooks contains visualization and experimental notebooks.

/tests contains testing and experimental code.

### Pre-requisites
- Set up environment using the branch's environment.yml and requirements.txt
- Have the data available (public link incoming)

## Usage
In the main branch, you can run the IonoPy and the IonCast LSTM models. 

The necessary environment can be created from requirements.txt.

IonoPy:
**Code for this model may be out of date, for the most up-to-date code refer to https://github.com/spaceml-org/ionopy)**
- Code for creating the model is contained in the ionopy/ folder
- To train the model, run scripts/train_ionopy.py

IonCast LSTM:
- All relevant datasets, models, events, and utility code are contained in the scripts/ folder
- To run the model, run scripts/run.py
- To evaluate the model on test data, run scripts/run.py in test mode.

### Training example

The following command runs the training script, assuming the current directory is `/scripts` and the data is stored in the directory `/mnt/data`. Results will be saved to the `/mnt/experiment-1' directory.

```bash
python run.py --data_dir /mnt/data --mode train --target_dir /mnt/experiment-1 --num_workers 4 --batch_size 4 --model_type IonCastConvLSTM --epochs 2 --learning_rate 1e-3 --weight_decay 0.0 --context_window 4 --prediction_window 4 --num_evals 4 --date_start 2023-07-01T00:00:00 --date_end 2023-08-01T00:00:00
```

Copyright 2025-2026 NASA
Copyright 2026-2026 Trillium Technologies Inc

Licensed under the Apache License, Version 2.0...

This work is the research product of FDL-X Heliolab a public/private partnership between NASA, Trillium Technologies Inc (trillium.tech) and commercial AI partners Google Cloud, NVIDIA and Pasteur Labs & ISI, developing open science for all Humankind.
