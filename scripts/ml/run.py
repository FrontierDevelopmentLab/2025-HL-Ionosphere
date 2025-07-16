import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from util import Tee
from util import set_random_seed
from datasets import JPLDGIMDataset


matplotlib.use('Agg')


def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    parser.add_argument('--jpld_gim_dir', type=str, default='jpld_gim_20100513-20240731', help='JPLD GIM dataset directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    log_file = os.path.join(args.target_dir, 'log.txt')

    set_random_seed(args.seed)

    with Tee(log_file):
        print(description)
        print('Log file:', log_file)
        print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
        print('Config:')
        pprint.pprint(vars(args), depth=2, width=50)



        start_time = datetime.datetime.now()
        print('Start time: {}'.format(start_time))

        data_dir_jpld_gim = os.path.join(args.data_dir, args.jpld_gim_dir)

        dataset_jpld_gim = JPLDGIMDataset(data_dir_jpld_gim, normalize=True)

        if args.mode == 'train':
            print('Training mode selected.')
            # Here you would implement your training logic
            # For example, initializing a model, optimizer, and running the training loop
            pass
        elif args.mode == 'test':
            raise NotImplementedError("Testing mode is not implemented yet.")

        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()