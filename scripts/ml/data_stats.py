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
from src.omniweb_dataset import OMNIDataset


matplotlib.use('Agg')


def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, data statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    parser.add_argument('--jpld_gim_dir', type=str, default='jpld_gim_20100513-20240731', help='JPLD GIM dataset directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to use')
    parser.add_argument('--instruments', nargs='+', default=['jpld_gim', 'omniweb'], help='List of instruments to process')

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

        for instrument in args.instruments:
            if instrument == 'jpld_gim':
                runs = [
                    ('normalized', JPLDGIMDataset(data_dir_jpld_gim, normalize=True), 'JPLD GIM (normalized)'),
                    ('unnormalized', JPLDGIMDataset(data_dir_jpld_gim, normalize=False), 'JPLD GIM'),
                ]
            elif instrument == 'omniweb':
                runs = [
                    ('normalized', OMNIDataset(data_dir_jpld_gim, normalize=True), 'OMNIWEB (normalized)'),
                    ('unnormalized', OMNIDataset(data_dir_jpld_gim, normalize=False), 'OMNIWEB'),
                ]
            else:
                print(f"Instrument '{instrument}' not recognized. Skipping.")
                continue
            
            for postfix, dataset, label in runs:
                print('\nProcessing {} {}'.format(instrument, postfix))
                if len(dataset) < args.num_samples:
                    indices = list(range(len(dataset)))
                else:
                    indices = np.random.choice(len(dataset), args.num_samples, replace=False)

                data = []
                for i in tqdm(indices, desc='Processing samples', unit='sample'):
                    data.append(dataset[int(i)])

                data = torch.stack(data).flatten()
                print('Data shape: {}'.format(data.shape))
                
                data_mean = torch.mean(data)
                data_std = torch.std(data)
                data_min = data.min()
                data_max = data.max()
                print('Mean: {}'.format(data_mean))
                print('Std : {}'.format(data_std))
                print('Min : {}'.format(data_min))
                print('Max : {}'.format(data_max))

                file_name_stats = os.path.join(args.target_dir, '{}_{}_data_stats.txt'.format(instrument, postfix))
                print('Saving data stats: {}'.format(file_name_stats))
                with open(file_name_stats, 'w') as f:
                    f.write('Mean: {}\n'.format(data_mean))
                    f.write('Std : {}\n'.format(data_std))
                    f.write('Min : {}\n'.format(data_min))
                    f.write('Max : {}\n'.format(data_max))

                file_name_hist = os.path.join(args.target_dir, '{}_{}_data_stats.pdf'.format(instrument, postfix))
                print('Saving histogram : {}'.format(file_name_hist))
                hist_samples = 10000
                indices = np.random.choice(len(data), hist_samples, replace=True)
                hist_data = data[indices]
                plt.figure()
                plt.hist(hist_data, log=True, bins=100)
                plt.tight_layout()
                plt.savefig(file_name_hist) 


if __name__ == '__main__':
    main()