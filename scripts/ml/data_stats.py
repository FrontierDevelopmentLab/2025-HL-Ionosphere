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
from dataset_jpld import JPLD
from dataset_celestrak import CelesTrak
from dataset_omniweb import OMNIWeb, omniweb_all_columns


matplotlib.use('Agg')


def sanitize_filename(filename):
    """Removes or replaces characters that are invalid for file paths."""
    return filename.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')


def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, data statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--jpld_dir', type=str, default='jpld/webdataset', help='JPLD GIM dataset directory')
    parser.add_argument('--celestrak_file_name', type=str, default='celestrak/kp_ap_processed_timeseries.csv', help='CelesTrak dataset file name')
    parser.add_argument('--omniweb_dir', type=str, default='omniweb_karman_2025', help='OMNIWeb dataset directory')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to use')
    parser.add_argument('--instruments', nargs='+', default=['jpld', 'celestrak', 'omniweb'], help='List of datasets to process')

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

        data_dir_jpld = os.path.join(args.data_dir, args.jpld_dir)
        data_dir_omniweb = os.path.join(args.data_dir, args.omniweb_dir)
        dataset_celestrak_file_name = os.path.join(args.data_dir, args.celestrak_file_name)

        for instrument in args.instruments:
            if instrument == 'jpld':
                runs = [
                    ('normalized', JPLD(data_dir_jpld, normalize=True), 'JPLD (normalized)'),
                    ('unnormalized', JPLD(data_dir_jpld, normalize=False), 'JPLD (unnormalized)'),
                ]
            elif instrument == 'celestrak':
                runs = []
                for column in ['Kp', 'Ap']:
                    runs.append((f'normalized_{column}', CelesTrak(dataset_celestrak_file_name, normalize=True, column=[column]), f'CELESTRAK {column} (normalized)'))
                    runs.append((f'unnormalized_{column}', CelesTrak(dataset_celestrak_file_name, normalize=False, column=[column]), f'CELESTRAK {column} (unnormalized)'))
            elif instrument == 'omniweb':
                runs = []
                for column in omniweb_all_columns:
                    runs.append((f'normalized_{column}', OMNIWeb(data_dir_omniweb, normalize=True, column=[column]), f'OMNIWeb {column} (normalized)'))
                    runs.append((f'unnormalized_{column}', OMNIWeb(data_dir_omniweb, normalize=False, column=[column]), f'OMNIWeb {column} (unnormalized)'))
            else:
                print(f"Instrument '{instrument}' not recognized. Skipping.")
                continue
            
            for postfix, dataset, label in runs:
                print('\nProcessing {} {}'.format(instrument, postfix))

                # Sanitize the postfix to create a valid filename
                postfix = sanitize_filename(postfix)

                if len(dataset) < args.num_samples:
                    indices = list(range(len(dataset)))
                else:
                    indices = np.random.choice(len(dataset), args.num_samples, replace=False)

                data = []
                for i in tqdm(indices, desc='Processing samples', unit='sample'):
                    d, date = dataset[int(i)]
                    data.append(d)

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

                if instrument != 'jpld':
                    # plot the whole dataset time series
                    dates = []
                    values = []
                    for i in range(0, len(dataset), len(dataset)//args.num_samples):
                        d = dataset[i]
                        dates.append(d[1])
                        values.append(d[0])

                    file_name_ts = os.path.join(args.target_dir, '{}_{}_time_series.pdf'.format(instrument, postfix))
                    print('Saving time series: {}'.format(file_name_ts))
                    plt.figure(figsize=(24,6))
                    plt.plot(dates, values)
                    plt.ylabel(label)
                    # Limit number of xticks
                    plt.xticks(np.arange(0, len(dates), step=len(dates)//40))
                    # Rotate xticks
                    plt.xticks(rotation=45)
                    # Shift xticks so that the end of the text is at the tick
                    plt.xticks(ha='right')
                    plt.tight_layout()
                    plt.savefig(file_name_ts)

if __name__ == '__main__':
    main()