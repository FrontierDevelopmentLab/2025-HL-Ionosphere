import argparse
import datetime
import os
import sys
import time
from glob import glob
import pprint
import pandas as pd
from tqdm import tqdm
import numpy as np

def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, process omniweb data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dir', type=str, help='Input directory with raw files', required=True)
    parser.add_argument('--target_dir', type=str, help='Target directory for csv.gz files', required=True)
    
    args = parser.parse_args()
    
    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    os.makedirs(args.target_dir, exist_ok=True)

    file_names = glob(os.path.join(args.input_dir, '*.csv'))

    if len(file_names) == 0:
        print('No files found in the input directory.')
        return
    
    print('Found {:,} files in the input directory.'.format(len(file_names)))
    dfs = []
    for file in tqdm(file_names, desc="reading in data"):
        df = pd.read_csv(file) 
        dfs.append(df)
    
    mean_df = compute_mean(dfs)
    std_df = compute_std(dfs)

    # save mean and stddev to new csv 
    column_names = dfs[0].loc[0].tolist()
    omni_stats = pd.DataFrame([mean_df.values, std_df.values], columns=column_names)

    stats_path = os.path.join(args.target_dir, "dataset_stats","omni_stats.csv")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    omni_stats.to_csv(stats_path, index=False)

    # forward fill nans
    for file in tqdm(file_names, desc="removes nan with forward filling"):
        target_file = os.path.join(args.target_dir, os.path.basename(file))
        df_ffill = df.ffill()# remove nans by replacing nan values with value at previous timestamp (5 minute cadence)
        df_ffill.to_csv(target_file, index=False, compression="gzip") # kept same format for consistency but could also be unzipped

    print(f"saved dataset to {args.target_dir}")
    print(f"saved statistics to {stats_path}")

def compute_mean(dfs):
    col_sums = None
    col_counts = None
    for i, df in tqdm(enumerate(dfs), desc="computing the dataset mean", total=len(dfs)):
        if i == 0:
            column_names = df.loc[0].tolist()
            print(column_names)
        assert column_names == df.loc[0].tolist(), "All dataframes must have the same columns in the same order."
        num_df = df.apply(pd.to_numeric, errors='coerce')
        if col_sums is None:
            col_sums = num_df.sum(skipna=True)
            col_counts = num_df.count()
        else:
            col_sums += num_df.sum()
            col_counts += num_df.count()
    mean_df = col_sums / col_counts
    return mean_df

def compute_std(dfs):
    col_counts = None
    col_sums = None
    col_sumsq = None  # sum of squares

    for i, df in tqdm(enumerate(dfs), desc="computing the dataset std", total=len(dfs)):
        if i == 0:
            column_names = df.loc[0].tolist()
            print(column_names)
        assert column_names == df.loc[0].tolist(), "All dataframes must have the same columns in the same order."
        
        num_df = df.apply(pd.to_numeric, errors='coerce')

        if col_counts is None:
            col_counts = num_df.count()
            col_sums = num_df.sum(skipna=True)
            col_sumsq = (num_df**2).sum(skipna=True)
        else:
            col_counts += num_df.count()
            col_sums += num_df.sum(skipna=True)
            col_sumsq += (num_df**2).sum(skipna=True)

    # Variance formula: Var = (sum_sq / n) - (mean)^2
    mean = col_sums / col_counts
    variance = (col_sumsq / col_counts) - (mean**2)

    # To avoid negative variance due to floating point errors, clip at 0
    variance = variance.clip(lower=0)

    std_dev = np.sqrt(variance)
    return std_dev

if __name__ == '__main__':
    main()