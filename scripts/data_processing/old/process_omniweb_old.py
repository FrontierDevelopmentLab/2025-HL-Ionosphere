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
# NOTE: this wont be used in the final processed data 
# NOTE: May be better to fill withmean value, as certain parameters have long gaps of missing data which forward filling will likely cause a significant bias
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

    file_names = sorted(glob(os.path.join(args.input_dir, '*.csv')))
    years = [int(file.split("_")[2]) for file in file_names] 
    assert sorted(years) == years # make sure files read in order since to fill based on last values of previous file, the previous fle needs to already have been filled
    if len(file_names) == 0:
        print('No files found in the input directory.')
        return
    
    print('Found {:,} files in the input directory.'.format(len(file_names)))
    dfs = []

    for file in tqdm(file_names, desc="reading in dataset"):
        df = pd.read_csv(file)
        dfs.append(df)
    
    mean_df = compute_mean(dfs)
    std_df = compute_std(dfs)

    # save mean and stddev to new csv 
    column_names = dfs[0].columns.tolist() # NOTE: any reference to headers will need to be updated once the cleaned files headers are fixed (ie referring to dfs[0].columns instead)
    omni_stats = pd.DataFrame([mean_df.values, std_df.values], columns=column_names)

    stats_path = os.path.join(args.target_dir, "dataset_stats","omni_stats.csv")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    omni_stats.to_csv(stats_path, index=False)

    ffill_all(dfs=dfs, filenames=file_names, args=args)

    print(f"saved dataset to {args.target_dir}")
    print(f"saved statistics to {stats_path}")

def compute_mean(dfs):
    col_sums = None
    col_counts = None
    for i, df in tqdm(enumerate(dfs), desc="computing the dataset mean", total=len(dfs)):
        if i == 0:
            column_names = df.columns.tolist()
            print(column_names)
        assert column_names == df.columns.tolist(), "All dataframes must have the same columns in the same order."
        num_df = df.apply(pd.to_numeric, errors='coerce')
        if col_sums is None:
            col_sums = num_df.sum(skipna=True)
            col_counts = num_df.count()
        else:
            col_sums += num_df.sum()
            col_counts += num_df.count()
    mean_df = col_sums / col_counts
    return mean_df

def ffill_all(dfs, filenames, args):
    dfs = sorted(dfs, key=lambda x: x.iloc[0]["Year"]) 
    dfs_filled = {}
    earliest_year = dfs[0].iloc[0]["Year"]
    for df, file in tqdm(zip(dfs, filenames), desc="removing nan with forward filling", total=len(dfs)):
        year = df.iloc[0]["Year"]
        base_name = os.path.basename(file)
        print(base_name, file)
        assert int(base_name.split("_")[2]) == year # make sure there isnt a mismatch between filename and df (there shouldnt be)
        target_file = os.path.join(args.target_dir, base_name)        

        # if year == earliest_year:
            # assert sum(df.iloc[0].isna()) == 0 # cant forward fill nans if they exist in the first entry of the dataset 
            # RMS_PFN is always NaN seemingly (including in 2006-01-01 00:00:00) but dont believe were using it
        if year != earliest_year:
            prev_year = year - 1
            df_prev = dfs_filled[prev_year]
            df = ffill_first_entry(df, df_prev)

        df_ffill = df.ffill()# remove nans by replacing nan values with value at previous timestamp (5 minute cadence)
        dfs_filled[year] = df_ffill
        df_ffill.to_csv(target_file, index=False)


def ffill_first_entry(df, df_prev):
    df_copy = df.copy() 
    year = df.iloc[0]["Year"]
    prev_year = year - 1
    nan_cols = df.columns[df.iloc[0].isna()]
    fill_vals = df_prev.iloc[-1][nan_cols]
    df_copy.loc[df_copy.index[0], nan_cols] = fill_vals.values
    return df_copy

def compute_std(dfs):
    col_counts = None
    col_sums = None
    col_sumsq = None  # sum of squares

    for i, df in tqdm(enumerate(dfs), desc="computing the dataset std", total=len(dfs)):
        if i == 0:
            column_names = df.columns.tolist()
            print(column_names)
        assert column_names == df.columns.tolist(), "All dataframes must have the same columns in the same order."
        
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
    # python process_omniweb.py --input_dir /mnt/ionosphere-data/omniweb/cleaned/ --target_dir /mnt/ionosphere-data/omniweb/processed/