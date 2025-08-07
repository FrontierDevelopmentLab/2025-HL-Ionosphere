import pandas as pd
from glob import glob
import argparse
import os
import re
import numpy as np
from tqdm import tqdm

def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, merge omniweb data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dir', type=str, help='Input directory with raw files', required=True)
    parser.add_argument('--target_dir', type=str, help='Target directory for csv.gz files', required=True)

    args = parser.parse_args()

    all_file_names = glob(os.path.join(args.input_dir, '*.csv'))
    file_names = sorted([f for f in all_file_names if re.search(r'omni_5min_\d{4}_cleaned\.csv$', os.path.basename(f))]) # filtering based on if it contains a year to filter out omni_5min_full_cleaned.csv if it already exists or any other file if it exists but shouldn't be included
  
    assert np.all(np.diff([int(file.split("_")[2]) for file in file_names]) == 1), "oops, seems like theres a gap in the data / certain year files missing" # make sure we have the proper ordering in the data + that we arent missing a year

    df = pd.concat((pd.read_csv(f) for f in tqdm(file_names, "reading csvs")), ignore_index=True, sort=False)

    # Save merged result
    df.to_csv(os.path.join(args.target_dir, "omni_5min_full_cleaned.csv"), index=False)




if __name__ == "__main__":
    main()