import argparse
import datetime
import os
import sys
import time
from glob import glob
import pprint
import pandas as pd
from tqdm import tqdm

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
    
    for file in tqdm(file_names, desc="forward filling yearly files"):
        df = pd.read_csv(file, compression="gzip").ffill() # remove nans by replacing nan values with value at previous timestamp (5 minute cadence)
        target_file = os.path.join(args.target_dir, os.path.basename(file))
        df.to_csv(target_file, index=False, compression="gzip") # kept same format for consistency but could also be unzipped

    print(f"saved dataset to {args.target_dir}")

if __name__ == '__main__':
    main()