import argparse
import datetime
import os
import sys
import time
from glob import glob
import pprint
import pandas as pd

'''
Process celestrak and SET solar index data files: currently if there are missing measurements, the value is 0.0.
This script finds these values, and replaces them with the nearest available value (in time).
This is done to ensure that the datasets are continuous and can be used for training models.
'''

def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, process celestrak and SET solar index data'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_file', type=str, help='Input file with raw data', required=True)
    parser.add_argument('--target_file', type=str, help='Target file for processed SET solar index data', required=True)
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)
    
    # Data processing
    df = pd.read_csv(args.input_file)
    
    # Find 0.0 values and replace them with NaN, then forward fill
    df.replace(0.0, pd.NA, inplace=True)
    df.ffill(inplace=True)  # forward fill NaN values

    # Save
    df.to_csv(args.target_file, index=False) # kept same format for consistency but could also be zipped

    print(f"saved dataset to {args.target_file}")

if __name__ == '__main__':
    main()