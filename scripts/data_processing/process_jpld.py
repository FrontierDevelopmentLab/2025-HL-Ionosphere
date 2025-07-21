import argparse
import datetime
import os
import sys
import pprint
from tqdm import tqdm
from glob import glob
import h5py
import gzip
import xarray as xr
import time
import pandas as pd
import numpy as np

# File Naming Convention	YYYY/SSSSDDD#.YYi.nc.gz
# where:
# YYYY	4-digit year
# SSSS	IGS monument name
# DDD	3-digit day of year
# #	file number for the day, typically 0
# YY	2-digit year
# .gz	gzip compressed file
#
# Sample URL:
# https://sideshow.jpl.nasa.gov/pub/iono_daily/gim_for_research/jpld/2017/jpld0010.17i.nc.gz

def jpld_date_to_filename(date):
    file_name = f"jpld{date:%j}0.{date:%y}i.nc.gz"
    return file_name

def jpld_filename_to_date(file_name):
    # Example: jpld0010.17i.nc.gz
    # Extract the day of year and year from the filename
    file_name = os.path.basename(file_name)
    if not file_name.startswith('jpld') or not file_name.endswith('.nc.gz'):
        raise ValueError(f"Invalid JPLD filename format: {file_name}")
    parts = file_name.split('.')
    if len(parts) < 3:
        raise ValueError(f"Invalid JPLD filename format: {file_name}")
    
    day_of_year = int(parts[0][4:7])  # Extract DDD from jpldDDD
    year = int(parts[1][:2]) + 2000  # Extract YY and convert to full year
    
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    return date


def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, JPLD GIM data converter raw to HDF5'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dir', type=str, help='Input directory with raw files', required=True)
    parser.add_argument('--target_dir', type=str, help='Target directory for HDF5 files', required=True)
    parser.add_argument('--num_max_files', type=int, default=None, help='Maximum number of files to process')
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    os.makedirs(args.target_dir, exist_ok=True)

    file_names = glob(os.path.join(args.input_dir, '**', '*.nc.gz'), recursive=True)

    file_names_prefixed_by_extension = [''.join(f.split(".")[::-1]) for f in file_names]
    # sort files by file_names_prefixed_by_extension
    file_names = [x for _, x in sorted(zip(file_names_prefixed_by_extension, file_names))]

    if len(file_names) == 0:
        print('No files found in the input directory.')
        return
    
    print('Found {:,} files in the input directory.'.format(len(file_names)))

    if args.num_max_files is not None:
        file_names = file_names[:args.num_max_files]
        print('Processing only the first {:,} files.'.format(args.num_max_files))

    # Create output filename with placeholder dates (will update later)
    target_file = os.path.join(args.target_dir, 'jpld_gim_temp.h5')
    
    # Initialize counters
    skipped_files = []
    total_records = 0
    first_timestamp = None
    last_timestamp = None
    
    # Create HDF5 file
    h5_file = h5py.File(target_file, 'w')
    
    try:
        # Create datasets with initial size and enable resizing
        # Use LZF compression for better training performance
        timestamps_dataset = h5_file.create_dataset(
            'timestamps', 
            (0,), 
            maxshape=(None,), 
            dtype=h5py.special_dtype(vlen=str),
            compression='lzf',  # LZF compression - much faster than gzip
            chunks=(512,)  # Larger chunks = fewer I/O operations
        )
        
        tecmaps_dataset = h5_file.create_dataset(
            'tecmaps', 
            (0, 180, 360), 
            maxshape=(None, 180, 360), 
            dtype=np.float32,
            compression='lzf',  # LZF compression - ~3x faster decompression than gzip
            chunks=(64, 180, 360)  # Larger chunks for better throughput
        )
        
        for file_name in tqdm(file_names, desc='Processing files'):
            with gzip.open(file_name, 'rb') as f:
                try:
                    ds = xr.open_dataset(f, engine='h5netcdf')            
                    data = ds['tecmap'].values  # Shape: (N, 180, 360)
                    times = ds['time'].values   # Shape: (N,)
                except Exception as e:
                    continue

                # Process entire file at once
                j2000_epoch = datetime.datetime(2000, 1, 1, 12, 0, 0)
                file_timestamps = []
                
                for time_index in range(len(times)):
                    timestamp_seconds = int(times[time_index].astype('datetime64[s]').astype(int))
                    tecmap_date = j2000_epoch + datetime.timedelta(seconds=timestamp_seconds)
                    file_timestamps.append(tecmap_date.isoformat())
                
                    # Track first and last timestamps
                    if first_timestamp is None:
                        first_timestamp = tecmap_date
                    last_timestamp = tecmap_date

                # Write entire file's data at once
                current_size = timestamps_dataset.shape[0]
                new_size = current_size + len(file_timestamps)
                
                timestamps_dataset.resize((new_size,))
                tecmaps_dataset.resize((new_size, 180, 360))
                
                timestamps_dataset[current_size:new_size] = file_timestamps
                tecmaps_dataset[current_size:new_size] = data
                
                total_records += len(file_timestamps)
            
    finally:
        # Close the HDF5 file
        h5_file.close()
    
    # Rename file with proper date range
    if first_timestamp and last_timestamp:
        date_start = first_timestamp.strftime('%Y%m%d%H%M')
        date_end = last_timestamp.strftime('%Y%m%d%H%M')
        final_target_file = os.path.join(args.target_dir, f'jpld_gim_{date_start}_{date_end}.h5')
        os.rename(target_file, final_target_file)
        target_file = final_target_file

    print(f'Processed {total_records:,} records from {len(file_names):,} files.')
    if skipped_files:
        print(f'Skipped {len(skipped_files):,} files due to errors or unexpected shapes:')
        for skipped_file in skipped_files:
            print(f' - {skipped_file}')

    print('HDF5 dataset created successfully.')
    print('Total records   : {:,}'.format(total_records))
    print('Start date      : {}'.format(first_timestamp))
    print('End date        : {}'.format(last_timestamp))
    print('Size on disk    : {:.2f} GiB'.format(os.path.getsize(target_file) / (1024 ** 3)))

    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()