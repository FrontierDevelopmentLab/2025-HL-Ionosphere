import argparse
import datetime
import os
import sys
import pprint
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gzip
import xarray as xr
import time

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
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, JPLD GIM data converter raw to Parquet'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dir', type=str, help='Input directory with raw files', required=True)
    parser.add_argument('--target_dir', type=str, help='Target directory for Parquet files', required=True)
    
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
    df = pd.DataFrame(columns=['date', 'tecmap'])

    timestamps = []
    tecmaps = []
    for file_name in tqdm(file_names, desc='Processing files'):
        date = jpld_filename_to_date(file_name)
        # print(file_name, date)
        with gzip.open(file_name, 'rb') as f:
            ds = xr.open_dataset(f, engine='h5netcdf')            
            data = ds['tecmap'].values # this will be a 3d array (time, lat, lon) with shape (96, 180, 360)
            
            for time_index in range(data.shape[0]):
                tecmap = data[time_index, :, :]
                tecmap_date = date + datetime.timedelta(minutes=time_index * 15)  # Assuming 15-minute cadence
                # print(tecmap_date, tecmap.shape)
                timestamps.append(tecmap_date)
                tecmaps.append(tecmap.tolist())

                # time.sleep(0.2)  # Add a short delay for debugging purposes


    datetime_array = pa.array(timestamps, type=pa.timestamp('s'))
    tecmap_array = pa.array(tecmaps, type=pa.list_(pa.list_(pa.float32())))

    table = pa.table({
        'date': datetime_array,
        'tecmap': tecmap_array
    })

    target_file = os.path.join(args.target_dir, 'jpld_gim_data.parquet')
    pq.write_table(table, target_file, compression='snappy')

    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()