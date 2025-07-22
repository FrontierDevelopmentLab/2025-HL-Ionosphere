import argparse
import datetime
import os
import sys
import pprint
from tqdm import tqdm
from glob import glob
import gzip
import xarray as xr
import numpy as np
import webdataset as wds

def jpld_filename_to_date(file_name):
    """Extracts the date from a JPLD GIM filename."""
    file_name = os.path.basename(file_name)
    if not file_name.startswith('jpld') or not file_name.endswith('.nc.gz'):
        raise ValueError(f"Invalid JPLD filename format: {file_name}")
    parts = file_name.split('.')
    if len(parts) < 3:
        raise ValueError(f"Invalid JPLD filename format: {file_name}")
    day_of_year = int(parts[0][4:7])
    year = int(parts[1][:2]) + 2000
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    return date

def write_tar_shard(records, target_dir, shard_number):
    """
    Takes a list of records and writes them to a numbered .tar shard.
    """
    if not records:
        return

    # Define the output tarball filename, e.g., 'jpld-gim-001.tar'
    # Using 3-digit padding for the shard number.
    tar_filename = os.path.join(target_dir, f"jpld-{shard_number:03d}.tar")
    
    # Use webdataset's TarWriter to create the archive
    with wds.TarWriter(tar_filename) as sink:
        for record in records:
            timestamp = record['timestamp']
            tecmap_array = record['tecmap']
            
            # Define the internal path and key for the sample
            # The key will be YYYY/MM/DD/HHMM.tecmap
            key = timestamp.strftime("%Y/%m/%d/%H%M.tecmap")
            
            # Create the sample dictionary. Webdataset will automatically
            # create a .npy file from the 'npy' key.
            sample = {
                "__key__": key,
                "npy": tecmap_array.astype(np.float32)
            }
            
            # Write the sample to the tar file
            sink.write(sample)

def main():
    description = 'NASA Heliolab 2025 - JPLD GIM data converter from raw to numbered WebDataset shards'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dir', type=str, help='Input directory with raw .nc.gz files', required=True)
    parser.add_argument('--target_dir', type=str, help='Target directory for output .tar files', required=True)
    
    args = parser.parse_args()
    print(description)
    start_time = datetime.datetime.now()
    
    os.makedirs(args.target_dir, exist_ok=True)
    
    file_names = glob(os.path.join(args.input_dir, '**', '*.nc.gz'), recursive=True)

    file_names_prefixed_by_extension = [''.join(f.split(".")[::-1]) for f in file_names]
    # sort files by file_names_prefixed_by_extension
    file_names = [x for _, x in sorted(zip(file_names_prefixed_by_extension, file_names))]

    if not file_names:
        print('No files found in the input directory.')
        return
        
    print(f'Found {len(file_names):,} files. Starting batch processing to create numbered tar shards...')

    # --- Batch Processing Logic for WebDataset ---
    monthly_records = []
    current_year, current_month = None, None
    total_records = 0
    shard_number = 1 # Start shard numbering at 1
    first_timestamp = None
    last_timestamp = None

    for file_name in tqdm(file_names, desc='Processing files'):
        try:
            file_date = jpld_filename_to_date(file_name)
            
            if current_year is None:
                current_year, current_month = file_date.year, file_date.month

            # If the file's month is different from the current batch, write the completed batch
            if file_date.year != current_year or file_date.month != current_month:
                tqdm.write(f"Writing {len(monthly_records)} records for {current_year}-{current_month:02d} to shard jpld-{shard_number:03d}.tar...")
                write_tar_shard(monthly_records, args.target_dir, shard_number)
                shard_number += 1
                total_records += len(monthly_records)
                
                # Reset for the new month
                monthly_records = []
                current_year, current_month = file_date.year, file_date.month

            # Read data from the source file
            with gzip.open(file_name, 'rb') as f:
                ds = xr.open_dataset(f, engine='h5netcdf')
                tecmaps = ds['tecmap'].values
                times = ds['time'].values

            j2000_epoch = datetime.datetime(2000, 1, 1, 12, 0, 0)
            for i in range(len(times)):
                timestamp_seconds = int(times[i].astype('datetime64[s]').astype(int))
                tecmap_date = j2000_epoch + datetime.timedelta(seconds=timestamp_seconds)
                
                # Append the record (timestamp and numpy array) to the current monthly batch
                record = {
                    'timestamp': tecmap_date,
                    'tecmap': tecmaps[i]
                }
                monthly_records.append(record)

                if first_timestamp is None:
                    first_timestamp = tecmap_date
                last_timestamp = tecmap_date
                
        except Exception as e:
            tqdm.write(f"Skipping file {os.path.basename(file_name)} due to error: {e}")
            continue

    # Write the final batch of records after the loop finishes
    if monthly_records:
        tqdm.write(f"Writing final batch of {len(monthly_records)} records to shard jpld-{shard_number:03d}.tar...")
        write_tar_shard(monthly_records, args.target_dir, shard_number)
        total_records += len(monthly_records)

    print('\nWebDataset shards created successfully.')
    print(f'Total records processed: {total_records:,}')
    print(f'Total shards created: {shard_number}')
    print(f'First timestamp: {first_timestamp}')
    print(f'Last timestamp: {last_timestamp}')
    total_size = sum(os.path.getsize(f) for f in glob(os.path.join(args.target_dir, '*.tar'))) / (1024 ** 3)
    print(f'Total size on disk: {total_size:.2f} GiB')
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == '__main__':
    main()
