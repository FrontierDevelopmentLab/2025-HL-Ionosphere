'''
This file processes the downloaded daily TEC global ionosphere maps (GIMs) from CDDIS: https://cddis.nasa.gov/archive/gnss/products/ionex/ 

The code to download this data is contained in data_gim_cddis_daily_download.py
'''
import argparse
import os
import tarfile
import glob
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


def parse_ionex_to_dataframe(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header_end = [i for i, l in enumerate(lines) if 'END OF HEADER' in l][0]
    header = lines[:header_end]

    for line in header:
        if 'LAT1 / LAT2 / DLAT' in line:
            lat1, lat2, dlat = map(float, line.split()[:3])
        if 'LON1 / LON2 / DLON' in line:
            lon1, lon2, dlon = map(float, line.split()[:3])

    lats = np.arange(lat1, lat2 - 0.1, -abs(dlat))  # top-to-bottom
    lons = np.arange(lon1, lon2 + 0.1, dlon)

    maps = []
    epochs = []

    i = header_end + 1
    while i < len(lines):
        if 'START OF TEC MAP' in lines[i]:
            epoch_line = lines[i+1]
            y, mo, d, h, mi, s = map(int, epoch_line[:36].split())
            if h == 24:

                h = 0
                base_date = datetime(y, mo, d) + timedelta(days=1)
                epoch = datetime(base_date.year, base_date.month, base_date.day, h, mi, s)
            else:
                epoch = datetime(y, mo, d, h, mi, s)
            epochs.append(epoch)

            grid = []
            i += 2
            while i < len(lines) and ('START OF TEC MAP' not in lines[i]) and ('END OF FILE' not in lines[i]):
                if 'LAT/LON1/LON2/DLON/H' in lines[i]:
                    row = []
                    i += 1
                    while i < len(lines):
                        line_strip = lines[i].strip()
                        if not line_strip or not re.match(r'^[-\d\s]+$', line_strip) or 'END OF TEC MAP' in line_strip:
                            break
                        vals = [float(v) if v != '9999' else np.nan for v in line_strip.split()]
                        row.extend(vals)
                        i += 1
                    if len(row) != len(lons):
                        row = (row + [np.nan]*len(lons))[:len(lons)]
                    grid.append(row)
                else:
                    i += 1
            while len(grid) < len(lats):
                grid.append([np.nan]*len(lons))
            grid = grid[:len(lats)]
            maps.append(np.array(grid))
        else:
            i += 1

    maps = np.array(maps)
    data = {}
    for lat_idx, lat in enumerate(lats):
        for lon_idx, lon in enumerate(lons):
            data[(lat, lon)] = maps[:, lat_idx, lon_idx]
    df = pd.DataFrame(data, index=epochs)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['lat', 'lon'])

    return df, maps, epochs



def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Process GIM vTEC data and create tar files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input folder containing GIM vTEC data files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output folder to save processed files and tar archives.')
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    current_year = None
    
    # Sort by year first then day, since extension contains the year and the filename the DoY,
    # prefix the extension on filename and argsort the files by the prefixed strings
    files = os.listdir(args.input_dir)
    extension_prefix_files = [''.join(f.split(".")[::-1]) for f in files]
    
    # sort files by extension_prefix_files
    files = [x for _,x in sorted(zip(extension_prefix_files,files))]

    # print(files[:100])
    # a = 1/0
    for filename in files:
        file_path = os.path.join(args.input_dir, filename)
        # Extract year from filename
        match = re.search(r'\.(\d{2})i$', filename)
        if match:
            year = int(match.group(1)) + 2000
        else:
            print(f"Could not extract year from filename: {filename}")
            continue
        if current_year is None or current_year != year:
            if current_year is not None:
                # Create tar file for the previous year
                tar_filename = os.path.join(args.output_dir, f"GIM_vTEC_{current_year}.tar.gz")
                print(f"Creating tar file for year {current_year}...")
                with tarfile.open(tar_filename, "w:gz") as tar:
                    for npy_file in tqdm(glob.glob(os.path.join(args.output_dir, f"GIM_vTEC_{current_year}*.npy")), desc=f"Adding files for {current_year}"):
                        tar.add(npy_file, arcname=os.path.basename(npy_file))
                print(f"Created tar file: {tar_filename}")
                # remove individual .npy files after archiving
                print(f"Removing individual .npy files for year {current_year}...")
                for npy_file in glob.glob(os.path.join(args.output_dir, f"GIM_vTEC_{current_year}*.npy")):
                    os.remove(npy_file)
            current_year = year
        try:
            print(f"Year: {current_year}, Day file: {filename}")
            df, maps, epochs = parse_ionex_to_dataframe(file_path)

            # Save individual 2D TEC maps as .npy
            for i, epoch in enumerate(epochs):
                ts = epoch.strftime("%Y%m%d_%H%M")
                npy_outfile = os.path.join(args.output_dir, f"GIM_vTEC_{ts}.npy")
                np.save(npy_outfile, maps[i])
            # print(f"Saved {len(epochs)} TEC maps from {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    
    
    

if __name__ == "__main__":
    main()