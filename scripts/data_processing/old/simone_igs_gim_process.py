'''
This file processes the downloaded daily vTEC global ionosphere maps (GIMs) from International GNSS Service (IGS): https://cddis.nasa.gov/archive/gnss/products/ionex/ 

We choose the uqrg files from IGS, since they are 15 min resolution and Roma et al. 2018 (https://link.springer.com/article/10.1007/s00190-017-1088-9) showed they are the most accurate.

Original file format: NNNNDDD0.YYi
NNNN: name of the organization
DDD: day of the year (001-365)
0: flag
YY: year (2000 + YY)
i: some index, unsure

Safe file format: YYYY/mm/dd/vTEC_YYYYmmdd.pkl
'''

import os, re, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def parse_ionex_to_daily_df(file_path):
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

    return pd.DataFrame({"map": maps}, index=pd.DatetimeIndex(epochs))

# ────────────────────────────────────────────────────────────────

from pathlib import Path

input_folder  = "ionosphere_central/vTEC_data" #change this to your input folder

for fname in os.listdir(input_folder):
    if not fname.endswith("i"):
        continue
    fpath = os.path.join(input_folder, fname)
    try:
        #print(f" parsing {fname}")
        df = parse_ionex_to_daily_df(fpath)
        df = df.iloc[:-1] # remove the last row which is the starting point of the day: to verify
        #print(" epochs parsed:", len(df))                       # DEBUG 1

        if df.empty:
            #print("DataFrame empty, skip.")
            continue

        day_str  = df.index[0].strftime("%Y%m%d")
        output_folder = f"ionosphere_central/daily_vtec_data/{day_str[:4]}/{day_str[4:6]}/{day_str[6:]}" #change first part to your output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        #print("output folder:", output_folder)                 # DEBUG 2
        outfile  = Path(output_folder, f"vTEC_{day_str}.pkl")
        #print("saving to:", outfile)                            # DEBUG 3

        df.to_pickle(outfile, protocol=4)
        #print("written\n")                                   # DEBUG 4

    except Exception as exc:
        print(" ERROR:", exc)
