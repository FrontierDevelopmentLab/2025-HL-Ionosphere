import numpy as np
import torch
from skyfield.api import load, wgs84
from datetime import datetime, timedelta
import tarfile
import io
import os

# --- User Params ---
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 1, 1)
save_dir = "./ionosphere-data/sunmoonearth/"
frame_step = timedelta(hours=1)

# --- Setup epochs ---
all_epochs = []
dt = start_date
while dt <= end_date:
    all_epochs.append(dt)
    dt += frame_step

# --- Skyfield Setup ---
ts = load.timescale()
eph = load('de421.bsp')
earth, moon, sun = eph['earth'], eph['moon'], eph['sun']

def get_subsolar_lonlat(dt):
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    subsolar = wgs84.subpoint(earth.at(t).observe(sun))
    lon = (subsolar.longitude.degrees + 180) % 360 - 180
    lat = subsolar.latitude.degrees
    return lon, lat

def get_sublunar_lonlat(dt):
    t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    sublunar = wgs84.subpoint(earth.at(t).observe(moon))
    lon = (sublunar.longitude.degrees + 180) % 360 - 180
    lat = sublunar.latitude.degrees
    return lon, lat

lats = np.linspace(-90, 90, 181)
lons = np.linspace(-180, 180, 361)
lat_len, lon_len = len(lats), len(lons)
lon_grid, lat_grid = np.meshgrid(lons, lats)

def add_to_tar(tar, arr, meta, idx):
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, arr=arr)
    npz_buf.seek(0)
    npz_name = f"{idx:05d}.npz"
    ti = tarfile.TarInfo(npz_name)
    ti.size = len(npz_buf.getbuffer())
    tar.addfile(ti, npz_buf)
    # meta
    meta_str = "\n".join(f"{k}: {v}" for k, v in meta.items())
    meta_bytes = meta_str.encode("utf-8")
    ti = tarfile.TarInfo(f"{idx:05d}.txt")
    ti.size = len(meta_bytes)
    tar.addfile(ti, io.BytesIO(meta_bytes))

# --- Main loop: write each time step to correct month's tar ---
current_tar = None
current_month = None
current_year = None
monthly_idx = 0

os.makedirs(save_dir, exist_ok=True)

for global_idx, dt in enumerate(all_epochs):
    year, month = dt.year, dt.month
    if (current_month != month) or (current_year != year):
        # Close previous tar
        if current_tar is not None:
            current_tar.close()
            print(f"Saved {tar_name} with {monthly_idx} samples.")
        # Open new tar for the month
        tar_name = os.path.join(save_dir, f"{year}_{month:02d}_SEMparameters.tar")
        current_tar = tarfile.open(tar_name, "w")
        current_month = month
        current_year = year
        monthly_idx = 0
        print(f"\nStarted new month: {year}-{month:02d}")

    # Calculate data
    solar_lon, solar_lat = get_subsolar_lonlat(dt)
    lunar_lon, lunar_lat = get_sublunar_lonlat(dt)
    ch0 = np.sqrt((lon_grid - solar_lon)**2 + (lat_grid - solar_lat)**2)
    ch1 = np.sqrt((lon_grid - lunar_lon)**2 + (lat_grid - lunar_lat)**2)
    arr = np.stack([ch0, ch1], axis=0).astype(np.float32)
    meta = {
        "datetime": dt.isoformat(),
        "solar_lon": solar_lon,
        "solar_lat": solar_lat,
        "lunar_lon": lunar_lon,
        "lunar_lat": lunar_lat,
    }
    add_to_tar(current_tar, arr, meta, monthly_idx)
    monthly_idx += 1

    if monthly_idx % 100 == 0:
        print(f"  {monthly_idx} written for {year}-{month:02d}")

# Close last tar
if current_tar is not None:
    current_tar.close()
    print(f"Saved {tar_name} with {monthly_idx} samples.")

print("Done.")
