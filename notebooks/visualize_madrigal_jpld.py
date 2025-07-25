import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime

#read a hdf5 file
import h5py

dir_path = '/mnt/ionosphere-data/madrigal_data/gps201016g.001.hdf5'

f = h5py.File(dir_path, 'r')

print("Top-level keys:", list(f.keys()))

metadata_group = f['Metadata']
print("Keys in Metadata group:", list(metadata_group.keys()))

def explore(name, obj):
    print(name)
f.visititems(explore)

# Load data
timestamps = f['Data/Array Layout/timestamps'][:]
tec = f['Data/Array Layout/2D Parameters/tec'][:]   # likely shape: (time, height or lat)
dtec = f['Data/Array Layout/2D Parameters/dtec'][:]  # likely shape: (time, height or lat)
glat = f['Data/Array Layout/gdlat'][:]    # if 1D
glon = f['Data/Array Layout/glon'][:]    # if 1D


timestamps_utc = np.array(timestamps, dtype='datetime64[s]')

print("TEC shape:", tec.shape)
print("GLAT shape:", glat.shape)
print("Timestamps shape:", timestamps.shape)

import matplotlib.pyplot as plt
# Choose a grid point
i = 70  # Example index for latitude
j = 20  # Example index for longitude

tec_time_series = tec[i, j, :]  # shape: (288,)
plt.plot(timestamps, tec_time_series)
plt.xlabel('Time index')
plt.ylabel('TEC')
plt.title(f'TEC Time Series at Grid Point ({i}, {j})')
plt.grid(True)
plt.show()

time_idx = 0
time_utc = timestamps_utc[time_idx]
#tec_snapshot = np.nansum(tec[:, :, time_idx:time_idx+5], axis = 2)  # Summing over the last dimension (time)
tec_snapshot = tec[:, :, time_idx]

#count the number of pixels with nan values 
nan_count = np.sum(np.isnan(tec_snapshot))
print(f'Number of NaN values in TEC snapshot: {nan_count}/{tec_snapshot.size}= {nan_count/tec_snapshot.size:.2%}')


fig = plt.figure(figsize=(10, 5))
plt.imshow(np.log10(tec_snapshot), origin='lower', aspect='auto', cmap='jet', vmin = 0, vmax = 1.5)
plt.colorbar(label='TEC')
plt.title(f'Madrigal GNSS GIM TEC Map on {time_utc} UTC')
plt.xlabel('Longitude index')
plt.ylabel('Latitude index')
plt.show()

#dtec_snapshot = np.nansum(dtec[:, :, time_idx:time_idx+5], axis = 2)  # Summing over the last dimension (time)
dtec_snapshot = dtec[:, :, time_idx]

fig = plt.figure(figsize=(10, 5))
plt.imshow(dtec_snapshot, origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='dTEC')
plt.title(f'Madrigal GNSS GIM dTEC Map on {time_utc} UTC')
plt.xlabel('Longitude index')
plt.ylabel('Latitude index')
plt.show()

import os
import gzip
import io
from netCDF4 import Dataset
from datetime import datetime, timedelta

#quick comparison with jpld file 
jpld_dir_path = '/mnt/ionosphere-data/jpld/raw/2020/'
jpld_file = 'jpld2900.20i.nc.gz'

compressed_path = os.path.join(jpld_dir_path, jpld_file)

with gzip.open(compressed_path, 'rb') as f:
    decompressed_data = f.read()

nc_buffer = io.BytesIO(decompressed_data)
nc = Dataset('in_memory.nc', mode='r', memory=nc_buffer.read())

#print(nc)
print("Variables in JPLD file:", nc.variables.keys())
print(nc.variables['time'])
print(nc.variables['tecmap'])  

j2000 = datetime(2000, 1, 1, 12, 0, 0)
time_seconds = nc.variables['time'][:]
time_datetimes = [j2000 + timedelta(seconds=float(t)) for t in time_seconds]
print("Time datetimes:", time_datetimes)

target_dt = datetime(2020, 10, 16, 0, 0, 0)
time_index = min(range(len(time_datetimes)), key=lambda i: abs(time_datetimes[i] - target_dt))

print("Closest available time:", time_datetimes[time_index])

import numpy as np
import matplotlib.pyplot as plt

# Read lat/lon and tec
tecmap = nc.variables['tecmap'][:]  # (96, 180, 360)

# Extract the map at the desired time index
vtec_map = tecmap[time_index, :, :]  # shape: (lat, lon)


# Plot
plt.figure(figsize=(10, 5))
plt.imshow(np.log10(vtec_map), origin='lower', aspect='auto', cmap='jet', vmin = 0, vmax = 1.5)
plt.title(f'JPLD GIM TEC Map on {time_datetimes[time_index]} UTC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='VTEC (TECU)')
plt.tight_layout()
plt.show()
