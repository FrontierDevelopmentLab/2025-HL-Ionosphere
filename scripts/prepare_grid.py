import numpy as np
from apexpy import Apex

# 1. Define your global grid
H, W = 180, 360  # adjust if needed
lats = np.linspace(87.5, -87.5, H)      # center of each 1-degree bin
lons = np.linspace(-180, 180, W, endpoint=False)
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')  # (H, W)

# Save geographic grids ONCE (same for all years)
np.save('lat_grid.npy', lat_grid)
np.save('lon_grid.npy', lon_grid)
print("Saved: lat_grid.npy, lon_grid.npy")

# 2. Loop over years for QD coordinates
for year in range(2010, 2026):   # 2025 included
    print(f"Processing year: {year}")
    apex = Apex(date=year)
    qd_lat = np.zeros((H, W))
    qd_lon = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            qd_lat[i, j], qd_lon[i, j] = apex.geo2qd(lat_grid[i, j], lon_grid[i, j], 110)
    # Save QD grids per year
    np.save(f'qd_lat_{year}.npy', qd_lat)
    np.save(f'qd_lon_{year}.npy', qd_lon)
    print(f"Saved: qd_lat_{year}.npy, qd_lon_{year}.npy")

print("Done!")