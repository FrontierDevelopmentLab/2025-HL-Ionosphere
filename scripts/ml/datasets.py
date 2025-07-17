import torch
from torch.utils.data import Dataset
from glob import glob
import os
import datetime
import gzip
import xarray as xr
from functools import lru_cache
import pandas as pd
import numpy as np
import pyarrow.parquet as pq


JPLDGIM_mean = 14.878721237182617
JPLDGIM_std = 14.894197463989258



# JPLD GIM Dataset working with raw NetCDF files
# Note: seems to be slow to do all data processing on the fly
# Preferred to use the Parquet dataset for faster access
# class JPLDGIMDatasetOld(Dataset):
#     def __init__(self, data_dir, date_start=None, date_end=None, normalize=True):
#         print('JPLD GIM Dataset')
#         self.data_dir = data_dir
#         self.normalize = normalize
#         dates_avialable = self.find_date_range(data_dir)
#         if dates_avialable is None:
#             raise ValueError("No data found in the specified directory.")
#         date_start_on_disk, date_end_on_disk = dates_avialable

#         self.date_start = date_start_on_disk if date_start is None else date_start
#         self.date_end = date_end_on_disk if date_end is None else date_end
#         if self.date_start > self.date_end:
#             raise ValueError("Start date cannot be after end date.")
#         if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
#             raise ValueError("Specified date range is outside the available data range.")

#         self.num_days = (self.date_end - self.date_start).days + 1
#         cadence = 15 # minutes
#         self.num_samples = int(self.num_days * (24 * 60 / cadence))
#         print('Number of days in dataset   : {:,}'.format(self.num_days))
#         print('Number of samples in dataset: {:,}'.format(self.num_samples))
#         # size on disk
#         size_on_disk = sum(os.path.getsize(f) for f in glob(f"{data_dir}/*/*.nc.gz"))
#         print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))

#     @staticmethod
#     def find_date_range(directory):
#         # print("Checking date range of data in directory: {}".format(directory))
#         days = sorted(glob(f"{directory}/*/*.nc.gz"))
#         if len(days) == 0:
#             return None

#         # print(directory)
#         # print(days[0])
#         # print(days[-1])
#         # example output:
#         # /disk2-ssd-8tb/data/2025-hl-ionosphere/gim_jpld_20100513-20240731/2010/jpld1330.10i.nc.gz
#         # /disk2-ssd-8tb/data/2025-hl-ionosphere/gim_jpld_20100513-20240731/2023/jpld1470.23i.nc.gz

#         days = [d.replace(directory, '') for d in days]
#         date_start = datetime.datetime.strptime(days[0].split('.')[0], "/%Y/jpld%j0")
#         date_end = datetime.datetime.strptime(days[-1].split('.')[0], "/%Y/jpld%j0")

#         print("Directory  : {}".format(directory))
#         print("Start date : {}".format(date_start.strftime('%Y-%m-%d')))
#         print("End date   : {}".format(date_end.strftime('%Y-%m-%d')))

#         return date_start, date_end
    
#     @staticmethod
#     def normalize(data):
#         return (data - JPLDGIM_mean) / JPLDGIM_std
#         # return torch.log1p(data)

#     @staticmethod
#     def unnormalize(data):
#         return data * JPLDGIM_std + JPLDGIM_mean

#     def __len__(self):
#         return self.num_samples
    
#     @lru_cache(maxsize=4096) # number of days to cache in memory, roughly 3 MiB per day
#     def _get_day_data(self, date):
#         file_name = f"jpld{date:%j}0.{date:%y}i.nc.gz"
#         file_path = os.path.join(self.data_dir, f"{date:%Y}", file_name)
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
        
#         with gzip.open(file_path, 'rb') as f:
#             ds = xr.open_dataset(f, engine='h5netcdf')
            
#             # Assuming 'tecmap' is the variable of interest
#             data = ds['tecmap'].values
#             # data_tensor shape torch.Size([96, 180, 360]) where 96 is nepochs, 180 is nlats, and 360 is nlons
#             data_tensor = torch.tensor(data, dtype=torch.float32)
#             if self.normalize:
#                 data_tensor = JPLDGIMDatasetOld.normalize(data_tensor)

#             return data_tensor

#     def __getitem__(self, index):
#         samples_per_day = 24 * 60 // 15  # 15-minute cadence
#         if isinstance(index, datetime.datetime):
#             date = index
#         elif isinstance(index, int):
#             if index < 0 or index >= self.num_samples:
#                 raise IndexError("Index out of range for the dataset.")
#             days = index // samples_per_day
#             minutes = (index % samples_per_day) * 15
#             date = self.date_start + datetime.timedelta(days=days, minutes=minutes)
#         else:
#             raise TypeError("Index must be an integer or a datetime object.")

#         data = self._get_day_data(date)
#         time_index = (index % samples_per_day)  # Get the time index within the
#         data = data[time_index, :, :]  # Select the specific time slice
#         data = data.unsqueeze(0)  # Add a channel dimension

#         return data, date.isoformat()  # Return the data and the date as a string


# Parquet dataset for JPLD GIM
# class JPLDGIMDataset(Dataset):
#     def __init__(self, parquet_file, normalize=True):
#         self.data = pd.read_parquet(parquet_file)
#         self.normalize = normalize
    
#     @staticmethod
#     def normalize(data):
#         return (data - JPLDGIM_mean) / JPLDGIM_std
#         # return torch.log1p(data)

#     @staticmethod
#     def unnormalize(data):
#         return data * JPLDGIM_std + JPLDGIM_mean

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         if isinstance(index, datetime.datetime):
#             index = self.data[self.data['date'] == index.isoformat()].index[0]
#         elif isinstance(index, int):
#             if index < 0 or index >= len(self.data):
#                 raise IndexError("Index out of range for the dataset.")
#         else:
#             raise TypeError("Index must be an integer or a datetime object.")
        
#         row = self.data.iloc[index]
#         tecmap = np.array(row['tecmap'].tolist())
#         if self.normalize:
#             tecmap = JPLDGIMDataset.normalize(tecmap)
#         tecmap = torch.tensor(tecmap, dtype=torch.float32)
#         tecmap = tecmap.unsqueeze(0)  # Add a channel dimension
#         date = row['date']
#         return tecmap, date.isoformat()

# Alternative implementation using PyArrow for better memory efficiency
# class JPLDGIMDataset(Dataset):
#     def __init__(self, parquet_file, normalize=True):
#         self.parquet_file = parquet_file
#         self.normalize = normalize
        
#         # Open parquet file with PyArrow
#         self.parquet_table = pq.ParquetFile(parquet_file)
        
#         # Read only the date column to get metadata
#         date_table = self.parquet_table.read(columns=['date'])
#         self.dates = date_table['date'].to_pylist()
#         self._length = len(self.dates)
        
#         # Create a mapping from datetime to row index
#         self._date_to_index = {
#             pd.to_datetime(date).to_pydatetime(): idx 
#             for idx, date in enumerate(self.dates)
#         }
    
#     @staticmethod
#     def normalize(data):
#         return (data - JPLDGIM_mean) / JPLDGIM_std

#     @staticmethod
#     def unnormalize(data):
#         return data * JPLDGIM_std + JPLDGIM_mean

#     def __len__(self):
#         return self._length

#     def __getitem__(self, index):
#         if isinstance(index, datetime.datetime):
#             if index not in self._date_to_index:
#                 raise KeyError(f"Date {index.isoformat()} not found in dataset")
#             row_index = self._date_to_index[index]
#         elif isinstance(index, int):
#             if index < 0 or index >= self._length:
#                 raise IndexError("Index out of range for the dataset.")
#             row_index = index
#         else:
#             raise TypeError("Index must be an integer or a datetime object.")
        
#         # Read specific row using PyArrow
#         row_group = row_index // self.parquet_table.metadata.row_group(0).num_rows
#         local_index = row_index % self.parquet_table.metadata.row_group(0).num_rows
        
#         # Read the row group containing our target row
#         table = self.parquet_table.read_row_group(row_group)
#         row = table.slice(local_index, 1).to_pandas().iloc[0]
        
#         tecmap = np.array(row['tecmap'])
        
#         if self.normalize:
#             tecmap = JPLDGIMDataset.normalize(tecmap)
            
#         tecmap = torch.tensor(tecmap, dtype=torch.float32)
#         tecmap = tecmap.unsqueeze(0)  # Add a channel dimension
#         date = row['date']
        
#         return tecmap, date.isoformat() if hasattr(date, 'isoformat') else str(date)

# More robust implementation that handles variable row group sizes
class JPLDGIMDataset(Dataset):
    def __init__(self, parquet_file, date_start=None, date_end=None, normalize=True):
        self.parquet_file = parquet_file
        self.normalize = normalize
        
        # Open parquet file with PyArrow
        self.parquet_table = pq.ParquetFile(parquet_file)
        
        # Build row group index mapping
        self._build_row_group_index()
        
        # Read only the date column to get metadata
        date_table = self.parquet_table.read(columns=['date'])
        all_dates = date_table['date'].to_pylist()
        
        # Convert to datetime objects for filtering
        all_datetimes = [pd.to_datetime(date).to_pydatetime() for date in all_dates]
        
        # Determine date range
        available_start = min(all_datetimes)
        available_end = max(all_datetimes) + datetime.timedelta(minutes=15)  # Add 15 minutes to include the last sample
        
        # Set date range (use available range if not specified)
        self.date_start = date_start if date_start is not None else available_start
        self.date_end = date_end if date_end is not None else available_end
        
        # Validate date range
        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < available_start or self.date_end > available_end:
            raise ValueError(f"Specified date range ({self.date_start.strftime('%Y-%m-%d')} to {self.date_end.strftime('%Y-%m-%d')}) "
                           f"is outside the available data range ({available_start.strftime('%Y-%m-%d')} to {available_end.strftime('%Y-%m-%d')}).")
        
        # Filter data by date range
        self._filter_by_date_range(all_dates, all_datetimes)
        
        print(f"Dataset date range: {self.date_start.strftime('%Y-%m-%d')} to {self.date_end.strftime('%Y-%m-%d')}")
        print(f"Number of samples: {len(self.dates):,}")
    
    def _filter_by_date_range(self, all_dates, all_datetimes):
        """Filter dates and create mappings based on the specified date range"""
        # Find indices within the date range
        valid_indices = []
        filtered_dates = []
        filtered_datetimes = []
        
        for idx, dt in enumerate(all_datetimes):
            if self.date_start <= dt <= self.date_end:
                valid_indices.append(idx)
                filtered_dates.append(all_dates[idx])
                filtered_datetimes.append(dt)
        
        # Store filtered data
        self.dates = filtered_dates
        self._length = len(self.dates)
        
        # Create mapping from datetime to filtered index
        self._date_to_index = {
            dt: idx for idx, dt in enumerate(filtered_datetimes)
        }
        
        # Create mapping from filtered index to original index
        self._filtered_to_original = {
            filtered_idx: original_idx 
            for filtered_idx, original_idx in enumerate(valid_indices)
        }
    
    def _build_row_group_index(self):
        """Build a mapping from global row index to (row_group, local_index)"""
        self.row_group_mapping = []
        cumulative_rows = 0
        
        for rg_idx in range(self.parquet_table.num_row_groups):
            rg_metadata = self.parquet_table.metadata.row_group(rg_idx)
            num_rows = rg_metadata.num_rows
            
            for local_idx in range(num_rows):
                self.row_group_mapping.append((rg_idx, local_idx))
            
            cumulative_rows += num_rows
    
    @staticmethod
    def normalize(data):
        return (data - JPLDGIM_mean) / JPLDGIM_std

    @staticmethod
    def unnormalize(data):
        return data * JPLDGIM_std + JPLDGIM_mean

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            if index not in self._date_to_index:
                raise KeyError(f"Date {index.isoformat()} not found in dataset")
            filtered_index = self._date_to_index[index]
        elif isinstance(index, int):
            if index < 0 or index >= self._length:
                raise IndexError("Index out of range for the dataset.")
            filtered_index = index
        else:
            raise TypeError("Index must be an integer or a datetime object.")
        
        # Map filtered index to original index
        original_index = self._filtered_to_original[filtered_index]
        
        # Get row group and local index for original data
        row_group_idx, local_index = self.row_group_mapping[original_index]
        
        # Read the specific row group and slice the target row
        table = self.parquet_table.read_row_group(row_group_idx)
        row = table.slice(local_index, 1).to_pandas().iloc[0]
        
        tecmap = torch.from_numpy(np.array(row['tecmap'].tolist(), dtype=np.float32))
        
        if self.normalize:
            tecmap = JPLDGIMDataset.normalize(tecmap)
            
        tecmap = tecmap.unsqueeze(0)  # Add a channel dimension
        date = row['date']
        
        return tecmap, date.isoformat() if hasattr(date, 'isoformat') else str(date)