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
import pyarrow.dataset as ds

JPLDGIM_mean = 14.878721237182617
JPLDGIM_std = 14.894197463989258

# More robust implementation that handles variable row group sizes
class JPLDGIMDataset(Dataset):
    def __init__(self, parquet_file, date_start=None, date_end=None, normalize=True):
        self.parquet_file = parquet_file
        self.normalize = normalize
        
        # Open parquet file with PyArrow        
        self.parquet_table = pq.ParquetFile(parquet_file)

        # Build row group index mapping
        self._build_row_group_index()
        
        # # Read only the date column to get metadata
        # date_table = self.parquet_table.read(columns=['date']) 
        # all_dates = date_table['date'].to_pylist() # This stip seems very slow
        
        # # Convert to datetime objects for filtering
        # all_datetimes = [pd.to_datetime(date).to_pydatetime() for date in all_dates]

        # Faster date read using pyarrow.dataset
        dataset = ds.dataset(parquet_file, format="parquet")
        df = dataset.to_table(columns=['date']).to_pandas()
        df['date'] = pd.to_datetime(df['date'])

        all_dates = df['date'].tolist()
        all_datetimes = all_dates # NOTE: the type of this <class 'pandas._libs.tslibs.timestamps.Timestamp'>
  
        # Determine date range
        available_start = min(all_datetimes)
        available_end = max(all_datetimes) + datetime.timedelta(minutes=15)  # Add 15 minutes to include the last sample
        
        # Set date range (use available range if not specified)
        self.date_start = date_start if date_start is not None else available_start
        self.date_end = date_end if date_end is not None else available_end
        
        # Validate date range 
        print(type(self.date_start), type(available_start))
        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < available_start or self.date_end > available_end:
            raise ValueError(f"Specified date range ({self.date_start.strftime('%Y-%m-%d')} to {self.date_end.strftime('%Y-%m-%d')}) "
                           f"is outside the available data range ({available_start.strftime('%Y-%m-%d')} to {available_end.strftime('%Y-%m-%d')}).")
        
        # Filter data by date range
        self._filter_by_date_range(all_dates, all_datetimes)
        
        print(f"Dataset date range: {self.date_start.strftime('%Y-%m-%d')} to {self.date_end.strftime('%Y-%m-%d')}")
        print(f"Number of samples: {len(self.dates):,}")

    def get_date_range(self):
        return self.date_start, self.date_end
    
    def set_date_range(self, date_start, date_end):
        self.date_start, self.date_end = date_start, date_end

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
