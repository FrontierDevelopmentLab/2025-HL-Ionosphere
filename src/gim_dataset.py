import torch
from glob import glob
import os
import datetime
import gzip
import xarray as xr
from functools import lru_cache

JPLDGIM_mean = 14.878721237182617
JPLDGIM_std = 14.894197463989258

# TODO: seems to be slow to do all data processing on the fly, consider working with a preprocessed dataset (netcdf -> npy done previously)
class JPLDGIMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, date_start=None, date_end=None, normalize=True):
        print('JPLD GIM Dataset')
        self.data_dir = data_dir
        self.normalize = normalize
        dates_avialable = self.find_date_range(data_dir)
        if dates_avialable is None:
            raise ValueError("No data found in the specified directory.")
        date_start_on_disk, date_end_on_disk = dates_avialable

        self.date_start = date_start_on_disk if date_start is None else date_start
        self.date_end = date_end_on_disk if date_end is None else date_end
        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
            raise ValueError("Specified date range is outside the available data range.")

        self.num_days = (self.date_end - self.date_start).days + 1
        cadence = 15 # minutes
        self.num_samples = int(self.num_days * (24 * 60 / cadence))
        print('Number of days in dataset   : {:,}'.format(self.num_days))
        print('Number of samples in dataset: {:,}'.format(self.num_samples))
        # size on disk
        size_on_disk = sum(os.path.getsize(f) for f in glob(f"{data_dir}/*/*.nc.gz"))
        print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))

    @staticmethod
    def find_date_range(directory):
        # print("Checking date range of data in directory: {}".format(directory))
        days = sorted(glob(f"{directory}/*/*.nc.gz"))
        if len(days) == 0:
            return None

        # print(directory)
        # print(days[0])
        # print(days[-1])
        # example output:
        # /disk2-ssd-8tb/data/2025-hl-ionosphere/gim_jpld_20100513-20240731/2010/jpld1330.10i.nc.gz
        # /disk2-ssd-8tb/data/2025-hl-ionosphere/gim_jpld_20100513-20240731/2023/jpld1470.23i.nc.gz

        days = [d.replace(directory, '') for d in days]
        date_start = datetime.datetime.strptime(days[0].split('.')[0], "/%Y/jpld%j0")
        date_end = datetime.datetime.strptime(days[-1].split('.')[0], "/%Y/jpld%j0")

        print("Directory  : {}".format(directory))
        print("Start date : {}".format(date_start.strftime('%Y-%m-%d')))
        print("End date   : {}".format(date_end.strftime('%Y-%m-%d')))

        return date_start, date_end
    
    @staticmethod
    def normalize(data):
        return (data - JPLDGIM_mean) / JPLDGIM_std
        # return torch.log1p(data)

    @staticmethod
    def unnormalize(data):
        return data * JPLDGIM_std + JPLDGIM_mean

    def __len__(self):
        return self.num_samples
    
    @lru_cache(maxsize=1024) # number of days to cache in memory, roughly 3 MiB per day
    def _get_day_data(self, date):
        file_name = f"jpld{date:%j}0.{date:%y}i.nc.gz"
        file_path = os.path.join(self.data_dir, f"{date:%Y}", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with gzip.open(file_path, 'rb') as f:
            ds = xr.open_dataset(f, engine='h5netcdf')
            
            # Assuming 'tecmap' is the variable of interest
            data = ds['tecmap'].values
            # data_tensor shape torch.Size([96, 180, 360]) where 96 is nepochs, 180 is nlats, and 360 is nlons
            data_tensor = torch.tensor(data, dtype=torch.float32)
            if self.normalize:
                data_tensor = JPLDGIMDataset.normalize(data_tensor)

            return data_tensor

    def __getitem__(self, index):
        samples_per_day = 24 * 60 // 15  # 15-minute cadence
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, int):
            if index < 0 or index >= self.num_samples:
                raise IndexError("Index out of range for the dataset.")
            days = index // samples_per_day
            minutes = (index % samples_per_day) * 15
            date = self.date_start + datetime.timedelta(days=days, minutes=minutes)
        else:
            raise TypeError("Index must be an integer or a datetime object.")

        data = self._get_day_data(date)
        time_index = (index % samples_per_day)  # Get the time index within the
        data = data[time_index, :, :]  # Select the specific time slice

        return data
