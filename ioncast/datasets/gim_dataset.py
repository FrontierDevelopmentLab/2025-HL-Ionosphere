import torch
from torch.utils.data import Dataset
import os
import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from functools import lru_cache
from glob import glob
from tqdm import tqdm
from glob import glob
import gzip
import xarray as xr
import tarfile
import pickle
from io import BytesIO

JPLD_mean = 14.796479225158691
JPLD_std = 14.694787979125977
JPLD_mean_of_log1p = 2.4017040729522705
JPLD_std_of_log1p = 0.8782428503036499


def jpld_normalize(data):
    # return (data - JPLD_mean) / JPLD_std
    data = torch.log1p(data)
    data = (data - JPLD_mean_of_log1p) / JPLD_std_of_log1p
    return data


def jpld_unnormalize(data):
    # return data * JPLD_std + JPLD_mean
    data = data * JPLD_std_of_log1p + JPLD_mean_of_log1p
    data = torch.expm1(data)
    return data


class TarRandomAccess():
    def __init__(self, data_dir):
        tar_files = sorted(glob(os.path.join(data_dir, '*.tar')))
        if len(tar_files) == 0:
            raise ValueError('No tar files found in data directory: {}'.format(data_dir))
        self.index = {}
        index_cache = os.path.join(data_dir, 'tar_files_index')
        if os.path.exists(index_cache):
            print('Loading tar files index from cache: {}'.format(index_cache))
            with open(index_cache, 'rb') as file:
                self.index = pickle.load(file)
        else:
            for tar_file in tqdm(tar_files, desc='Indexing tar files'):
                with tarfile.open(tar_file) as tar:
                    for info in tar.getmembers():
                        self.index[info.name] = (tar.name, info)
            print('Saving tar files index to cache: {}'.format(index_cache))
            with open(index_cache, 'wb') as file:
                pickle.dump(self.index, file)
        self.file_names = list(self.index.keys())

    def __getitem__(self, file_name):
        d = self.index.get(file_name)
        if d is None:
            return None
        tar_file, tar_member = d
        with tarfile.open(tar_file) as tar:
            data = BytesIO(tar.extractfile(tar_member).read())
        return data


class WebDataset():
    def __init__(self, data_dir, decode_func=None):
        self.tars = TarRandomAccess(data_dir)
        if decode_func is None:
            self.decode_func = self.decode
        else:
            self.decode_func = decode_func
        
        self.index = {}
        self.prefixes = []
        for file_name in self.tars.file_names:
            p = file_name.split('.', 1)
            if len(p) == 2:
                prefix, postfix = p
                if prefix not in self.index:
                    self.index[prefix] = []
                    self.prefixes.append(prefix)
                self.index[prefix].append(postfix)

    def decode(self, data, file_name):
        if file_name.endswith('.npy'):
            data = np.load(data)
        else:
            raise ValueError('Unknown data type for file: {}'.format(file_name))    
        return data
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        if isinstance(index, str):
            prefix = index
        elif isinstance(index, int):
            prefix = self.prefixes[index]
        else:
            raise ValueError('Expecting index to be int or str')
        sample = self.index.get(prefix)
        if sample is None:
            return None
        
        data = {}
        data['__prefix__'] = prefix
        for postfix in sample:
            file_name = prefix + '.' + postfix
            d = self.decode(self.tars[file_name], file_name)
            data[postfix] = d
        return data


# JPLD GIM Dataset working with raw NetCDF files
# Note: seems to be slow to do all data processing on the fly
# Preferred to use the Parquet dataset for faster access
class JPLDRaw(Dataset):
    def __init__(self, data_dir, date_start=None, date_end=None, normalize=True):
        print('JPLD Dataset')
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

        days = [d.replace(directory, '') for d in days]
        date_start = datetime.datetime.strptime(days[0].split('.')[0], "/%Y/jpld%j0")
        date_end = datetime.datetime.strptime(days[-1].split('.')[0], "/%Y/jpld%j0")

        print("Directory  : {}".format(directory))
        print("Start date : {}".format(date_start.strftime('%Y-%m-%d')))
        print("End date   : {}".format(date_end.strftime('%Y-%m-%d')))

        return date_start, date_end
    
    def __len__(self):
        return self.num_samples
    
    @lru_cache(maxsize=4096) # number of days to cache in memory, roughly 3 MiB per day
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
                data_tensor = jpld_normalize(data_tensor)

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
        data = data.unsqueeze(0)  # Add a channel dimension

        return data, date.isoformat()  # Return the data and the date as a string


class JPLDGIMDataset(Dataset):
    def __init__(self, data_dir, date_start=None, date_end=None, date_exclusions=None, normalize=True):
        self.data_dir = data_dir
        self.normalize = normalize
        print('\nJPLD')

        print('Directory  : {}'.format(self.data_dir))
        self.data = WebDataset(data_dir)

        self.date_start, self.date_end = self.find_date_range()
        if date_start is not None:
            if isinstance(date_start, str):
                date_start = datetime.datetime.fromisoformat(date_start)
            
            if (date_start >= self.date_start) and (date_start < self.date_end):
                self.date_start = date_start
            else:
                print('Start date out of range, using default')
        if date_end is not None:
            if isinstance(date_end, str):
                date_end = datetime.datetime.fromisoformat(date_end)
            if (date_end > self.date_start) and (date_end <= self.date_end):
                self.date_end = date_end
            else:
                print('End date out of range, using default')
        self.delta_minutes = 15
        total_minutes = int((self.date_end - self.date_start).total_seconds() / 60)
        total_steps = total_minutes // self.delta_minutes
        print('Start date : {}'.format(self.date_start))
        print('End date   : {}'.format(self.date_end))
        print('Delta      : {} minutes'.format(self.delta_minutes))

        self.date_exclusions = date_exclusions
        if self.date_exclusions is not None:
            print('Date exclusions:')
            date_exclusions_postfix = '_exclusions'
            for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                print('  {} - {}'.format(exclusion_date_start, exclusion_date_end))
                date_exclusions_postfix += '__{}_{}'.format(exclusion_date_start.isoformat(), exclusion_date_end.isoformat())
        else:
            date_exclusions_postfix = ''

        self.dates = []
        dates_cache = 'dates_index_{}_{}{}'.format(self.date_start.isoformat(), self.date_end.isoformat(), date_exclusions_postfix)
        dates_cache = os.path.join(self.data_dir, dates_cache)
        if os.path.exists(dates_cache):
            print('Loading dates from cache: {}'.format(dates_cache))
            with open(dates_cache, 'rb') as f:
                self.dates = pickle.load(f)
        else:        
            for i in tqdm(range(total_steps), desc='Filtering dates'):
                date = self.date_start + datetime.timedelta(minutes=self.delta_minutes*i)
                exists = True
                prefix = self.date_to_prefix(date)
                data = self.data.index.get(prefix)
                if data is None:
                    exists = False

                if self.date_exclusions is not None:
                    for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                        if (date >= exclusion_date_start) and (date < exclusion_date_end):
                            exists = False
                            break
                if exists:
                    self.dates.append(date)
            print('Saving dates to cache: {}'.format(dates_cache))
            with open(dates_cache, 'wb') as f:
                pickle.dump(self.dates, f)
            
        if len(self.dates) == 0:
            raise RuntimeError('No data found in the specified range ({}) - ({})'.format(self.date_start, self.date_end))

        self.dates_set = set(self.dates)
        self.name = 'JPLD'

        print('TEC maps total    : {:,}'.format(total_steps))
        print('TEC maps available: {:,}'.format(len(self.dates)))
        print('TEC maps dropped  : {:,}'.format(total_steps - len(self.dates)))


    @lru_cache(maxsize=100000)
    def prefix_to_date(self, prefix):
        return datetime.datetime.strptime(prefix, '%Y/%m/%d/%H%M')
    
    @lru_cache(maxsize=100000)
    def date_to_prefix(self, date):
        return date.strftime('%Y/%m/%d/%H%M')

    def find_date_range(self):
        prefix_start = self.data.prefixes[0]
        prefix_end = self.data.prefixes[-1]
        date_start = self.prefix_to_date(prefix_start)
        date_end = self.prefix_to_date(prefix_end)
        return date_start, date_end
    
    def __repr__(self):
        return 'JPLD ({} - {})'.format(self.date_start, self.date_end)


    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            date = self.dates[index]
        elif isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_data(date)    
        return data, date.isoformat()
    
    def get_data(self, date):
        # if date < self.date_start or date > self.date_end:
        #     raise ValueError('Date ({}) out of range for JPLD ({} - {})'.format(date, self.date_start, self.date_end))

        if date not in self.dates_set:
            print('Date not found in JPLD : {}'.format(date))
            return None
        
        if self.date_exclusions is not None:
            for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                if (date >= exclusion_date_start) and (date < exclusion_date_end):
                    raise RuntimeError('Should not happen')

        prefix = self.date_to_prefix(date)
        data = self.data[prefix]
        tecmap = data['tecmap.npy']
        tecmap = torch.from_numpy(tecmap).unsqueeze(0)  # Add a channel dimension
        if self.normalize:
            tecmap = jpld_normalize(tecmap)

        return tecmap

    @staticmethod
    def normalize(data):
        return jpld_normalize(data)
    
    @staticmethod
    def unnormalize(data):
        return jpld_unnormalize(data)