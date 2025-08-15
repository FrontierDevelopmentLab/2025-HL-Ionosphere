import torch
import glob as glob
import os
import numpy as np
import datetime
from pathlib import Path
import pandas as pd
import h5py
from torch.utils.data import Dataset


def array_to_list_of_arrays(m):
    # For a matrix (2d numpy array), returns a list of arrays where each array is a row of the matrix.
    ret = []
    for row in m:
        ret.append(row)
    return ret



# function to change any date that ends with HH:12 -> HH:15, HH:24 -> HH:30, HH:36 -> HH:45, HH:48 -> HH:00
def adjust_date(date):
    if date.minute == 12:
        return date.replace(minute=15)
    elif date.minute == 24:
        return date.replace(minute=30)
    elif date.minute == 36:
        return date.replace(minute=45)
    elif date.minute == 48:
        # Handle hour rollover properly
        return date.replace(minute=0) + datetime.timedelta(hours=1)
    return date

# The underlying data has 12 mins cadence, e.g., 00:12, 00:24, 00:36, 00:48, ...
# ionosphere-data/sdocore/sdo_core_dataset_21504.h5
class SDOCore(Dataset):
    def __init__(self, file_name, date_start=None, date_end=None, rewind_minutes=36):
        self.file_name = file_name
        self.rewind_minutes = rewind_minutes

        print('\nSDOCore')
        print('File                  : {}'.format(file_name))

        data = {}
        with h5py.File(file_name, 'r') as f:
            data['year'] = f['year'][:]
            data['month'] = f['month'][:]
            data['day'] = f['day'][:]
            data['hour'] = f['hour'][:]
            data['minute'] = f['minute'][:]
            # Convert latent arrays to float32 before storing
            latent_data = f['latent'][:]
            data['latent'] = [arr.astype(np.float32) for arr in latent_data]

        self.data = pd.DataFrame(data)
        self.data['date'] = pd.to_datetime(self.data[['year', 'month', 'day', 'hour', 'minute']])
        self.data = self.data.drop(columns=['year', 'month', 'day', 'hour', 'minute'])

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

        print('Start date              : {}'.format(self.date_start))
        print('End date                : {}'.format(self.date_end))

        self.dates_internal = self.data['date'].tolist()
        self.dates_set_internal = set(self.dates_internal)

        # ugly hack to align this dataset with minutes
        # :12, :24, :36, :48 to :15, :30, :45, :00
        # adjust all internal dates with the adjust_date function and present them as self.dates
        self.dates = [adjust_date(date) for date in self.dates_internal]
        self.dates_set = set(self.dates)
        # also adjust the date_start and date_end
        self.date_start = adjust_date(self.date_start)
        self.date_end = adjust_date(self.date_end)

        print('Start date (adjusted)   : {}'.format(self.date_start))
        print('End date (adjusted)     : {}'.format(self.date_end))

        self.name = 'SDOCore'

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
        if date not in self.dates_set_internal:
            # print('Date not found in SDOCore : {}'.format(date))
            # try to find the closest date before the given date, start from the given date and go backwards in minute steps, do not go more than rewind_minutes back
            rewind_steps = self.rewind_minutes
            for i in range(1, rewind_steps):
                date_rewind = date - datetime.timedelta(minutes=i)
                if date_rewind in self.dates_set_internal:
                    date = date_rewind
                    # print('Rewinding to the closest date: {}'.format(date))
                    break
            else:
                return None  # No data found within the rewind range

        row = self.data[self.data['date'] == date]
        data = row.iloc[0]['latent'] if not row.empty else None
        data = torch.tensor(data, dtype=torch.float32)
        return data

    def find_date_range(self):
        if self.data is not None and not self.data.empty:
            return self.data['date'].min(), self.data['date'].max()
        return None, None

    def normalize_data(self, data):
        return data
    
    def unnormalize_data(self, data):
        return data
    

if __name__ == "__main__":
    # Example usage
    dataset = SDOCore(file_name='/disk2-ssd-8tb/data/2025-hl-ionosphere/sdocore/sdo_core_dataset_21504.h5', date_start='2020-01-01', date_end='2017-12-31')
    print(dataset.data.head())
    print("Dataset loaded successfully.")
    print(len(dataset))
    d, date = dataset[0]
    print(d.shape)
    print(dataset.date_start)
