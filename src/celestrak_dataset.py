'''
Solar indices pytorch dataset

Reads in data from the Celestrak dataset.

Full dataset information:
    Celestrak
    kp_ap_timeseries.csv:
        Datetime, Ap, Kp
    example:
        1957-10-01 00:00:00,43.0,32.0
        ...
    
'''

import torch
import torch.utils.data.Dataset as Dataset
import os
import numpy as np
import datetime
import pandas as pd

class CelestrakDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, date_start=None, date_end=None, normalize=True, cadence=180):
        print('Celestrak Dataset')

        self.data_file = data_file
        self.normalize = normalize
        self.cadence = cadence  # in minutes

        # Load the data file
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        self.df = pd.read_csv(data_file, parse_dates=['Datetime'], index_col='Datetime')
        print(f"Head of data file: \n{self.df.head()}")

        # Normalize the data if required
        if normalize:
            self.df = self.normalize(self.df)

        # Get the date range from the data file
        dates_available = self.find_date_range(data_file)
        if dates_available is None:
            raise ValueError("No data found in the specified file.")
        date_start_on_disk, date_end_on_disk = dates_available

        # If no start or end date is provided, use the dates from the file
        self.date_start = date_start_on_disk if date_start is None else date_start
        self.date_end = date_end_on_disk if date_end is None else date_end

        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
            raise ValueError("Specified date range is outside the available data range.")

        # Calculate the number of days and samples in the dataset
        self.num_days = (self.date_end - self.date_start).days + 1
        self.num_samples = int(self.num_days * (24 * 60 / cadence))
        assert self.num_samples == len(self.df), "Number of samples does not match the length of the data file."

        print('Number of days in dataset   : {:,}'.format(self.num_days))
        print('Number of samples in dataset: {:,}'.format(self.num_samples))

        # Calculate the size of the dataset on disk
        size_on_disk = sum(os.path.getsize(f) for f in glob(data_file))
        print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))

    @staticmethod
    def find_date_range():
        print("Checking date range of data in file: {}".format(data_file))

        # Get the first and last dates from self.df
        start_idx = self.df.index.min()
        end_idx = self.df.index.max()

        # Convert to datetime objects
        date_start = datetime.datetime.strptime(start_idx, '%Y-%m-%d %H:%M:%S')
        date_end = datetime.datetime.strptime(end_idx, '%Y-%m-%d %H:%M:%S')

        print("Start date : {}".format(date_start.strftime('%Y-%m-%d %H:%M:%S')))
        print("End date   : {}".format(date_end.strftime('%Y-%m-%d %H:%M:%S')))

        return date_start, date_end
    
    @staticmethod
    def normalize(df):
        for col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

    @staticmethod
    def unnormalize(df):
        for col in df.columns:
            df[col] = df[col] * df[col].std() + df[col].mean()
        return df

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, int):
            if index < 0 or index >= self.num_samples:
                raise IndexError("Index out of range for the dataset.")
            date = df.index[self.date_start + datetime.timedelta(minutes=index * self.cadence)]
        else:
            raise TypeError("Index must be an integer or a datetime object.")

        data = self.df.loc[date]
        # Create a 1D tensor listing the associated values with date
        data_tensor = torch.tensor(data.values, dtype=torch.float32)

        return data_tensor #, date ?

