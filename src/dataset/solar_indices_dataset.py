'''
Solar indices pytorch dataset

Reads in data from the Space Environment Technologies Solar Index datasets.

Indices_F10_processed.csv:
    Datetime, F10, F81c, S10, S81c, M10, M81c, Y10, Y81c

    F81 is an 81 day smoothed F10.7 index
    S81 is an 81 day smoothed sunspot number index
    M81 is an 81 day smoothed Mg II index
    Y81 is an 81 day smoothed 10.7 cm solar radio flux index

Simone says use F10, S10, M10, Y10

Whenever there isn't a measurement, the value is 0.0. This is dealth with by forward filling in process_celestrak_and_indices.py.
'''

import torch
import glob as glob
import os
import numpy as np
import datetime
import pandas as pd

class SolarIndexDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, date_start=None, date_end=None, normalize=True, cadence=24*60):
        print('Solar Index Dataset')

        self.data_file = data_file
        self.normalize = normalize
        self.cadence = cadence  # in minutes

        # Load the data file
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        df = pd.read_csv(data_file, usecols=['Datetime', 'F10', 'S10', 'M10', 'Y10'])

        # Convert the Datetime column from '%YYYY-%m-%d' to '%Y-%m-%d %H:%M:%S'
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d')
        df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Convert the Datetime column to the index
        df.set_index('Datetime', inplace=True)

        # Normalize the data if required
        # Note: if
        #   solarindex = SolarIndexDataset()
        #   solarindex.normalize // return True
        #   SolarIndexDataset.normalize // function
        if normalize:
            self.df = SolarIndexDataset.normalize(df)
        else:
            self.df = df

        print(f"Head of data file: \n\n{df.head()}\n")

        # Get the date range from the data file
        dates_available = self.find_date_range(data_file, self.df)
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
        size_on_disk = sum(os.path.getsize(f) for f in glob.glob(data_file))
        print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))

    @staticmethod
    def find_date_range(data_file, df):
        print("Checking date range of data in file: {}".format(data_file))

        # Get the first and last dates from self.df
        start_idx = df.index.min()
        end_idx = df.index.max()

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
            date_string = date.strftime('%Y-%m-%d %H:%M:%S')
            df_index = self.df.index.searchsorted(date_string)
        elif isinstance(index, int):
            if index < 0 or index >= self.num_samples:
                raise IndexError("Index out of range for the dataset.")
            df_index = index
            date_string = self.df.index[df_index]
            date = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') # convert to datetime obj for return
        else:
            raise TypeError("Index must be an integer or a datetime object.")

        print(f"Index in DataFrame: {df_index}, Date: {date_string}, value: {self.df.iloc[df_index].values}")

        # Create a 1D tensor listing the associated values with date
        data = self.df.iloc[df_index].values
        data_tensor = torch.tensor(data, dtype=torch.float32)

        return data_tensor, date.isoformat() if hasattr(date, 'isoformat') else str(date)
