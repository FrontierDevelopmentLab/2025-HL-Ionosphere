import torch
from torch.utils.data import Dataset
import datetime
import numpy as np

import glob
import os

# directory = "/path/to/dir"




class QuasiDipole(Dataset):
    def __init__(self, data_dir, date_start=None, date_end=None, delta_minutes=15, k = 1):
        self.data_dir = data_dir
        self.date_start = date_start
        self.date_end = date_end
        self.delta_minutes = delta_minutes

        if self.date_start is None:
            self.date_start = datetime.datetime(2010, 5, 13, 0, 0, 0)
        if self.date_end is None:
            self.date_end = datetime.datetime(2024, 8, 1, 0, 0, 0)

        current_date = self.date_start
        self.dates = []
        while current_date <= self.date_end:
            self.dates.append(current_date)
            current_date += datetime.timedelta(minutes=self.delta_minutes)

        data_dict = {}
        for file_path in glob.glob(os.path.join(data_dir, "qd_*_*.npy")):
            filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
            _, coord, year = filename_no_ext.split("_")
            year = int(year)
            qd_grid = np.load(file_path)
            if year not in data_dict:
                data_dict[year] = {}
            data_dict[year][coord] = qd_grid
        
        self.dataset = {}
        for year in data_dict.keys():
            lat_grid = np.deg2rad(data_dict[year]["lat"])            
            lon_grid = np.deg2rad(data_dict[year]["lon"])
            feats = torch.tensor(
                np.array([
                    np.sin(k * lat_grid),
                    np.cos(k * lat_grid),
                    np.sin(k * lon_grid),
                    np.sin(k * lon_grid)
                    ]))
            self.dataset[year] = feats

        self.dates_set = set(self.dates)
        self.name = "QuasiDipole"

        print('\nQuasi Dipole Lat/Long coords')
        print('Start date              : {}'.format(self.date_start))
        print('End date                : {}'.format(self.date_end))
        print('Delta                   : {} minutes'.format(self.delta_minutes))
        print('Note that all files are yearly, but treated as if given in delta_minutes cadence')

    
    # def _load_files(self) -> dict[int, torch.Tensor]:

    # for file_path in glob.glob(os.path.join(self.data_dir, ".npy")):

    # print(file_path)        pass


    def __len__(self):
        return len(self.dates_set)

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        elif isinstance(index, int):
            if index < 0 or index >= len(self.dates):
                raise IndexError("Index out of range for the dataset.")
            date = self.dates[index]
        else:
            raise ValueError('Expecting index to be datetime.datetime or str (in the format of 2022-11-01T00:01:00), but got {}'.format(type(index)))

        if date not in self.dates_set:
            raise ValueError('Date {} not found in the dataset'.format(date))

        if date.tzinfo is None:
            date = date.replace(tzinfo=datetime.timezone.utc)

        return self.get_data(date), date.isoformat()


    def get_data(self, date):
        year = date.year
        return self.dataset[year]