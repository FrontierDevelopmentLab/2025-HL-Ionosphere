'''
Solar indices pytorch dataset

Reads in data from the OMNI and Celestrak datasets.

Full dataset information:
    OMNIWeb
        # 'Year', 'Day', 'Hour', 'Minute',
        # 'IMF_ID', 'SW_ID', 'Npts_IMF', 'Npts_SW', 'Pct_interp',
        # 'Time_shift', 'RMS_shift', 'RMS_PFN', 'DBOT1',
        # 'B_mag', 'Bx_GSE', 'By_GSE', 'Bz_GSE', 'By_GSM', 'Bz_GSM',
        # 'RMS_B_scalar', 'RMS_B_vector',
        # 'V_flow', 'Vx', 'Vy', 'Vz',
        # 'Density', 'Temp', 'P_dyn', 'E_field', 'Beta', 'Mach_Alfven',
        # 'X_GSE', 'Y_GSE', 'Z_GSE',
        # 'BSN_X', 'BSN_Y', 'BSN_Z',
        # 'AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',
        # 'PC_N', 'Mach_sonic',
        # 'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV
        # NaN values fill data timestamp where data is not present
        # each file is actuall a csv.gzip file

    Celestrak
        DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,
        AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OB

We only use the following columns (for initial research):
OMNIWeb
    'Year', 'Day', 'Hour', 'Minute',

    # Solar
    'B_mag', 'Bx_GSE', 'By_GSM', 'Bz_GSM', #IMF interplanetary magnetic field
    'RMS_B_scalar', 'RMS_B_vector',

    # Solar
    'V_flow', 'Vx', 'Vy', 'Vz',
    'Density', 'Temp', 

    # Solar derivations (tbd on including)
    #'P_dyn', 'E_field', 'Beta', 'Mach_Alfven',

    # Geomagnetic
    'AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',

    # Solar particle flux
    'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV'


Celestrak
    
'''

import torch
import os
import numpy as np
import datetime
from glob import glob
import pandas as pd
import time


# TODO: NaNs currently not dealt with, this should go into a new script, omniweb_process
# TODO: Compute mean and std, will update data_stats.py
class OMNIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        omni_dir,
        date_start=None, 
        date_end=None, 
        normalize=True,
        transform=None,
        omni_columns = None,
        sampled_cadence = 15,
        ):
        super().__init__()

        print('OMNI Dataset')
        self.omni_dir = omni_dir
        self.normalize = normalize
        dates_avialable = self.find_date_range(omni_dir)
        if dates_avialable is None:
            raise ValueError("No data found in the specified directory.")
        if omni_columns is None:
                # 'Year', 'Day', 'Hour', 'Minute',
            self.omni_columns = [
                # Solar
                'B_mag', 'Bx_GSE', 'By_GSM', 'Bz_GSM', #IMF interplanetary magnetic field
                'RMS_B_scalar', 'RMS_B_vector',
                # Solar
                'V_flow', 'Vx', 'Vy', 'Vz',
                'Density', 'Temp', 
                # Solar derivations (tbd on including)
                #'P_dyn', 'E_field', 'Beta', 'Mach_Alfven',

                # Geomagnetic
                'AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',

                # Solar particle flux
                'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV'
            ]
        else:
            self.omni_columns = omni_columns

        date_start_on_disk, date_end_on_disk = dates_avialable

        self.date_start = date_start_on_disk if date_start is None else date_start
        self.date_end = date_end_on_disk if date_end is None else date_end

        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
            raise ValueError("Specified date range is outside the available data range.")

        self.num_days = (self.date_end - self.date_start).days + 1
        self.true_cadence = 5 # minutes
        self.sampled_cadence = sampled_cadence # minutes (actual cadence is 5 minutes but sample at 15 minutes for consistency)
        self.num_samples = int(self.num_days * (24 * 60 / self.sampled_cadence))

        print('Number of days in dataset   : {:,}'.format(self.num_days))
        print('Number of samples in dataset: {:,}'.format(self.num_samples))

        # size on disk
        size_on_disk = sum(os.path.getsize(f) for f in glob(f"{omni_dir}/*.csv"))
        print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))
    
    
    @staticmethod
    def normalize(data):
        raise NotImplementedError("Normalization not implemented yet.")
        # return torch.log1p(data)

    @staticmethod
    def unnormalize(data):
        raise NotImplementedError("Unnormalization not implemented yet.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, int):
            if index < 0 or index >= self.num_samples:
                raise IndexError("Index out of range.")
            minutes = index * self.sampled_cadence
            date = self.date_start + datetime.timedelta(minutes=minutes)
        else:
            raise TypeError("Index must be an integer or datetime object.")
        
        data, shifted_date = self._get_data_by_date(date)
        return data, shifted_date
    
    def _get_data_by_date(self, date: datetime.datetime):
        # get nearest available timestamp
        nearest_minute = date.minute - date.minute % self.true_cadence #
        date = date.replace(second = 0)
        date = date.replace(minute = nearest_minute)

        filename = os.path.join(self.omni_dir, f"omni_5min{date.year:04d}.csv")
        datetime_str = datetime.datetime.strftime(date, "%Y-%m-%d %H:%M:%S")

        df = pd.read_csv(filename, compression="gzip")
        data_row = df[df["Datetime"] == datetime_str]

        data_tensor = torch.tensor(data_row[self.omni_columns].values, dtype=torch.float32)

        if self.normalize:
            data_tensor = OMNIDataset.normalize(data_tensor)
        print()
        return data_tensor, date

    @staticmethod
    def find_date_range(directory):
        # print("Checking date range of data in directory: {}".format(directory))
        files = sorted(glob(f"{directory}/*.csv"))
        
        if len(files) == 0:
            return None

        date_start_str = pd.read_csv(files[0], compression="gzip")["Datetime"].iloc[0] # '2006-01-01 00:00:00' example
        date_end_str = pd.read_csv(files[-1], compression="gzip")["Datetime"].iloc[-1] # '2006-01-01 00:00:00' example
        date_start = datetime.datetime.strptime(date_start_str, "%Y-%m-%d %H:%M:%S")
        date_end = datetime.datetime.strptime(date_end_str, "%Y-%m-%d %H:%M:%S")

        print("Directory  : {}".format(directory))
        print("Start date : {}".format(date_start.strftime('%Y-%m-%d')))
        print("End date   : {}".format(date_end.strftime('%Y-%m-%d')))

        return date_start, date_end