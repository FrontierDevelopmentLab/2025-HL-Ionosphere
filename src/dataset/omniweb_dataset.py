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
    
'''
# TODO: make TimeSeriesDataset / CSVDataset base class
# TODO: should be made more robust to certain conditions (what if omni_rows empty) Also what if other parameters have major nan issues like the goes rows, this is more of a preprocessing problem
# TODO: not great to load the csvs each time, 
# 

from base_datasets import PandasDataset

import torch
import os
import numpy as np
import datetime
from glob import glob
import pandas as pd
import time

class OMNIDataset(PandasDataset):
    def __init__(self, file_dir, date_start=None, date_end=None, normalize=True, rewind_minutes=50, date_exclusions=None, column=None): # 50 minutes rewind defualt

        print('\nOMNIWeb dataset')
        print('File                 : {}'.format(file_name))
        # delta_minutes = 1
        delta_minutes = 15 # unclear what this is immediately, i think its supposed to match cadence but something to check out tmo

        file_name = os.path.join(file_dir, "omni_5min_full_cleaned.csv")
        stats_dir = os.path.join(file_dir, "omni_stats.csv")

        self.stats_df = pd.read_csv(stats_dir)

        data = pd.read_csv(file_name)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        # data = data.sort_values(by='Datetime')
        print('Rows                 : {:,}'.format(len(data)))

        if column is None:
            self.column = [
                # Solar
                'B_mag', 'Bx_GSE', 'By_GSM', 'Bz_GSM', #IMF interplanetary magnetic field
                'RMS_B_scalar', 'RMS_B_vector',
                # Solar
                'V_flow', 'Vx', 'Vy', 'Vz',
                'Density', 'Temp', 

                # Solar derivations (tbd on including) # NOTE: what was the consensus on  including these by default?
                'P_dyn', 'E_field', 'Beta', 'Mach_Alfven', 

                # Geomagnetic
                # 'AE', 'AL', 'AU', #  (Missing in 2019-2020)
                'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',

                # # Solar particle flux # Unusable in Omni dataset
                # 'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV' # (missing 2020 onwards)
            ]
        else:
            self.column = column
        # Remove outliers based on quantiles, # NOTE: is this something we care about? we have already removed the 999... so missing data based outliers shouldnt be an issue unless if some were missed (there was one atleast missed but was for an unused column)
        q_low = data[self.column].quantile(0.001)
        q_hi  = data[self.column].quantile(0.999)
        data = data[(data[self.column] < q_hi) & (data[self.column] > q_low)]

        super().__init__('OMNIWeb dataset', data, column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    # NOTE: what is the reason these methods were kept as instance methods but also passing the data in as an argument in the RSTNRadio dataset (or the other dattasets as well)?
    def normalize_data(self, data, omni_columns): 
        stats = torch.tensor(self.stats_df[omni_columns].values, dtype=torch.float32)
        omni_means = stats[0]
        omni_stds = stats[1]
        return (data - omni_means) / omni_stds

    def unnormalize_data(self, data, omni_columns):
        stats = torch.tensor(self.stats_df[omni_columns].values, dtype=torch.float32)
        omni_means = stats[0]
        omni_stds = stats[1]
        return data * omni_stds + omni_means

    # def unnormalize_data(self, data):
    #     data = data * data * data
    #     mean_cuberoot_data = 2.264420986175537
    #     std_cuberoot_data = 0.9795352816581726
    #     data = data * std_cuberoot_data
    #     data = data + mean_cuberoot_data
    #     return data


# stats_file = "/mnt/ionosphere-data/omniweb/processed/dataset_stats/omni_stats.csv"
# stats_df = pd.read_csv(stats_file)

# class OMNIDataset(torch.utils.data.Dataset):
#     def __init__(
#         self, 
#         omni_dir,
#         date_start=None, 
#         date_end=None, 
#         normalize=True,
#         omni_columns = None,
#         sampled_cadence = 15,
#         ):
#         super().__init__()

#         print('OMNI Dataset')
#         self.omni_dir = omni_dir
       
#         self.normalize = normalize
#         dates_avialable = self.find_date_range(omni_dir)
#         if dates_avialable is None:
#             raise ValueError("No data found in the specified directory.")
#         if omni_columns is None:
#                 # 'Year', 'Day', 'Hour', 'Minute',
#             self.omni_columns = [
#                 # Solar
#                 'B_mag', 'Bx_GSE', 'By_GSM', 'Bz_GSM', #IMF interplanetary magnetic field
#                 'RMS_B_scalar', 'RMS_B_vector',
#                 # Solar
#                 'V_flow', 'Vx', 'Vy', 'Vz',
#                 'Density', 'Temp', 
#                 # Solar derivations (tbd on including)
#                 #'P_dyn', 'E_field', 'Beta', 'Mach_Alfven',

#                 # Geomagnetic
#                 # 'AE', 'AL', 'AU', #  (Missing in 2019-2020)
#                 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',

#                 # # Solar particle flux # Unusable in Omni dataset
#                 # 'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV' # (missing)
#             ]
#         else:
#             self.omni_columns = omni_columns

#         date_start_on_disk, date_end_on_disk = dates_avialable

#         self.date_start = date_start_on_disk if date_start is None else date_start
#         self.date_end = date_end_on_disk if date_end is None else date_end

#         if self.date_start > self.date_end:
#             raise ValueError("Start date cannot be after end date.")
#         if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
#             raise ValueError("Specified date range is outside the available data range.")

#         self.num_days = (self.date_end - self.date_start).days + 1
#         self.true_cadence = 5 # minutes
#         self.sampled_cadence = sampled_cadence # minutes (actual cadence is 5 minutes but sample at 15 minutes for consistency)
#         self.num_samples = int(self.num_days * (24 * 60 / self.sampled_cadence))

#         print('Number of days in dataset   : {:,}'.format(self.num_days))
#         print('Number of samples in dataset: {:,}'.format(self.num_samples))

#         # size on disk
#         size_on_disk = sum(os.path.getsize(f) for f in glob(f"{omni_dir}/*.csv"))
#         print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))
    
    
#     # I think this might be worth making into an instance method since need to pass in also
#     # which columns are expected, but kept it as a static method for consistency between classes
#     @staticmethod
#     def normalize(data, omni_columns): 
#         stats = torch.tensor(stats_df[omni_columns].values, dtype=torch.float32)
#         omni_means = stats[0]
#         omni_stds = stats[1]
#         return (data - omni_means) / omni_stds

#     @staticmethod
#     def unnormalize(data, omni_columns):
#         stats = torch.tensor(stats_df[omni_columns].values, dtype=torch.float32)
#         omni_means = stats[0]
#         omni_stds = stats[1]        
#         return data * omni_stds + omni_means

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, index):
#         if isinstance(index, datetime.datetime):
#             date = index
#         elif isinstance(index, int):
#             if index < 0 or index >= self.num_samples:
#                 raise IndexError("Index out of range.")
#             minutes = index * self.sampled_cadence
#             date = self.date_start + datetime.timedelta(minutes=minutes)
#         else:
#             raise TypeError("Index must be an integer or datetime object.")
        
#         data, shifted_date = self._get_data_by_date(date)
#         return data, shifted_date
    
#     def _get_data_by_date(self, date: datetime.datetime):
#         # get nearest available timestamp
#         nearest_minute = date.minute - date.minute % self.true_cadence #
#         date = date.replace(second = 0)
#         date = date.replace(minute = nearest_minute)

#         filename = os.path.join(self.omni_dir, f"omni_5min_{date.year:04d}_cleaned.csv") # filename cleaned postfix should maybe be removed
#         datetime_str = datetime.datetime.strftime(date, "%Y-%m-%d %H:%M:%S")

#         df = pd.read_csv(filename)
#         data_row = df[df["Datetime"] == datetime_str]
#         data_tensor = torch.tensor(data_row[self.omni_columns].values, dtype=torch.float32)

#         if self.normalize:
#             data_tensor = OMNIDataset.normalize(data_tensor,self.omni_columns)

#         return data_tensor, date.isoformat() if hasattr(date, 'isoformat') else str(date)


#     def get_date_range(self):
#         return self.date_start, self.date_end
    
#     def set_date_range(self, date_start, date_end):
#         self.date_start, self.date_end = date_start, date_end

#     @staticmethod
#     def find_date_range(directory):
#         # print("Checking date range of data in directory: {}".format(directory))
#         files = sorted(glob(f"{directory}/*.csv"))
        
#         if len(files) == 0:
#             return None

#         date_start_str = pd.read_csv(files[0])["Datetime"].iloc[0] # '2006-01-01 00:00:00' example
#         date_end_str = pd.read_csv(files[-1])["Datetime"].iloc[-1] # '2006-01-01 00:00:00' example
#         date_start = datetime.datetime.strptime(date_start_str, "%Y-%m-%d %H:%M:%S")
#         date_end = datetime.datetime.strptime(date_end_str, "%Y-%m-%d %H:%M:%S")

#         print("Directory  : {}".format(directory))
#         print("Start date : {}".format(date_start.strftime('%Y-%m-%d')))
#         print("End date   : {}".format(date_end.strftime('%Y-%m-%d')))

#         return date_start, date_end