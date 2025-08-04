import torch
import os
from pathlib import Path
import pandas as pd

from dataset_pandasdataset import PandasDataset


all_columns = ['omniweb__ae_index__[nT]',
               'omniweb__al_index__[nT]',
               'omniweb__au_index__[nT]',
               'omniweb__sym_d__[nT]', 
               'omniweb__sym_h__[nT]', 
               'omniweb__asy_d__[nT]',
               'omniweb__bx_gse__[nT]', 
               'omniweb__by_gse__[nT]', 
               'omniweb__bz_gse__[nT]',
               'omniweb__speed__[km/s]', 
               'omniweb__vx_velocity__[km/s]', 
               'omniweb__vy_velocity__[km/s]', 
               'omniweb__vz_velocity__[km/s]']

# ionosphere-data/omniweb_karman_2025
class OMNIWeb(PandasDataset):
    def __init__(self, data_dir, date_start=None, date_end=None, normalize=True, rewind_minutes=50, date_exclusions=None, column=all_columns, delta_minutes=15): # 50 minutes rewind defualt
        file_name_indices = os.path.join(data_dir, 'omniweb_indices_15min.csv')
        file_name_magnetic_field = os.path.join(data_dir, 'omniweb_magnetic_field_15min.csv')
        file_name_solar_wind = os.path.join(data_dir, 'omniweb_solar_wind_15min.csv')

        print('\nOMNIWeb')
        print('File indices           : {}'.format(file_name_indices))
        print('File magnetic field    : {}'.format(file_name_magnetic_field))
        print('File solar wind        : {}'.format(file_name_solar_wind))
        # delta_minutes = 1
        # delta_minutes = 15 # unclear what this is immediately, i think its supposed to match cadence but something to check out tmo

        data_indices = pd.read_csv(file_name_indices)
        data_magnetic_field = pd.read_csv(file_name_magnetic_field)
        data_solar_wind = pd.read_csv(file_name_solar_wind)

        #concat all columns to a single dataframe
        data = pd.concat([data_indices, data_magnetic_field, data_solar_wind], axis=1)

        # Remove duplicate columns (keep first occurrence)
        data = data.loc[:, ~data.columns.duplicated()]

        print('Data shape             : {}'.format(data.shape))
        print('Data columns           : {}'.format(data.columns.tolist()))
        print('Data index             : {}'.format(data.index.name))
        print(data.head())
        
        # rename all__dates_datetime__ to Datetime
        data.rename(columns={'all__dates_datetime__': 'Datetime'}, inplace=True)
        # convert Datetime to datetime
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.copy()

        self.column = column

        super().__init__('OMNIWeb', data, self.column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    # NOTE: what is the reason these methods were kept as instance methods but also passing the data in as an argument in the RSTNRadio dataset (or the other dattasets as well)?
    def normalize_data(self, data): 
        return data

    def unnormalize_data(self, data):
        return data

