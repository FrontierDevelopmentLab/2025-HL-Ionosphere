import torch
import os
from pathlib import Path
import pandas as pd

from dataset_pandasdataset import PandasDataset

# Consider the following scaling, not currently implemented:
# omniweb__ae_index__[nT] clamp(0, inf) -> log1p -> z-score
# omniweb__al_index__[nT] -= 11 -> clamp(-inf, 0) -> neg -> log1p -> z-score
# omniweb__au_index__[nT] += 9 -> log1p -> z-score
# omniweb__sym_d__[nT] z-score
# omniweb__sym_h__[nT] z-score
# omniweb__asy_d__[nT] z-score
# omniweb__bx_gse__[nT] z-score
# omniweb__by_gse__[nT] z-score
# omniweb__bz_gse__[nT] z-score
# omniweb__speed__[km/s] z-score
# omniweb__vx_velocity__[km/s] z-score
# omniweb__vy_velocity__[km/s] z-score
# omniweb__vz_velocity__[km/s] z-score


omniweb_all_columns = ['omniweb__ae_index__[nT]',
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

omniweb_column_means = {'omniweb__ae_index__[nT]': 171.78419494628906,
                        'omniweb__al_index__[nT]': -104.44080352783203,
                        'omniweb__au_index__[nT]': 61.04460144042969,
                        'omniweb__sym_d__[nT]': -0.24560000002384186,
                        'omniweb__sym_h__[nT]': -10.619400024414062,
                        'omniweb__asy_d__[nT]': 18.769800186157227,
                        'omniweb__bx_gse__[nT]': -0.030805999413132668,
                        'omniweb__by_gse__[nT]': 0.009642007760703564,
                        'omniweb__bz_gse__[nT]': 0.033528003841638565,
                        'omniweb__speed__[km/s]': 418.8190002441406,
                        'omniweb__vx_velocity__[km/s]': -417.7730407714844,
                        'omniweb__vy_velocity__[km/s]': 0.2976999580860138,
                        'omniweb__vz_velocity__[km/s]': -2.238880157470703}

omniweb_column_stds = {'omniweb__ae_index__[nT]': 198.90577697753906,
                       'omniweb__al_index__[nT]': 145.14234924316406,
                       'omniweb__au_index__[nT]': 67.07356262207031,
                       'omniweb__sym_d__[nT]': 2.9613230228424072,
                       'omniweb__sym_h__[nT]': 17.97284507751465,
                       'omniweb__asy_d__[nT]': 13.876084327697754,
                       'omniweb__bx_gse__[nT]': 3.473517417907715,
                       'omniweb__by_gse__[nT]': 4.016483306884766,
                       'omniweb__bz_gse__[nT]': 3.231107711791992,
                       'omniweb__speed__[km/s]': 94.15789031982422,
                       'omniweb__vx_velocity__[km/s]': 89.83809661865234,
                       'omniweb__vy_velocity__[km/s]': 23.594484329223633,
                       'omniweb__vz_velocity__[km/s]': 21.573925018310547}
   
omniweb_all_columns_means = torch.tensor([omniweb_column_means[col] for col in omniweb_all_columns], dtype=torch.float32)
omniweb_all_columns_stds = torch.tensor([omniweb_column_stds[col] for col in omniweb_all_columns], dtype=torch.float32)

# ionosphere-data/omniweb_karman_2025
class OMNIWeb(PandasDataset):
    def __init__(self, data_dir, date_start=None, date_end=None, normalize=True, rewind_minutes=50, date_exclusions=None, column=omniweb_all_columns, delta_minutes=15): # 50 minutes rewind defualt
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

    def normalize_data(self, data): 
        if self.column == omniweb_all_columns:
            data = (data - omniweb_all_columns_means) / omniweb_all_columns_stds
        else:
            means = torch.tensor([omniweb_column_means[col] for col in self.column], dtype=torch.float32)
            stds = torch.tensor([omniweb_column_stds[col] for col in self.column], dtype=torch.float32)
            data = (data - means) / stds
        return data

    def unnormalize_data(self, data):
        if self.column == omniweb_all_columns:
            data = data * omniweb_all_columns_stds + omniweb_all_columns_means
        else:
            means = torch.tensor([omniweb_column_means[col] for col in self.column], dtype=torch.float32)
            stds = torch.tensor([omniweb_column_stds[col] for col in self.column], dtype=torch.float32)
            data = data * stds + means
        return data

