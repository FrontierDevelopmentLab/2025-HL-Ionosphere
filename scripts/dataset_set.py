import torch
import os
from pathlib import Path
import pandas as pd

from dataset_pandasdataset import PandasDataset
from util import yeojohnson, yeojhonson_inverse

# Space Environment Technologies (SET) dataset
# https://setinc.com/space-environment-technologies/


set_all_columns = ['space_environment_technologies__f107_obs__',
                    'space_environment_technologies__f107_average__',
                    'space_environment_technologies__s107_obs__',
                    'space_environment_technologies__s107_average__',
                    'space_environment_technologies__m107_obs__',
                    'space_environment_technologies__m107_average__',
                    'space_environment_technologies__y107_obs__',
                    'space_environment_technologies__y107_average__',
                    'JB08__d_st_dt__[K]']

set_yeojohnson_lambdas = {'space_environment_technologies__f107_obs__': -0.763051477303,
                                 'space_environment_technologies__f107_average__': -0.551868341046,
                                 'space_environment_technologies__s107_obs__': 0.751363821795,
                                 'space_environment_technologies__s107_average__': 0.830731313661,
                                 'space_environment_technologies__m107_obs__': -0.407092166244,
                                 'space_environment_technologies__m107_average__': -0.305245272819,
                                 'space_environment_technologies__y107_obs__': 0.381375169921,
                                 'space_environment_technologies__y107_average__': 0.409993940749,
                                 'JB08__d_st_dt__[K]': 0.484842249526}

set_mean_of_yeojohnson = {'space_environment_technologies__f107_obs__': 1.272172135660,
                                      'space_environment_technologies__f107_average__': 1.672840949524,
                                      'space_environment_technologies__s107_obs__': 42.899690917474,
                                      'space_environment_technologies__s107_average__': 56.218558006899,
                                      'space_environment_technologies__m107_obs__': 2.086639724956,
                                      'space_environment_technologies__m107_average__': 2.486937790517,
                                      'space_environment_technologies__y107_obs__': 13.190408983866,
                                      'space_environment_technologies__y107_average__': 14.416800581748,
                                      'JB08__d_st_dt__[K]': 11.168264154109}

set_std_of_yeojohnson = {'space_environment_technologies__f107_obs__': 0.009865107914,
                                    'space_environment_technologies__f107_average__': 0.025485688944,
                                    'space_environment_technologies__s107_obs__': 14.521479851065,
                                    'space_environment_technologies__s107_average__': 21.361781197713,
                                    'space_environment_technologies__m107_obs__': 0.053843493314,
                                    'space_environment_technologies__m107_average__': 0.083871042526,
                                    'space_environment_technologies__y107_obs__': 2.167670990277,
                                    'space_environment_technologies__y107_average__': 2.403143006206,
                                    'JB08__d_st_dt__[K]': 5.629712229161}

set_all_columns_yeojohnson_lambdas = torch.tensor([set_yeojohnson_lambdas[col] for col in set_all_columns], dtype=torch.float32)
set_all_columns_mean_of_yeojohnson = torch.tensor([set_mean_of_yeojohnson[col] for col in set_all_columns], dtype=torch.float32)
set_all_columns_std_of_yeojohnson = torch.tensor([set_std_of_yeojohnson[col] for col in set_all_columns], dtype=torch.float32)

# ionosphere-data/set/karman-2025_data_sw_data_set_sw.csv
class SET(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=1440, date_exclusions=None, column=set_all_columns, delta_minutes=15): # 50 minutes rewind defualt
        print('\nSpace Environment Technologies')
        print('File           : {}'.format(file_name))

        data = pd.read_csv(file_name)

        # rename all__dates_datetime__ to Datetime
        data.rename(columns={'all__dates_datetime__': 'Datetime'}, inplace=True)
        # convert Datetime to datetime
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.copy()

        self.column = column

        stem = Path(file_name).stem
        new_stem = f"{stem}_deltamin_{delta_minutes}_rewind_{rewind_minutes}" 
        cadence_matched_fname = Path(file_name).with_stem(new_stem)
        if cadence_matched_fname.exists():
            print(f"Using cached file: {cadence_matched_fname}")
            data = pd.read_csv(cadence_matched_fname)
        else:
            data = PandasDataset.fill_to_cadence(data, delta_minutes=delta_minutes, rewind_minutes=rewind_minutes)
            data.to_csv(cadence_matched_fname) # the fill to cadence can take a while, so cache file

        super().__init__('Space Environment Technologies', data, self.column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data): 
        if self.column == set_all_columns:
            data = yeojohnson(data, set_all_columns_yeojohnson_lambdas)
            data = (data - set_all_columns_mean_of_yeojohnson) / set_all_columns_std_of_yeojohnson
        else:
            lambdas = torch.tensor([set_yeojohnson_lambdas[col] for col in self.column], dtype=torch.float32)
            means = torch.tensor([set_mean_of_yeojohnson[col] for col in self.column], dtype=torch.float32)
            stds = torch.tensor([set_std_of_yeojohnson[col] for col in self.column], dtype=torch.float32)
            data = yeojohnson(data, lambdas)
            data = (data - means) / stds
        return data

    def unnormalize_data(self, data):
        if self.column == set_all_columns:
            data = data * set_all_columns_std_of_yeojohnson + set_all_columns_mean_of_yeojohnson
            data = yeojhonson_inverse(data, set_all_columns_yeojohnson_lambdas)
        else:
            lambdas = torch.tensor([set_yeojohnson_lambdas[col] for col in self.column], dtype=torch.float32)
            means = torch.tensor([set_mean_of_yeojohnson[col] for col in self.column], dtype=torch.float32)
            stds = torch.tensor([set_std_of_yeojohnson[col] for col in self.column], dtype=torch.float32)
            data = data * stds + means
            data = yeojhonson_inverse(data, lambdas)
        return data