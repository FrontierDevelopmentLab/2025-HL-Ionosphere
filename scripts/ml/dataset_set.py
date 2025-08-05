import torch
import os
from pathlib import Path
import pandas as pd

from dataset_pandasdataset import PandasDataset

# Space Environment Technologies (SET) dataset
# https://setinc.com/space-environment-technologies/

set_all_columns = ['F10', 'S10', 'M10', 'Y10']

set_column_means = {'F10': 113.35810089111328,
                    'S10': 109.36700439453125,
                    'M10': 114.98369598388672,
                    'Y10': 114.7717056274414}

set_column_stds = {'F10': 44.7425422668457,
                   'S10': 43.588809967041016,
                   'M10': 43.91071319580078,
                   'Y10': 39.15766906738281}

set_all_columns_means = torch.tensor([set_column_means[col] for col in set_all_columns], dtype=torch.float32)
set_all_columns_stds = torch.tensor([set_column_stds[col] for col in set_all_columns], dtype=torch.float32)

# ionosphere-data/set/space_env_tech_indices_Indices_F10_processed.csv
class SET(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=1440, date_exclusions=None, column=set_all_columns, delta_minutes=15): # 50 minutes rewind defualt
        print('\nSpace Environment Technologies')
        print('File           : {}'.format(file_name))

        data = pd.read_csv(file_name)

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
        data = (data - set_all_columns_means) / set_all_columns_stds
        return data

    def unnormalize_data(self, data):
        data = data * set_all_columns_stds + set_all_columns_means
        return data

