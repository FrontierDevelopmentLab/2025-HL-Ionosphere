'''
Solar indices pytorch dataset

Reads in data from the Celestrak dataset.

Original file columns:
    DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,
    AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OB

Full dataset information:
    kp_ap_timeseries_processed.csv:
        Datetime, Ap, Kp
    example:
        1957-10-01 00:00:00,43.0,32.0
        ...
Whenever there isn't a measurement, the value is 0.0. This is dealth with by forward filling in process_celestrak_and_indices.py.
'''

import torch
import glob as glob
import os
import numpy as np
import datetime
from pathlib import Path
import pandas as pd

from dataset_pandasdataset import PandasDataset


CelesTrak_mean_of_log1p = torch.tensor([1.0448144674301147, 2.2307636737823486], dtype=torch.float32) # Kp, Ap
CelesTrak_std_of_log1p = torch.tensor([0.4444892108440399, 0.8150267004966736], dtype=torch.float32) # Kp, Ap

# Ap scaling
# https://en.wikipedia.org/wiki/A-index
# The Ap-index is the averaged planetary A-index based on data from a set of specific Kp stations.
#
# Kp scaling
# https://en.wikipedia.org/wiki/K-index
# This is an integer index in the range 0 to 9, where 1 is calm and 5 or more indicate geomagnetic storms.
#
# For both features, we apply log1p scaling and then (z-score) normalization.

# ionosphere-data/celestrak/kp_ap_processed_timeseries.csv
class CelesTrak(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=180, date_exclusions=None, delta_minutes=15, column=['Kp', 'Ap']): # 180 minutes rewind default matching dataset cadence (NOTE: what is a good max value for rewind_minutes?)
        print('\nCelesTrak')
        print('File                  : {}'.format(file_name))

        data = pd.read_csv(file_name)
        # data['Datetime'] = pd.to_datetime(data['Datetime'])
        # data = data.sort_values(by='Datetime')
        if column != ['Kp', 'Ap'] and column != ['Kp'] and column != ['Ap']:
            raise ValueError(f"Unknown column configuration: {column}. Expected ['Kp', 'Ap'], ['Kp'], or ['Ap'].")
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

        super().__init__('CelesTrak', data, self.column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data):
        if self.column == ['Kp', 'Ap']:
            data = torch.log1p(data)
            data = (data - CelesTrak_mean_of_log1p) / CelesTrak_std_of_log1p
        elif self.column == ['Kp']:
            data = torch.log1p(data)
            data = (data - CelesTrak_mean_of_log1p[0]) / CelesTrak_std_of_log1p[0] # Kp
        elif self.column == ['Ap']:
            data = torch.log1p(data)
            data = (data - CelesTrak_mean_of_log1p[1]) / CelesTrak_std_of_log1p[1] # Ap
        else:
            raise ValueError(f"Unknown column configuration: {self.column}. Expected ['Kp', 'Ap'], ['Kp'], or ['Ap'].")
        return data
    
    def unnormalize_data(self, data):
        if self.column == ['Kp', 'Ap']:
            data = data * CelesTrak_std_of_log1p + CelesTrak_mean_of_log1p
            data = torch.expm1(data)
        elif self.column == ['Kp']:
            data = data * CelesTrak_std_of_log1p[0] + CelesTrak_mean_of_log1p[0] # Kp
            data = torch.expm1(data)
        elif self.column == ['Ap']:
            data = data * CelesTrak_std_of_log1p[1] + CelesTrak_mean_of_log1p[1] # Ap
            data = torch.expm1(data)
        else:
            raise ValueError(f"Unknown column configuration: {self.column}. Expected ['Kp', 'Ap'], ['Kp'], or ['Ap'].")
        return data
