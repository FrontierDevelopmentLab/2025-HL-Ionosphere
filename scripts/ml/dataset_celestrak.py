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


# celestrak_file = "/mnt/ionosphere-data/celestrak/kp_ap_processed_timeseries.csv"
class CelesTrak(PandasDataset):
    def __init__(self, file_name, date_start=None, date_end=None, normalize=True, rewind_minutes=180, date_exclusions=None, delta_minutes=15): # 180 minutes rewind defualt matching dataset cadence (NOTE: what is a good max value for rewind_minutes?)
        print('CelesTrak')
        print('File                 : {}'.format(file_name))

        data = pd.read_csv(file_name)
        # data['Datetime'] = pd.to_datetime(data['Datetime'])
        # data = data.sort_values(by='Datetime')
        print('Rows                 : {:,}'.format(len(data)))

        self.column = ["Kp", "Ap"]

        df_mean = data[self.column].mean()
        df_std = data[self.column].mean()
        self.col_means = torch.tensor(np.array(df_mean))
        self.col_std = torch.tensor(np.array(df_std))

        stem = Path(file_name).stem
        new_stem = f"{stem}_deltamin_{delta_minutes}_rewind_{rewind_minutes}" 
        cadence_matched_fname = Path(file_name).with_stem(new_stem)
        if cadence_matched_fname.exists():
            data = pd.read_csv(cadence_matched_fname)
        else:
            data = PandasDataset.fill_to_cadence(data, delta_minutes=delta_minutes, rewind_time=rewind_minutes)
            data.to_csv(cadence_matched_fname) # the fill to cadence can take a while, so cache file

        # Remove outliers based on quantiles, # NOTE: is this something we care about in this dataset? 
        # q_low = data[self.column].quantile(0.001)
        # q_hi  = data[self.column].quantile(0.999)
        # data = data[(data[self.column] < q_hi) & (data[self.column] > q_low)] # quantile stuff breaking code, we can reintroduce later

        super().__init__('CelesTrak', data, self.column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    # NOTE: what is the reason these methods were kept as instance methods but also passing the data in as an argument in the RSTNRadio dataset (or the other dattasets as well)?
    def normalize_data(self, data): 
        return (data - self.col_means) / self.col_std
        # for col in self.column:
        #     data[col] = (data[col] - self.df_mean[col]) / self.df_std[col]
        # return data
    
    def unnormalize_data(self, data):
        # for col in self.column:
        #     data[col] = data[col] * self.df_std[col] + self.df_mean[col]
        return data * self.col_std + self.col_means
        # return data
