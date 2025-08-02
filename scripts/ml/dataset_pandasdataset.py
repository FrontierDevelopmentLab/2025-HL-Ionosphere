import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os
import datetime
from tqdm import tqdm
import pandas as pd
from warnings import warn


class PandasDataset(Dataset):
    def __init__(self, name, data_frame, column, delta_minutes, date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None):
        self.name = name
        self.data = data_frame
        self.column = column
        self.delta_minutes = delta_minutes

        print('Delta minutes         : {:,}'.format(self.delta_minutes))
        self.normalize = normalize
        print('Normalize             : {}'.format(self.normalize))
        self.rewind_minutes = rewind_minutes
        print('Rewind minutes        : {:,}'.format(self.rewind_minutes))
        print('Columns               : {}'.format(self.column))
        print('Rows before processing: {:,}'.format(len(self.data)))

        self.data[column] = self.data[column].astype(np.float32)
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.data = self.data.dropna() # This drops ~10% of omniweb, a bit wasteful as it drops entire row at a time but not too bad
        # Get dates available
        self.data['Datetime'] = pd.to_datetime(self.data['Datetime']) # this line wasnt present previously but is necessary if the col contains strings instead of pandas timestamps for the date.to_pydatetime() to run as expected
        self.dates = [date.to_pydatetime() for date in self.data['Datetime']]

        self.dates_set = set(self.dates)
        self.date_start = self.dates[0]
        self.date_end = self.dates[-1]

        # Adjust dates available
        if date_start is not None:
            if isinstance(date_start, str):
                date_start = datetime.datetime.fromisoformat(date_start)
            if (date_start >= self.date_start) and (date_start < self.date_end):
                self.date_start = date_start
            else:
                print('Start date out of range, using default')
        if date_end is not None:
            if isinstance(date_end, str):
                date_end = datetime.datetime.fromisoformat(date_end)
            if (date_end > self.date_start) and (date_end <= self.date_end):
                self.date_end = date_end
            else:
                print('End date out of range, using default')        

        # if not 'CRaTER' in self.name: # Very bad hack, need to fix this for CRaTER # NOTE CRaTER Not relevant for ionosphere datasets
            # if the date of the first row of data after self.date_start does not end in minutes :00, :15, :30, or :45, move forward to the next minute that does
        time_out = 1000
        while True: # Is this hardcoded 15 min value meant to equal the cadence?
            first_row = self.data[self.data['Datetime'] >= self.date_start].iloc[0]
            first_row_date = first_row['Datetime']
            if first_row_date.minute % 15 != 0:
                print('Adjust startdate(old): {}'.format(first_row_date))
                first_row_date = first_row_date + datetime.timedelta(minutes=15 - (first_row_date.minute % 15))
                print('Adjust startdate(new): {}'.format(first_row_date))
                self.date_start = first_row_date
                time_out -= 1
                if time_out == 0:
                    raise RuntimeError('Time out in adjusting start date for {}'.format(self.name))
            else:
                break
        

        # Filter out dates outside the range
        self.data = self.data[(self.data['Datetime'] >=self.date_start) & (self.data['Datetime'] <=self.date_end)]

        # Filter out dates within date_exclusions
        self.date_exclusions = date_exclusions
        if self.date_exclusions is not None:
            print('Date exclusions:')
            for exclusion_date_start, exclusion_date_end in self.date_exclusions:
                print('  {} - {}'.format(exclusion_date_start, exclusion_date_end))
                self.data = self.data[~self.data['Datetime'].between(exclusion_date_start, exclusion_date_end)]

        # Get dates available (redo to make sure things match up)q
        self.dates = [date.to_pydatetime() for date in self.data['Datetime']]
        self.dates_set = set(self.dates)
        self.date_start = self.dates[0]
        self.date_end = self.dates[-1]
        print('Start date            : {}'.format(self.date_start))
        print('End date              : {}'.format(self.date_end))

        print('Rows after processing : {:,}'.format(len(self.data)))
        # self.data.set_index("Datetime", inplace=True) # sets the indexing to be done with datetime

        # print('Memory usage          : {:,} bytes'.format(self.data.memory_usage(deep=True).sum()))



    def normalize_data(self, data):
        raise NotImplementedError('normalize_data not implemented')
    
    def unnormalize_data(self, data):
        raise NotImplementedError('unnormalize_data not implemented')
    
    def __repr__(self):
        return '{} ({} - {})'.format(self.name, self.date_start, self.date_end)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        elif isinstance(index, int): # THIS WILL CAUSE ERRORS WHEN DEALING WITH MULTPLIE DATASETS
            date = self.data.iloc[index]['Datetime'] 
            warn("Should not index dataset by int if aligning multiple datasets, this is error prone if datasets have holes.")
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_data(date)
        return data, date.isoformat()            
   
    def get_data(self, date):
        # if date < self.date_start or date > self.date_end:
        #     raise ValueError('Date ({}) out of range for RadLab ({}; {} - {})'.format(date, self.instrument, self.date_start, self.date_end))

        if date not in self.dates_set:
            print('{} date not found: {}'.format(self.name, date))
            # find the most recent date available before date, in pandas dataframe self.data
            dates_available = self.data[self.data['Datetime'] < date]
            if len(dates_available) > 0:
                date_previous = dates_available.iloc[-1]['Datetime']
                if date - date_previous < datetime.timedelta(minutes=self.rewind_minutes):
                    print('{} rewinding to  : {}'.format(self.name, date_previous))
                    data = self.data[self.data['Datetime'] == date_previous][self.column]
                else:
                    return None
            else:
                return None
        else:
            data = self.data[self.data['Datetime'] == date][self.column]
            if len(data) == 0:
                raise RuntimeError('Should not happen')
        data = torch.tensor(data.values[0], dtype=torch.float32)
        if self.normalize:
            data = self.normalize_data(data)

        return data

    def get_series(self, date_start, date_end, delta_minutes=None, omit_missing=True):
        if delta_minutes is None:
            delta_minutes = self.delta_minutes
        dates = []
        values = []
        date = date_start
        while date <= date_end:
            value = self.get_data(date)
            value_available = True
            if value is None:
                if omit_missing:
                    value_available = False
                else:
                    value = torch.tensor(float('nan'))
            if value_available:
                dates.append(date)
                values.append(value)
            date += datetime.timedelta(minutes=delta_minutes)
        if len(dates) == 0:
            # raise ValueError('{} no data found between {} and {}'.format(self.name, self.instrument, date_start, date_end))
            return None, None
        values = torch.stack(values).flatten()
        return dates, values
    
    @staticmethod
    def fill_to_cadence(df, delta_minutes=15, rewind_time = 50):
        df["Datetime"] = pd.to_datetime(df["Datetime"]) # strs to convert to datetime objs

        df = df.sort_values("Datetime").reset_index(drop=True) # to make sure df is ordered properly
        filled_rows = []
        rewind_time = datetime.timedelta(minutes=rewind_time)

        for i in tqdm(range(1, len(df))):
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]

            t0 = prev_row["Datetime"]
            t1 = curr_row["Datetime"]

            if (t1 - t0) > rewind_time:  # no infilling if gap larger than rewind time
                filled_rows.append(prev_row)
                continue
            # Generate target timestamps at delta_minutes cadence
            # First aligned timestamp after t0
            remainder = (t0.minute % delta_minutes) 
            correction = datetime.timedelta(minutes=(delta_minutes - remainder) % delta_minutes)
            aligned_start = (t0.replace(second=0, microsecond=0) + correction)

            # Create date range between aligned_start and t1
            timestamps = pd.date_range(
                start=aligned_start,
                end=t1,
                freq=f'{delta_minutes}min'
            )

            # Add the original row
            filled_rows.append(prev_row)

            # For each interpolated timestamp, duplicate previous row and set new time
            for ts in timestamps:
                if ts == t0: continue # already added row corresponding to t0 (prev_row)
                if ts < t1:  # Only fill between, not including t1
                    new_row = prev_row.copy()
                    new_row["Datetime"] = ts
                    filled_rows.append(new_row)

        # Append the final row
        filled_rows.append(df.iloc[-1])
        # df_filled
        return pd.DataFrame(filled_rows).reset_index(drop=True)
