import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os
import datetime
from tqdm import tqdm
import pandas as pd

class PandasDataset(Dataset):
    def __init__(self, name, data_frame, column, delta_minutes, date_start=None, date_end=None, normalize=True, rewind_minutes=15, date_exclusions=None):
        self.name = name
        self.data = data_frame
        self.column = column
        self.delta_minutes = delta_minutes

        print('Delta minutes        : {:,}'.format(self.delta_minutes))
        self.normalize = normalize
        print('Normalize            : {}'.format(self.normalize))
        self.rewind_minutes = rewind_minutes
        print('Rewind minutes       : {:,}'.format(self.rewind_minutes))
        print(f"column:{column}")
        print(self.data)
        self.data[column] = self.data[column].astype(np.float32)

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(self.data)
        self.data = self.data.dropna() # careful to remove cols that are always nan before hand if they exist since nans are being dropped in pd dataset
        print(self.data)
        
        # Get dates available
        # self.dates = [date.to_pydatetime() for date in self.data['Datetime']] #  NOTE This line seems to be from an old version of pandas potentially also assumes date is a datetime obj? wheraas its a string fix below:
        # New fix to above line of code is given in the below 2 lines
        datetime_series = pd.to_datetime(self.data['Datetime'])  
        self.dates = datetime_series.dt.to_pydatetime().tolist()

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
        print('Start date           : {}'.format(self.date_start))
        print('End date             : {}'.format(self.date_end))

        print('Rows after processing: {:,}'.format(len(self.data)))
        # print('Memory usage         : {:,} bytes'.format(self.data.memory_usage(deep=True).sum()))



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
        elif isinstance(index, int):
            date = self.data.iloc[index]['Datetime']
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
        data = torch.tensor(data.values[0])
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

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime


class CompositeDataset(Dataset): 
    '''Note: dont use composite dataset, rather use sequences'''
    def __init__(self, datasets):
        self._datasets = datasets
        # print("Number of datasets stored in the composite is: {}".format(len(datasets)))
        self._date_start = self._datasets[0].date_start
        self._date_end = self._datasets[0].date_end
        self._delta_minutes = self._datasets[0].delta_minutes
        self._length = len(self._datasets[0])
        for dataset in self._datasets:
            if dataset.date_start != self._date_start:
                raise ValueError(
                    "Expecting all datasets to have the same starting date"
                )
            if dataset.date_end != self._date_end:
                raise ValueError("Expecting all datasets to have the same ending date")
            if dataset.delta_minutes != self._delta_minutes:
                raise ValueError(
                    "Expecting all datasets to have the same delta seconds"
                )
            # if len(dataset) != self._length: # NOTE: this is removed temporarily but lengths can not be the same?
            #     raise ValueError(
            #         "Expecting all datasets to have the same length - synchronize dates or delta_seconds"
            #     )

        print("\nComposite - Length             : {}".format(self._length))
        print("Composite - Start date         : {}".format(self._date_start))
        print("Composite - End date           : {}".format(self._date_end))
        print("Composite - Time delta         : {} minutes".format(self._delta_minutes))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        data = []
        for i, dataset in enumerate(self._datasets):
            # time_start = datetime.datetime.now()
            data.append(dataset[index])
            # time_spent = (datetime.datetime.now() - time_start).total_seconds()
            # print('dataset {} took {} seconds'.format(i, time_spent))
        return tuple(data)

    def get_datasets(self):
        return self._datasets

class UnionDataset(Dataset):
    """ fills gaps by merging all passed in datasets, if a dataset without a gap at indexed time is found, returns its (and only its) value """
    def __init__(self, datasets):
        self.datasets = datasets

        print('\nConcatenated datasets')
        for dataset in self.datasets:
            print('Dataset : {}'.format(dataset))

        # check that there is no overlap in the .dates_set of each dataset
        self.dates_set = set()
        self.date_start = datetime.datetime(9999, 12, 31, 23, 59, 59)
        self.date_end = datetime.datetime(1, 1, 1, 0, 0, 0)
        for dataset in self.datasets:
            for date in dataset.dates_set:
                if date < self.date_start:
                    self.date_start = date
                if date > self.date_end:
                    self.date_end = date
                # if date in self.dates_set:
                #     raise ValueError('Overlap in dates_set between datasets')
                self.dates_set.add(date)

    def __len__(self):
        # return sum([len(dataset) for dataset in self.datasets])
        return len(self.dates_set)

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be datetime.datetime or str (in the format of 2022-11-01T00:01:00)')
        for dataset in self.datasets:
            value, date = dataset[index]
            if value is not None:
                return value, date
        return None, None

class Sequences(Dataset):
    def __init__(self, datasets, delta_minutes=1, sequence_length=10):
        super().__init__()
        self.datasets = datasets
        self.delta_minutes = delta_minutes
        self.sequence_length = sequence_length

        self.date_start = max([dataset.date_start for dataset in self.datasets])
        self.date_end = min([dataset.date_end for dataset in self.datasets])
        if self.date_start > self.date_end:
            raise ValueError('No overlapping date range between datasets')

        print('\nSequences')
        print('Start date              : {}'.format(self.date_start))
        print('End date                : {}'.format(self.date_end))
        print('Delta                   : {} minutes'.format(self.delta_minutes))
        print('Sequence length         : {}'.format(self.sequence_length))
        print('Sequence duration       : {} minutes'.format(self.delta_minutes*self.sequence_length))

        self.sequences = self.find_sequences()
        if len(self.sequences) == 0:
            print('**** No sequences found ****')
        print('Number of sequences     : {:,}'.format(len(self.sequences)))
        if len(self.sequences) > 0:
            print('First sequence          : {}'.format([date.isoformat() for date in self.sequences[0]]))
            print('Last sequence           : {}'.format([date.isoformat() for date in self.sequences[-1]]))

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        # print('constructing sequence')
        sequence = self.sequences[index]
        sequence_data = []
        for dataset in self.datasets:
            data = []
            for i, date in enumerate(sequence):
                if i == 0:
                    # All data is available at the first step in sequence (by construction of sequences by find_sequence)
                    d, _ = dataset[date]
                    data.append(d)
                else:
                    if date in dataset.dates_set:
                        d, _ = dataset[date]
                        data.append(d)
                    else:
                        data.append(data[i-1])
            data = torch.stack(data)
            sequence_data.append(data)
        sequence_data.append([date.isoformat() for date in sequence])
        # print('done constructing sequence')
        return tuple(sequence_data)


    def find_sequences(self):
        sequences = []
        sequence_start = self.date_start
        while sequence_start <= self.date_end - datetime.timedelta(minutes=(self.sequence_length-1)*self.delta_minutes):
            # New sequence
            sequence = []
            sequence_available = True
            for i in range(self.sequence_length):
                date = sequence_start + datetime.timedelta(minutes=i*self.delta_minutes)
                if i == 0:
                    for dataset in self.datasets:
                        if date not in dataset.dates_set:
                            sequence_available = False
                            break
                if not sequence_available:
                    break
                sequence.append(date)
            if sequence_available:
                sequences.append(sequence)
            # Move to next sequence
            sequence_start += datetime.timedelta(minutes=self.delta_minutes)
        return sequences
