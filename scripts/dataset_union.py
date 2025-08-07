import torch
from torch.utils.data import Dataset
import datetime


# The intended use case is to produce a union of multiple dataset instances of the same type
# e.g. multiple JPLD datasets with different date ranges
class Union(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        print('\nUnion of datasets')
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
                if date in self.dates_set:
                    print('Warning: Overlap in dates_set between datasets in the union')
                self.dates_set.add(date)
        self.dates = sorted(self.dates_set)
        self.name = 'Union'

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        elif isinstance(index, int):
            if index < 0 or index >= len(self.dates):
                raise IndexError("Index out of range for the dataset.")
            date = self.dates[index]
        else:
            raise ValueError('Expecting index to be datetime.datetime or str (in the format of 2022-11-01T00:01:00), but got {}'.format(type(index)))
        for dataset in self.datasets:
            if date in dataset.dates_set:
                value, date = dataset[date]
                return value, date
        return None, None