import torch
from torch.utils.data import Dataset
import datetime


class Sequences(Dataset):
    def __init__(self, datasets, delta_minutes=15, sequence_length=4):
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
        sequence_data = self.get_sequence_data(sequence)
        return sequence_data

    def get_sequence_data(self, sequence): # sequence is a list of datetime objects
        if sequence[0] < self.date_start or sequence[-1] > self.date_end:
            raise ValueError('Sequence dates must be within the dataset date range ({}) - ({}), but got a request ({}) - ({})'.format(self.date_start, self.date_end, sequence[0], sequence[-1]))

        sequence_data = []
        for dataset in self.datasets:
            data = []
            for i, date in enumerate(sequence):
                if i == 0:
                    # Data from all datasets must be available at the first step in sequence
                    if date not in dataset.dates_set:
                        raise ValueError('First date of the sequence {} not found in dataset {}'.format(date, dataset.name))
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
        return tuple(sequence_data)

# Output of DataLoader with Sequences dataset is a tuple of:
# (Dset1_batch, Dset2_batch, ..., timestamps_batch)
# Each Dset_batch is a tensor of shape [B, T, F] where F are the features for each dataset
# timestamps_batch is a list of timestamps for each sequence step, btached- i.e. [(timestamp0_batch0, timestamp0_batch1, ...), (timestamp1_batch0, timestamp1_batch1, ...), ...]
# [[B, T, C, H, W], [B, T, F], [(B)] - where B is batch size, T is sequence length, C is number of channels, H is height, W is width, F is number of features]

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