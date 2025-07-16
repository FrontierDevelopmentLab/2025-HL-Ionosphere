import torch
import os
import numpy as np
import datetime

class GIMDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()

        self.timestamped_files = []
        for file in os.listdir(directory):
            if file.endswith('.npy'):
                self.timestamped_files.append({
                    'file_path': os.path.join(directory, file),
                    'timestamp': self.extract_timestamp(file)
                })

        self.transform = transform

    def extract_timestamp(self, file):
        split_name = file.split("_")
        year_month_day = split_name[2]
        hour_minute = split_name[3]
        year = int(year_month_day[:4])
        month = int(year_month_day[4:6])
        day = int(year_month_day[6:8])
        hour = int(hour_minute[:2])
        minute = int(hour_minute[2:4])
        return datetime.datetime(year, month, day, hour, minute)

    def __len__(self):
        return len(self.timestamped_files)

    def __getitem__(self, idx):
        file_info = self.timestamped_files[idx]
        vtec_map = np.load(file_info['file_path'])
        timestamp = file_info['timestamp']
        if self.transform:
            vtec_map = self.transform(vtec_map)
        return vtec_map, timestamp


