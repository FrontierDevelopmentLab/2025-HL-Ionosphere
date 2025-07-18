import torch
from src.gim_dataset import JPLDGIMDataset
from src.solar_indices_dataset import SolarIndexDataset
from src.celestrak_dataset import CelestrakDataset

# Combine all datasets into one

class CompositeDatasset(torch.utils.data.Dataset):
    def __init__(
            self, 
            gim_parquet_file, 
            date_start=None, 
            date_end=None, 
            normalize=True
    ):
        
        self.gim_dataset = JPLDGIMDataset(gim_parquet_file, date_start, date_end, normalize)
        self.solar_index_dataset = SolarIndexDataset(gim_parquet_file, date_start, date_end, normalize)
        self.celestrak_dataset = CelestrakDataset(gim_parquet_file, date_start, date_end, normalize)