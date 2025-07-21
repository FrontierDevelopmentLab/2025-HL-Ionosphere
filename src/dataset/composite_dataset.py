import torch
import datetime
from src.dataset.gim_dataset import JPLDGIMDataset
from src.dataset.solar_indices_dataset import SolarIndexDataset
from src.dataset.celestrak_dataset import CelestrakDataset
from src.dataset.omniweb_dataset import OMNIDataset
# from gim_dataset import JPLDGIMDataset
# from solar_indices_dataset import SolarIndexDataset
# from celestrak_dataset import CelestrakDataset
# from omniweb_dataset import OMNIDataset
import datetime
# Combine all datasets into one

# TODO : Currently if start date isnt set each dataset will ahve a different start date and their integer based indexing will not be synched up 
# with one another, should have getter/setter methods for start / end date and set start / end date as the intersction between all datasets OR 
# need a more intelligent integer index to timestamp conversion within each dataset class 
class CompositeDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            gim_parquet_file, 
            celestrak_data_file, 
            solar_index_data_file,
            omniweb_dir,
            date_start=None, 
            date_end=None, 
            normalize=True
    ):
        
        self.gim_dataset = JPLDGIMDataset(gim_parquet_file, date_start, date_end, normalize)
        self.celestrak_dataset = CelestrakDataset(celestrak_data_file, date_start, date_end, normalize)
        self.solar_index_dataset = SolarIndexDataset(solar_index_data_file, date_start, date_end, normalize)
        self.omniweb_dataset = OMNIDataset(omniweb_dir, date_start, date_end, normalize)

        print(f"Composite Dataset created with the following datasets start and end dates:")
        print(f"  - JPLD GIM Dataset: date range: {self.gim_dataset.date_start} to {self.gim_dataset.date_end}")
        print(f"  - Celestrak Dataset: date range: {self.celestrak_dataset.date_start} to {self.celestrak_dataset.date_end}")
        print(f"  - Solar Index Dataset: date range: {self.solar_index_dataset.date_start} to {self.solar_index_dataset.date_end}")
        print(f"  - OMNIWEB Dataset: date range: {self.omniweb_dataset.date_start} to {self.omniweb_dataset.date_end}")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime) or isinstance(index, int):
            gim_data = self.gim_dataset[index]
            celestrak_data = self.celestrak_dataset[index]
            solar_index_data = self.solar_index_dataset[index]
            omniweb_data = self.omniweb_dataset[index]
        else:
            raise TypeError("Index must be either a datetime or an integer.")

        # TODO: decide return format
        return {
            'gim': gim_data,
            'celestrak': celestrak_data,
            'solar_index': solar_index_data,
            'omniweb': omniweb_data
        }