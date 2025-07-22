import torch
import datetime

from src.dataset.gim_dataset import JPLDGIMDataset
from src.dataset.solar_indices_dataset import SolarIndexDataset
from src.dataset.celestrak_dataset import CelestrakDataset
from src.dataset.omniweb_dataset import OMNIDataset

# TODO: dont allow index based indexing, rather convert to timestamp within the composite dataset class, then pass in the timestamp
# for indexing within composite dataset, even if some missing data, wont have compounding deletion error
# TODO: Incorporate a date_exclusion, similar to JPLDGIMDataset, This should be handled in composite and keep from getting those dates
# TODO: Handle all of the indexing wrt individual cadences in CompositeDataset, instead of indiducal datasets. This will clean up code of the individuals and 
# make it readable in CompositeDataset to understand how each indivudal dataset is sampled/organized.
class CompositeDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            gim_parquet_file, 
            celestrak_data_file, 
            solar_index_data_file,
            omniweb_dir,
            date_start=None, 
            date_end=None,
            date_exclusions=None, # TODO: implement this
            normalize=True
    ):
        # Initialize the indivudal datasets
        self.gim_dataset = JPLDGIMDataset(gim_parquet_file, date_start, date_end, date_exclusions, normalize)
        self.celestrak_dataset = CelestrakDataset(celestrak_data_file, date_start, date_end, normalize)
        self.solar_index_dataset = SolarIndexDataset(solar_index_data_file, date_start, date_end, normalize)
        self.omniweb_dataset = OMNIDataset(omniweb_dir, date_start, date_end, normalize)

        # Set the date start and end based on the datasets
        if date_start is None or date_end is None: 
            gim_start, gim_end = self.gim_dataset.get_date_range()
            celestrak_start, celestrak_end = self.celestrak_dataset.get_date_range()
            solar_index_start, solar_index_end = self.solar_index_dataset.get_date_range()
            omniweb_start, omniweb_end = self.omniweb_dataset.get_date_range()
            composite_start = max(gim_start, celestrak_start, solar_index_start, omniweb_start)
            composite_end = min(gim_end, celestrak_end, solar_index_end, omniweb_end)
            
            if composite_start > composite_end:
                raise ValueError("No overlap found between all datasets.")
            
            self.gim_dataset.set_date_range(composite_start, composite_end)
            self.celestrak_dataset.set_date_range(composite_start, composite_end)
            self.solar_index_dataset.set_date_range(composite_start, composite_end)
            self.omniweb_dataset.set_date_range(composite_start, composite_end)

        print(f"Composite Dataset created with the following datasets start and end dates:")
        print(f"  - JPLD GIM Dataset: date range: {self.gim_dataset.date_start} to {self.gim_dataset.date_end}")
        print(f"  - Celestrak Dataset: date range: {self.celestrak_dataset.date_start} to {self.celestrak_dataset.date_end}")
        print(f"  - Solar Index Dataset: date range: {self.solar_index_dataset.date_start} to {self.solar_index_dataset.date_end}")
        print(f"  - OMNIWEB Dataset: date range: {self.omniweb_dataset.date_start} to {self.omniweb_dataset.date_end}")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # TODO: Handle complex indexing of dates based ont ehir cadences/ints here, rather than in individual dataset files
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