import torch
import datetime
from src import JPLDGIMDataset
from src import SolarIndexDataset
from src import CelestrakDataset
from src import OMNIDataset
from src.dataset.solar_position_dataset import SolarPositionDataset
from src.dataset.lunar_position_dataset import LunarPositionDataset

import datetime
# Combine all datasets into one

# TODO: dont allow index based indexing, rather convert to timestamp within the __getitem__ of the composite dataset class, then pass in the timestamp
# for indexing within composite dataset, even if some missing data, wont have compounding deletion error
# TODO: Incorporate a date_exclusion, similar to JPLDGIMDataset, This should be handled in composite and keep from getting those dates
# TODO: Handle all of the indexing wrt individual cadences in CompositeDataset, instead of indiducal datasets. This will clean up code of the individuals and 
# make it readable in CompositeDataset to understand how each indivudal dataset is sampled/organized.
class CompositeDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_jpld_dir, 
            celestrak_data_file, 
            solar_index_data_file,
            omniweb_dir,
            date_start=None, 
            date_end=None,
            date_exclusions=None, 
            normalize=True
    ):
        self.cadence = 15
        self.gim_dataset = JPLDGIMDataset(dataset_jpld_dir, date_start=date_start, date_end=date_end, normalize=normalize, date_exclusions=None)
        self.celestrak_dataset = CelestrakDataset(celestrak_data_file, date_start, date_end, normalize, self.cadence)
        self.solar_index_dataset = SolarIndexDataset(solar_index_data_file, date_start, date_end, normalize, self.cadence)
        self.omniweb_dataset = OMNIDataset(omniweb_dir, date_start, date_end, normalize, sampled_cadence=self.cadence)
        self.solar_position_dataset = SolarPositionDataset(date_start, date_end, normalize, self.cadence)
        self.lunar_position_dataset = LunarPositionDataset(date_start, date_end, normalize, self.cadence)

        # Set the date start and end based on the datasets
        if date_start is None or date_end is None: 
            gim_start, gim_end = self.gim_dataset.get_date_range()
            celestrak_start, celestrak_end = self.celestrak_dataset.get_date_range()
            solar_index_start, solar_index_end = self.solar_index_dataset.get_date_range()
            omniweb_start, omniweb_end = self.omniweb_dataset.get_date_range()
            solar_position_start, solar_position_end = self.solar_position_dataset.get_date_range()
            lunar_position_start, lunar_position_end = self.lunar_position_dataset.get_date_range()
            self.composite_start = max(gim_start, celestrak_start, solar_index_start, omniweb_start, solar_position_start, lunar_position_start)
            self.composite_end = min(gim_end, celestrak_end, solar_index_end, omniweb_end, solar_position_end, lunar_position_end)
            
            if self.composite_start > self.composite_end:
                raise ValueError("No overlap found between all datasets.")
            
            self.gim_dataset.set_date_range(self.composite_start, self.composite_end)
            self.celestrak_dataset.set_date_range(self.composite_start, self.composite_end)
            self.solar_index_dataset.set_date_range(self.composite_start, self.composite_end)
            self.omniweb_dataset.set_date_range(self.composite_start, self.composite_end)
            self.solar_position_dataset.set_date_range(self.composite_start, self.composite_end)
            self.lunar_position_dataset.set_date_range(self.composite_start, self.composite_end)

        print(f"Composite Dataset created with the following datasets start and end dates:")
        print(f"  - JPLD GIM Dataset: date range: {self.gim_dataset.date_start} to {self.gim_dataset.date_end}")
        print(f"  - Celestrak Dataset: date range: {self.celestrak_dataset.date_start} to {self.celestrak_dataset.date_end}")
        print(f"  - Solar Index Dataset: date range: {self.solar_index_dataset.date_start} to {self.solar_index_dataset.date_end}")
        print(f"  - OMNIWEB Dataset: date range: {self.omniweb_dataset.date_start} to {self.omniweb_dataset.date_end}")
        print(f"  - Solar Position Dataset: date range: {self.solar_position_dataset.date_start} to {self.solar_position_dataset.date_end}")
        print(f"  - Lunar Position Dataset: date range: {self.lunar_position_dataset.date_start} to {self.lunar_position_dataset.date_end}")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, int):
            minutes = index * self.cadence
            date = self.composite_start + datetime.timedelta(minutes=minutes)

        else:
            raise TypeError("Index must be either a datetime or an integer.")

        gim_data = self.gim_dataset[date]
        celestrak_data = self.celestrak_dataset[date]
        solar_index_data = self.solar_index_dataset[date]
        omniweb_data = self.omniweb_dataset[date]
        solar_position_data = self.solar_position_dataset[date]
        lunar_position_data = self.lunar_position_dataset[date]

        # TODO: decide return format
        return {
            'gim': gim_data,
            'celestrak': celestrak_data,
            'solar_index': solar_index_data,
            'omniweb': omniweb_data,
            'solar_position': solar_position_data,
            'lunar_position': lunar_position_data
        }