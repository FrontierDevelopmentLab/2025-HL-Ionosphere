# This file just sets up the paths so that from all future files using this package you can do:
# from src import CompositeDataset, etc.
from .dataset.gim_dataset import *
from .dataset.omniweb_dataset import *
from .dataset.celestrak_dataset import *
from .dataset.solar_indices_dataset import *
from .dataset.base_datasets import *
# from .dataset.composite_dataset import *