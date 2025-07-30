import src
from datetime import datetime
from pprint import pprint 

# omni_dir = "/mnt/ionosphere-data/omniweb/cleaned/"
# gim_webdataset = "/mnt/disks/disk-main-data-1/data/jpld/webdataset/"
# celestrak_file = "/mnt/ionosphere-data/celestrak/kp_ap_processed_timeseries.csv"
# solar_index_file = "/mnt/ionosphere-data/solar_env_tech_indices/Indices_F10_processed.csv"
omni_dir = "/mnt/disks/disk-main-data-1/data/omniweb/cleaned/"
gim_webdataset = "/mnt/disks/disk-main-data-1/data/jpld/webdataset/"
celestrak_file = "/mnt/disks/disk-main-data-1/data/celestrak/kp_ap_processed_timeseries.csv"
solar_index_file = "/mnt/disks/disk-main-data-1/data/solar_env_tech_indices/Indices_F10_processed.csv"

date_start = datetime(year = 2010, month = 10, day=1, hour=0, minute=0)
date_end = datetime(year = 2024, month = 5, day=1, hour=0, minute=0)
# date_start = datetime(2)
# composite_dataset = src.CompositeDataset([omni_dataset, celestrak_dataset, solar_index_dataset, gim_dataset]) # Dont use composite dataset, use Sequences 

omni_dataset = src.OMNIDataset(file_dir=omni_dir, delta_minutes=15, date_start=date_start, date_end=date_end)
celestrak_dataset = src.CelestrakDataset(file_name=celestrak_file, delta_minutes=15, date_start=date_start, date_end=date_end)
solar_index_dataset = src.SolarIndexDataset(file_name=solar_index_file, delta_minutes=15, date_start=date_start, date_end=date_end)
gim_dataset = src.JPLDGIMDataset(data_dir=gim_webdataset, date_start = date_start, date_end=date_end)


sequence_dataset = src.Sequences([gim_dataset, omni_dataset, celestrak_dataset, solar_index_dataset], delta_minutes=15, sequence_length=5)
print("\nsequence:")
pprint(sequence_dataset[100])
date = datetime(year = 2019, month = 10, day=23, hour=15, minute=5)
# print("omni:", omni_dataset[date])
# print("\ncelestrak:", celestrak_dataset[date])
# print("\nsolar index:", solar_index_dataset[date])
# # print("\ncomposite:", composite_dataset[date])
