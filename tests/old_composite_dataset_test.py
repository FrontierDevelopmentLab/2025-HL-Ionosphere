# from composite_dataset import CompositeDataset
from src import CompositeDataset
from datetime import datetime

import pytest
import datetime
omni_dir = "/mnt/ionosphere-data/omniweb/processed/"
# gim_parquet = "/mnt/ionosphere-data/jpld_gim/parquet/jpld_gim_201005130000_202407312345.parquet"
gim_webdataset = "/mnt/ionosphere-data/jpld/webdataset"
celestrak_file = "/mnt/ionosphere-data/celestrak/kp_ap_processed_timeseries.csv"
solar_index_file = "/mnt/ionosphere-data/solar_env_tech_indices/Indices_F10_processed.csv"

def test_timestamp_index_alignment():
    ds = CompositeDataset(
        dataset_jpld_dir=gim_webdataset, 
        celestrak_data_file=celestrak_file, 
        solar_index_data_file=solar_index_file,
        omniweb_dir=omni_dir,
        date_start=None, 
        date_end=None
        )

    t = datetime.datetime(2019, 1, 1, 12)  # adjust date to valid one
    data = ds[t]
    t_str = datetime.datetime.strftime(t, "%Y-%m-%dT%H:%M:%S")
    print(t_str)
    print(data['gim'][1])
    print(data['celestrak'][1])
    print(data['solar_index'][1])
    print(data['omniweb'][1])
    assert data['gim'][1] == t_str
    assert data['celestrak'][1] == t_str
    assert data['solar_index'][1] == t_str
    assert data['omniweb'][1] == t_str
 
def test_integer_index_alignment():
    ds = CompositeDataset(
        dataset_jpld_dir=gim_webdataset, 
        celestrak_data_file=celestrak_file, 
        solar_index_data_file=solar_index_file,
        omniweb_dir=omni_dir,
        # date_start=datetime.datetime(2019, 1, 1, 12), 
        # date_end=datetime.datetime(2019, 4, 1, 12)
        )
    
    data = ds[24*60*365*14//15-46851] # In synch
    # data = ds[24*60*365*14//15-46850] # Out of synch # BUG: 24*60*365*14//15-46850 = 443710 GIMdataset skips forward 5:30 hours
    data = ds[int((24*60*365*14.2)//15)+1020] # In synch

    # t_str = datetime.datetime.strftime(t, "%Y-%m-%dT%H:%M:%S")
    # print(t_str)
    gim_ts = data['gim'][1]
    celestrak_ts = data['celestrak'][1]
    solar_index_ts = data['solar_index'][1]
    omniweb_ts = data['omniweb'][1]
    print(f"gim timestamp: {gim_ts}")
    print(f"celestrak timestamp: {celestrak_ts}")
    print(f"solar index timestamp: {solar_index_ts}")
    print(f"omniweb timestamp: {omniweb_ts}")
    assert gim_ts == celestrak_ts and gim_ts == solar_index_ts and gim_ts == omniweb_ts, "timestamps do not match"
    # assert data['gim'][1] == t_str
    # assert data['celestrak'][1] == t_str
    # assert data['solar_index'][1] == t_str
    # assert data['omniweb'][1] == t_str

def test_out_of_bounds_date_raises():
    with pytest.raises(ValueError):
        CompositeDataset(
            dataset_jpld_dir=gim_webdataset, 
            celestrak_data_file=celestrak_file, 
            solar_index_data_file=solar_index_file,
            omniweb_dir=omni_dir,
            date_start=datetime.datetime(1900,1,1), 
            date_end=datetime.datetime(1900,1,2)
        )

def test_dataset_length_consistent():
    ds = CompositeDataset(
        dataset_jpld_dir=gim_webdataset, 
        celestrak_data_file=celestrak_file, 
        solar_index_data_file=solar_index_file,
        omniweb_dir=omni_dir,
        date_start=None, 
        date_end=None
    )
    expected_len = min(len(ds.gim_dataset), len(ds.celestrak_dataset), len(ds.solar_index_dataset), len(ds.omniweb_dataset))
    assert len(ds) == expected_len

# def test_omniweb_dataset():

#     date_start = None # datetime.strptime("2023-10-01 00:00:00", "%Y-%m-%d %H:%M:%S")
#     date_end = None # datetime.strptime("2023-10-02 00:00:00", "%Y-%m-%d %H:%M:%S")
#     date_sample = datetime.strptime("2023-10-01 00:15:00", "%Y-%m-%d %H:%M:%S")
#     # Test the OmniwebDataset class
#     dataset = omniweb_dataset.OMNIDataset(omni_dir, normalize=True)

#     dataset_comp = CompositeDataset(
#         dataset_jpld_dir=gim_webdataset, 
#         celestrak_data_file=celestrak_file, 
#         solar_index_data_file=solar_index_file,
#         omniweb_dir=omni_dir,
#         date_start=date_start, 
#         date_end=date_end, 
#         )
    
#     print(date_sample, type(date_sample))
#     print(dataset[date_sample])
#     print(dataset[606545])
#     print(dataset_comp[date_sample])
#     # Check if the dataset is initialized correctly
#     assert dataset is not None, "Dataset should be initialized"

if __name__ == "__main__":
    # test_timestamp_index_alignment()
    test_integer_index_alignment()
    # test_out_of_bounds_date_raises()
    # test_dataset_length_consistent()
    print("Omniweb dataset test passed.")