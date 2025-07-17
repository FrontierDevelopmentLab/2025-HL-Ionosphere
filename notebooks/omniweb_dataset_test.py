import src.omniweb_dataset
from datetime import datetime

def test_omniweb_dataset():
    omni_dir = "/mnt/ionosphere-data/omniweb/processed/"

    date = datetime.strptime("2023-10-01 00:14:00", "%Y-%m-%d %H:%M:%S")
    # Test the OmniwebDataset class
    dataset = src.omniweb_dataset.OMNIDataset(omni_dir, normalize=False)
    print(date, type(date))
    print(dataset.__getitem__(date))
    
    # Check if the dataset is initialized correctly
    assert dataset is not None, "Dataset should be initialized"
    
if __name__ == "__main__":
    test_omniweb_dataset()
    print("Omniweb dataset test passed.")