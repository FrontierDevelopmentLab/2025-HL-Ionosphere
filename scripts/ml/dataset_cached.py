import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CachedDataset(Dataset):
    """A wrapper dataset that pre-loads and caches all items from another dataset into memory."""
    def __init__(self, dataset):
        self.dataset = dataset
        print("Pre-loading and caching dataset into memory...")
        self.data = [dataset[i] for i in tqdm(range(len(dataset)), desc="Caching")]
        print("Caching complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]