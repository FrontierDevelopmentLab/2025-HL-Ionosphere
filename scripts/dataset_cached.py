import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import shutil


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


def format_bytes(num_bytes):
    """Format a number of bytes as a human-readable string (MiB, GiB, TiB, etc)."""
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    

class CachedBatchDataset(Dataset):
    """
    A PyTorch Dataset that creates, caches, and serves entire batches of data.

    On first use, it creates a DataLoader internally to batch data from the
    source_dataset, saves these batches to disk, and then reads from the
    cache in all subsequent epochs.

    Args:
        source_dataset (Dataset): The original dataset to pull data from.
        cache_dir (str): The directory to store and read cached batches.
        batch_size (int): The batch size to use when creating batches.
        collate_fn (callable, optional): The collate function for the internal DataLoader.
        force_recache (bool): If True, deletes any existing cache and rebuilds it.
    """
    def __init__(self, source_dataset, cache_dir, batch_size, collate_fn=None, force_recache=False):
        self.source_dataset = source_dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.batch_files = []

        if force_recache and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

        print('\n***CachedBatchDataset***')
        # Check if the cache needs to be built.
        if not os.path.exists(self.cache_dir) or not os.listdir(self.cache_dir):
            print(f"Cache not found. Building cache: {self.cache_dir}")
            self._build_cache()
        else:
            print(f"Existing cache     : {self.cache_dir}")
            self._load_from_cache()
        print()

    def _build_cache(self):
        """Creates batches with an internal DataLoader and saves them to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create a temporary, single-worker loader to build the cache safely.
        # shuffle=False is critical for a deterministic cache.
        caching_loader = DataLoader(
            self.source_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,  # MUST be False to create a deterministic cache
            num_workers=0   # MUST be 0 to prevent race conditions while writing files
        )

        num_batches = len(caching_loader)
        # Determine the padding width from the total number of batches
        pad_width = len(str(num_batches - 1))

        for i, batch in enumerate(tqdm(caching_loader, desc="Caching batches")):
            filepath = os.path.join(self.cache_dir, f"batch_{i:0{pad_width}d}.pt")
            torch.save(batch, filepath)
            self.batch_files.append(filepath)

            # After saving the first batch, estimate total size and check disk space.
            if i == 0:
                first_batch_size = os.path.getsize(filepath)
                estimated_total_size = first_batch_size * num_batches
                _, _, free_space = shutil.disk_usage(self.cache_dir)

                print('Size of first batch          : {:.2f} MiB'.format(first_batch_size / (1024**2)))
                print('Number of batches            : {:,}'.format(num_batches))
                print('Estimated total size of cache:', format_bytes(estimated_total_size))

                if estimated_total_size > free_space:
                    # Clean up the partially created cache before raising
                    shutil.rmtree(self.cache_dir)
                    raise OSError(
                        f"Insufficient disk space for cache. "
                        f"Estimated required space: {estimated_total_size / (1024**3):.2f} GB. "
                        f"Available space: {free_space / (1024**3):.2f} GB."
                    )

    def _load_from_cache(self):
        """Loads the list of file paths from the existing cache directory."""
        self.batch_files = sorted([
            os.path.join(self.cache_dir, f) 
            for f in os.listdir(self.cache_dir) if f.endswith(".pt")
        ])
        total_size = sum(os.path.getsize(f) for f in self.batch_files)
        print(f"Number of batches  : {len(self.batch_files):,}")
        print(f"Total size of cache: {format_bytes(total_size)}")
        print(f"Size of each batch : {format_bytes(total_size / len(self.batch_files))}")

    def __len__(self):
        # The length is the number of cached batches.
        return len(self.batch_files)

    def __getitem__(self, idx):
        # Fetches a pre-made batch from disk.
        return torch.load(self.batch_files[idx])