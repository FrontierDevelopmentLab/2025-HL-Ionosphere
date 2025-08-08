import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import shutil
import json
import io
try:
    import lz4.frame
except ImportError:
    lz4 = None

from util import format_bytes


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
        num_workers (int): The number of subprocesses to use for data loading during cache creation. Defaults to 0 (main process).
        force_recache (bool): If True, deletes any existing cache and rebuilds it.
        compression (str, optional): The compression to use. 'lz4' or None. Defaults to 'lz4'.
    """
    def __init__(self, source_dataset, cache_dir, batch_size, collate_fn=None, num_workers_to_build_cache=0, force_recache=False, compression='lz4'):
        self.source_dataset = source_dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers_to_build_cache = num_workers_to_build_cache
        self.compression = compression
        if self.compression == 'lz4' and lz4 is None:
            raise ImportError("lz4 compression is selected, but the 'lz4' package is not installed. Please run 'pip install lz4'.")
        self.batch_files = []
        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')

        if force_recache and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

        print('\n***CachedBatchDataset***')
        if self._is_cache_valid():
            print(f"Valid cache found  : {self.cache_dir}")
            self._load_from_cache()
        else:
            if os.path.exists(self.cache_dir):
                print(f"Incomplete or invalid cache found. Deleting and rebuilding: {self.cache_dir}")
                shutil.rmtree(self.cache_dir)
            else:
                print(f"Cache not found. Building cache: {self.cache_dir}")
            self._build_cache()
        print()

    def _is_cache_valid(self):
        """Checks if the cache is complete and valid."""
        if not os.path.exists(self.metadata_path):
            return False
        
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False # Corrupted metadata file

        if 'num_batches' not in metadata:
            return False

        expected_batches = metadata['num_batches']
        
        # Count actual .pt files
        try:
            actual_batches = len([f for f in os.listdir(self.cache_dir) if f.endswith(".pt")])
        except FileNotFoundError:
            return False

        return actual_batches == expected_batches

    def _build_cache(self):
        """Creates batches with an internal DataLoader and saves them to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create a temporary loader to build the cache.
        caching_loader = DataLoader(
            self.source_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,  # MUST be False to create a deterministic cache
            num_workers=self.num_workers_to_build_cache
        )

        num_batches = len(caching_loader)
        pad_width = len(str(num_batches - 1))

        for i, batch in enumerate(tqdm(caching_loader, desc="Caching batches")):
            filepath = os.path.join(self.cache_dir, f"batch_{i:0{pad_width}d}.pt")
            
            # Serialize to an in-memory buffer
            buffer = io.BytesIO()
            torch.save(batch, buffer)
            buffer.seek(0)
            data = buffer.read()

            # Compress if enabled
            if self.compression == 'lz4':
                data = lz4.frame.compress(data)
            
            # Write to disk
            with open(filepath, 'wb') as f:
                f.write(data)

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
                    shutil.rmtree(self.cache_dir)
                    raise OSError(
                        f"Insufficient disk space for cache. "
                        f"Estimated required space: {format_bytes(estimated_total_size)}. "
                        f"Available space: {format_bytes(free_space)}."
                    )
        
        # Write metadata file only after all batches are successfully saved.
        metadata = {'num_batches': len(self.batch_files)}
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

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
        filepath = self.batch_files[idx]
        with open(filepath, 'rb') as f:
            data = f.read()

        # Decompress if enabled
        if self.compression == 'lz4':
            data = lz4.frame.decompress(data)
        
        # Load tensor from the in-memory buffer
        buffer = io.BytesIO(data)
        return torch.load(buffer)