import torch
from torch.utils.data import DataLoader, Dataset
import os
import shutil
import json
import uuid
import time
import io
from tqdm import tqdm

try:
    import lz4.frame
except ImportError:
    lz4 = None

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from util import format_bytes

class _CachingIterator:
    """
    An internal iterator used for the first epoch to build the cache.
    It wraps a standard DataLoader, and for each batch it yields, it saves
    a copy to the cache directory as a side-effect in the main process.
    """
    def __init__(self, parent_loader):
        self.parent_loader = parent_loader
        self.internal_loader = DataLoader(
            parent_loader.dataset,
            batch_size=parent_loader.batch_size,
            collate_fn=parent_loader.collate_fn,
            num_workers=parent_loader.num_workers,
            shuffle=parent_loader.shuffle,
            pin_memory=parent_loader.pin_memory,
            persistent_workers=parent_loader.persistent_workers if parent_loader.num_workers > 0 else False,
            prefetch_factor=parent_loader.prefetch_factor if parent_loader.num_workers > 0 else None
        )
        desc = f"Building cache ({parent_loader.name})" if parent_loader.name else "Building cache"
        self.pbar = tqdm(total=len(self.internal_loader), desc=desc)
        self.internal_iter = iter(self.internal_loader)
        self.num_batches_processed = 0
        self.printed_disk_stats = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.internal_iter)
            
            filename = f"batch_{uuid.uuid4().hex}.pt"
            filepath = os.path.join(self.parent_loader.cache_dir, filename)
            
            buffer = io.BytesIO()
            torch.save(batch, buffer)
            buffer.seek(0)
            data = buffer.read()

            if self.parent_loader.compression == 'lz4':
                data = lz4.frame.compress(data)
            elif self.parent_loader.compression == 'zstd':
                cctx = zstd.ZstdCompressor()
                data = cctx.compress(data)
            
            with open(filepath, 'wb') as f:
                f.write(data)
            
            self.num_batches_processed += 1
            
            if not self.printed_disk_stats:
                self._estimate_and_check_disk_space(filepath)
                self.printed_disk_stats = True

            self.pbar.update(1)
            return batch
        except StopIteration:
            self.pbar.close()
            self._finalize_cache()
            raise StopIteration

    def _estimate_and_check_disk_space(self, first_filepath):
        first_batch_size = os.path.getsize(first_filepath)
        num_batches = len(self.internal_loader)
        estimated_total_size = first_batch_size * num_batches
        _, _, free_space = shutil.disk_usage(self.parent_loader.cache_dir)

        # Use tqdm.write to print messages without breaking the progress bar
        tqdm.write('CachedDataLoader')
        if self.parent_loader.name:
            tqdm.write(f"Name                : {self.parent_loader.name}")
        tqdm.write(f"Cache directory     : {self.parent_loader.cache_dir}")
        tqdm.write(f"Size of first batch : {format_bytes(first_batch_size)}")
        tqdm.write(f"Number of batches   : {num_batches:,}")
        tqdm.write(f"Estimated total size: {format_bytes(estimated_total_size)}")
        tqdm.write(f"Available disk space: {format_bytes(free_space)}")

        if estimated_total_size > free_space:
            shutil.rmtree(self.parent_loader.cache_dir)
            raise OSError(
                f"Insufficient disk space for cache. "
                f"Estimated required space: {format_bytes(estimated_total_size)}. "
                f"Available space: {format_bytes(free_space)}."
            )

    def _finalize_cache(self):
        """Write the metadata file to mark the cache as complete and valid."""
        metadata = {'num_batches': self.num_batches_processed}
        with open(self.parent_loader.metadata_path, 'w') as f:
            json.dump(metadata, f)
        self.parent_loader.is_cache_valid = True
        tqdm.write("Cache building complete.")


class _CacheFileDataset(Dataset):
    """A simple dataset that loads individual pre-batched files from a cache."""
    def __init__(self, files, compression):
        self.files = files
        self.compression = compression

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        with open(filepath, 'rb') as f:
            data = f.read()

        if self.compression == 'lz4':
            data = lz4.frame.decompress(data)
        elif self.compression == 'zstd':
            dctx = zstd.ZstdDecompressor()
            data = dctx.decompress(data)
        
        buffer = io.BytesIO(data)
        # The loaded object is already a batch
        return torch.load(buffer, weights_only=False)


class CachedDataLoader:
    """
    A DataLoader that builds a cache of batches on-the-fly during the first
    epoch, and reads from that cache in all subsequent epochs. This allows for
    non-blocking, multi-process data loading and caching.
    """
    def __init__(self, dataset, batch_size, cache_dir, num_workers=0, collate_fn=None, shuffle=True, force_recache=False, compression='zstd', pin_memory=False, persistent_workers=False, prefetch_factor=None, name=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.compression = compression
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.name = name
        self.cached_files = [] # Attribute to hold the list of cached files
        
        if self.compression == 'lz4' and lz4 is None:
            raise ImportError("lz4 compression is selected, but the 'lz4' package is not installed. Please run 'pip install lz4'.")
        if self.compression == 'zstd' and zstd is None:
            raise ImportError("zstd compression is selected, but the 'zstandard' package is not installed. Please run 'pip install zstandard'.")

        self.metadata_path = os.path.join(self.cache_dir, 'metadata.json')
        self.num_source_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        if force_recache and os.path.exists(self.cache_dir):
            print(f"Forcing recache. Deleting old cache: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print('\nCachedDataLoader')
        if self.name:
            print(f"Name                : {self.name}")
        self.is_cache_valid = self._check_cache_validity()
        if self.is_cache_valid:
            print(f"Using existing cache: {self.cache_dir}")
            # Scan the directory and cache the file list once
            self.cached_files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith(".pt")]
            self._print_cache_stats()
            print()
        else:
            print(f"Cache directory     : {self.cache_dir}")
            print('Cache not found or invalid. Will build on-the-fly during first epoch.')

    def _check_cache_validity(self):
        if not os.path.exists(self.metadata_path):
            return False
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            expected_batches = metadata.get('num_batches')
            if not expected_batches: return False
                
            file_count = len([f for f in os.listdir(self.cache_dir) if f.endswith(".pt")])
            if file_count != expected_batches:
                print(f"Cache file count mismatch (expected {expected_batches}, found {file_count}). Rebuilding.")
                return False

        except (IOError, json.JSONDecodeError):
            print("Corrupted cache metadata. Rebuilding.")
            return False
        
        return True

    def _print_cache_stats(self):
        # Use the cached file list for stats
        if not self.cached_files:
            return
        total_size = sum(os.path.getsize(f) for f in self.cached_files)
        num_files = len(self.cached_files)
        print(f"Number of batches   : {num_files:,}")
        print(f"Size of each batch  : {format_bytes(total_size / num_files)}")
        print(f"Total size of cache : {format_bytes(total_size)}")

    def __iter__(self):
        if self.is_cache_valid:
            # If the file list hasn't been populated yet (i.e., this is the first
            # epoch after the cache was just built), scan the directory now.
            if not self.cached_files:
                self.cached_files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith(".pt")]

            # The cache is valid, create a fast dataset and loader to read from it.
            # This re-uses the standard DataLoader for multi-worker performance.
            cache_dataset = _CacheFileDataset(self.cached_files, self.compression)
            
            # When reading from cache, each "sample" is a full batch, so batch_size must be 1
            # and collate_fn must be a simple pass-through.
            return iter(DataLoader(
                cache_dataset,
                batch_size=1, # Each file is already a batch
                collate_fn=lambda x: x[0], # Unpack the single-item list
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
            ))
        else:
            # The cache is not valid, return the iterator that builds it.
            return _CachingIterator(self)

    def __len__(self):
        if self.is_cache_valid:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)['num_batches']
        return self.num_source_batches