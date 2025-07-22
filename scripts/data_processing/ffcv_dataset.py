import src 
import datetime
import os

# https://github.com/libffcv/ffcv 
# TODO for JPLD GIM dataset, use the parquet file directly

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
my_dataset = src.composite_dataset.CompositeDataset(
            gim_parquet_file=gim_data,
            celestrak_data_file=gim_data,
write_path = '/output/path/for/converted/ds.beton'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(max_resolution=256),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(my_dataset)