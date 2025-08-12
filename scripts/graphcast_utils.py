import torch
from warnings import warn
import numpy as np

# Pass in batch or dataset.get_sequence_data output to stack_features
def stack_features(
        sequence_features, 
        image_size=(180, 360), 
        batched=True, # False when the batch dimension doesnt exist
    ): 
    """
    S
    Sequence features is a sequence returned from either a dataloader of a Sequences dataset or a singular Sequences dataset sequence, the timestamps are assumed to be removed
    Sequences dataset structure:
    [Tensor[DSet 1 batch]    | Tensor[DSet 2 batch]   | ... | Tensor[DSet k batch]   | List[tuples(timestamps)]] |
    [torch.Size([B, T, F])   | torch.Size([B, T, F])  | ... | torch.Size([B, T, F])  | [T, B]                    |

    Example: (Default format of dataset order will be fist n datasets will be images, and rest of k-n datasets will be tabular)
    | Tensor[JPLD batch]               | Tensor[OMNIWeb batch]   | Tensor[Celestrak batch] | Tensor[Solar Index batch]  | List[tuples(timestamps)] |
    | torch.Size([B, T, C, H, W])      | torch.Size([B, T, F])   | torch.Size([B, T, F])   | torch.Size([B, T, F])      | [T, B]                   |
    | torch.Size([1, 10, 1, 180, 360]) | torch.Size([1, 10, 16]) | torch.Size([1, 10, 2])  | torch.Size([1, 10, 4])     | [10, 1]                  | 
    """

    # if not timestamps_removed:
    #     sequence_features = sequence_features[:-1]

    # assert n_img_datasets > 0, f"n_img_datasets {n_img_datasets} must be greater than 0"
    assert len(sequence_features) > 0, f"sequence_features of length {len(sequence_features)} must not be empty"
    if not isinstance(sequence_features, list) and not isinstance(sequence_features, tuple):
        raise ValueError('Expecting a list or tuple of features')
    c = []
    batch_dim = 0
    if batched:
        batch_dim = 1
    for i, f in enumerate(sequence_features):
        if isinstance(f, np.ndarray):
            f = torch.tensor(f)
        if not isinstance(f, torch.Tensor):
            continue # NOTE, is this what we want to assume, maybe we either dont assume the instance or we make it clear we exect tensors
            # f = torch.tensor(f, dtype=torch.float32)

        if f.ndim == 1 + batch_dim: # single channel time series data [T,] or [B, T]
            warn(f"sequence_feature ({i}) contains {f.ndim} dimensions with batched set to {batched}, " + 
                 f"assuming shape [{"B, T" if batched else "T,"}] and adding C, H, W dimensions. " +
                 f"Current feature shape = {f.shape}") # warning added as this is a somewhat unusual shape to recieve
            f_shape = f.shape
            f = f.view(*f.shape, 1, 1, 1)
            f = f.expand(f_shape + (1,) + image_size) # add channel dim
            f = f.expand(image_size)
            f = f.unsqueeze(0)

        elif f.ndim == 2 + batch_dim: # vector data [T, C] or [B, T, C]
            f_shape = f.shape
            f = f.view(*f.shape, 1, 1)
            f = f.expand(f_shape + image_size)

        elif f.ndim == 3 + batch_dim and f.shape[-2:] == image_size: # vector data [T, H, W] or [B, T, H, W]
            warn(f"sequence_feature ({i}) contains {f.ndim} dimensions with batched set to {batched}, " + 
                 f"assuming shape [{"B, T, H, W" if batched else "T, H, W"}] and adding C dimension. " +
                 f"Current feature shape = {f.shape}") # warning added as this is a somewhat unusual shape to recieve
            f_shape = f.shape
            f = f.view(*f_shape[:-2], 1, *f_shape[-2:])
        else:
            assert f.ndim == 4 + batch_dim and f.shape[-2:] == image_size
        c.append(f)
    c = torch.cat(c, dim=1 + batch_dim)
    
    if not batched:
        c = c.unsqueeze(0)
    return c


def calc_shapes_for_stack_features(seq_dataset_batch, ordered_datasets, context_window, batched=True):
    """
    TODO: maybe rename function, not very clear what its referring to
    ordered_datasets should include the names of all used datasets in the order they will appear in the sequence dataset
    """
    # Calculate the number of image-like features and make sure image-like datasets are before non-image-like datasets in the dataset list
    img_tensor_ndim = 4
    channel_idx = 1

    if batched: # this accounts for the offset caused by the inclusion of the batch dimension
        img_tensor_ndim += 1
        channel_idx += 1

    len_forcing_channel = 0
    forcing_channels = None

    # if 'sunmoon' is in the aux_datasets, we need to find the channel index it begins at
    if 'sunmoon' in ordered_datasets: 
        sunmoon_idx = ordered_datasets.index('sunmoon') #(Removing this, instead passing in JPLD prepended into aux_datasets passed into helper func) + 1 # Add 1 to account for the JPLD dataset at index 0
        sunmoon_channel_idx = 0

    for idx, T in enumerate(seq_dataset_batch):
        if isinstance(T, torch.Tensor):
            if 'sunmoon' in ordered_datasets:
                # Get the index of the final channel before the sunmoon dataset
                if idx < sunmoon_idx:
                    sunmoon_channel_idx += T.shape[channel_idx] 

                # If we are at the sunmoon dataset, we need to get the number of channels in the sunmoon dataset
                elif idx == sunmoon_idx:
                    len_forcing_channel = T.shape[channel_idx] # Get the number of channels in the sunmoon dataset
                    forcing_channels = list(range(sunmoon_channel_idx, sunmoon_channel_idx + len_forcing_channel)) # Get the channel indices for the forcing channels

    # Get the number of channels in the input and compute the number of features
    dummy_batch = stack_features(seq_dataset_batch, batched=batched)
    _, _, n_channels, _, _ = dummy_batch.shape # B, T, C, H, W

    n_feats = context_window * n_channels + len_forcing_channel

    return n_feats, n_channels, forcing_channels

