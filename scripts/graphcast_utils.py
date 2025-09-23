import torch
from warnings import warn
import numpy as np
import datetime

# For sunlocking of batched data
import skyfield.api

EPH = skyfield.api.load('de421.bsp')
SUN_BODY = EPH['sun']
EARTH_BODY = EPH['earth']
TS = skyfield.api.load.timescale()


# Pass in batch or dataset.get_sequence_data output to stack_features
def stack_features(
        sequence_features, 
        image_size=(180, 360), 
        batched=True, # False when the batch dimension doesnt exist
    ): 
    """
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
    image_indices = []

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

            image_indices.append(i)
        else:
            assert f.ndim == 4 + batch_dim and f.shape[-2:] == image_size
        
        c.append(f)

    c = torch.cat(c, dim=1 + batch_dim)
    
    if not batched:
        c = c.unsqueeze(0)

    return c, image_indices


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
    dummy_batch, _ = stack_features(seq_dataset_batch, batched=batched)
    _, _, n_channels, _, _ = dummy_batch.shape # B, T, C, H, W

    n_feats = context_window * n_channels + len_forcing_channel

    return n_feats, n_channels, forcing_channels

def sunlock_features(stacked_features, subsolar_lats, subsolar_lons, quasidipole_lat=None, quasidipole_lon=None, image_indices=None, latitude_lock=False):
    """ 
    Convert stacked_features to be sunlocked. Assumes longitude-locking to the subsolar point, but can
    optionally also latitude-lock to the subsolar point and the quasdipole of the Earth (geomagnetic coordinates).

    Inputs:
        stacked_features: features of shape (B, T, C, H, W) that are stacked along the channel dimension. 
        subsolar_lats: latitude of the point directly under the Sun of shape (B, T).
        subsolar_lons: longitude of the point directly under the Sun of shape (B, T).
        quasidipole_lat: map of quasidipole latitudes (H, W) in degrees, same grid as stacked_features.
                        If None, no quasidipole shift is applied.
        quasidipole_lon: map of quasidipole longitudes (H, W) in degrees, same grid as stacked_features
                        If None, no quasidipole shift is applied.
        image_indices: list of indices of image-like features in stacked_features to be sunlocked.
                       If None, all features are assumed to be image-like and will be sunlocked.
        latitude_lock: if True, also latitude-lock to the subsolar latitude
    ------------------------------------------------------------------------------
    Outputs:
        stacked_features: sunlocked features of shape (B, T, C, H, W)
    ------------------------------------------------------------------------------
    """
    assert stacked_features.ndim == 5, f"stacked_features must have 5 dimensions (B, T, C, H, W), got {stacked_features.ndim}"
    if (quasidipole_lat is None) != (quasidipole_lon is None):
        raise ValueError("Both quasidipole_lat and quasidipole_lon must be provided, or neither.")
    
    # Get shifts in pixels (float)
    lon_shifts = (subsolar_lons / 360. * stacked_features.shape[4]) + 60 # shift in pixels, (B, T)

    if latitude_lock:
        lat_shifts = (subsolar_lats / 180. * stacked_features.shape[3]) # shift in pixels, (B, T) # TODO : Double check that the negative sign is correct for lat shift

    if image_indices is None:
        image_indices = range(stacked_features.shape[2])

    # Shift each image feature by the subsolar longitude
    stacked_features[:, :, image_indices] = circular_shift(stacked_features[:, :, image_indices], lon_shifts, dim=-1)
    # If latitude_lock is True, shift each image feature by the subsolar latitude
    if latitude_lock:
        stacked_features[:, :, image_indices] = circular_shift(stacked_features[:, :, image_indices], lat_shifts, dim=-2)

    if quasidipole_lon is not None:
        raise NotImplementedError("Quasidipole shifting is not implemented yet.")
        # Shift the quasdipole maps by the subsolar point first
        quasidipole_lon = circular_shift(quasidipole_lon, lon_shifts, dim=-1)

        if latitude_lock:
            quasidipole_lat = circular_shift(quasidipole_lat, lat_shifts, dim=-2)

        # TODO: Map features using the quasidipole maps. notebooks/magnetic_sunlock_testing.ipynb has some code to do this, but it is slow.

    return stacked_features


def get_subsolar_points(grid_nodes, timestamps, batched=True):
    """ 
    Calculate the subsolar point (latitude and longitude) for each timestamp in the batch.
    Inputs:
        grid_nodes: tensor of shape (B, T, C, H, W) to get the batch size and time steps.
        timestamps: list of list of timestamps of shape (T, B) in ISO format strings.
        batched: if True, grid_nodes has a batch dimension.
    Outputs:
        sub_lats: tensor of shape (B, T) with the latitude of the subsolar point in degrees.
        sub_lons: tensor of shape (B, T) with the longitude of the subsolar point in degrees.
    ------------------------------------------------------------------------------
    """
    B, T, C, H, W = grid_nodes.shape # note that even if batched=False, grid_nodes will still have a batch dimension of 1 since it is passed through stack_features already

    sub_lats = torch.empty((B, T))
    sub_lons = torch.empty((B, T))
    for b_idx in range(B):
        for t_idx in range(T):
            if batched:
                ts = timestamps[t_idx][b_idx]
            else:
                ts = timestamps[t_idx]
                
            utc_dt = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
            # print(utc_dt)
            celestial_body = SUN_BODY 
            t = TS.from_datetime(utc_dt)

            astrometric = EARTH_BODY.at(t).observe(celestial_body)
            subpoint = skyfield.api.wgs84.subpoint_of(astrometric)

            sub_lats[b_idx, t_idx] = subpoint.latitude.degrees
            sub_lons[b_idx, t_idx] = subpoint.longitude.degrees
    return sub_lats, sub_lons
    

def circular_shift(batch, shift, dim=-1):
    """
    Circularly shift a tensor along one axis (default: longitude, axis=-1)
    with wraparound and linear interpolation. Assumes 5D tensor with shape [B, T, C, H, W].

    shift > 0 shifts left, shift < 0 shifts right.

    Inputs:
        batch: torch.Tensor with shape [B, T, C, H, W]
        shift: torch.Tensor with shape [B, T]
        dim: dimension to shift along (default: -1, longitude axis)
    ------------------------------------------------------------------------------
    Outputs:
        shifted batch: torch.Tensor with shape [B, T, C, H, W]
    ------------------------------------------------------------------------------
    """
    if dim < 0:
        dim = batch.ndim + dim

    B, T, C, H, W = batch.shape
    n = batch.shape[dim]

    # Prepare indices
    indices = torch.arange(n, device=batch.device, dtype=batch.dtype)  # [n]
    indices = indices.view(1, 1, 1, 1, n)  # shape for broadcasting
    # Broadcast shift to shape [B, T, 1, 1, 1] (or appropriate)
    shift_shape = [B, T] + [1] * (batch.ndim - 2)
    shift = shift.view(*shift_shape)
    shifted_indices = (indices + shift) % n  # shape [B, T, 1, 1, n] (if dim=-1)
    
    # as the shifts can and will be floats (subpixel shifting), i0 and i1 are the floor and ciel fo the shift 
    # eg if shift = 3.2, i0 = 3, i1 = 4
    i0 = torch.floor(shifted_indices).long()
    i1 = (i0 + 1) % n
    frac = shifted_indices - i0 # frac is the subpixel amount to shift so if shift = 3.2, frac = 0.2

    # Prepare gather indices
    # Expand i0/i1 to match batch shape
    expand_shape = list(batch.shape)
    expand_shape[dim] = n
    i0 = i0.expand(*expand_shape)
    i1 = i1.expand(*expand_shape)
    frac = frac.expand(*expand_shape)

    # Gather along the shifting dimension
    img_i0 = torch.gather(batch, dim, i0) # the image(s) shifted by i0 pixels
    img_i1 = torch.gather(batch, dim, i1) # the image(s) shifted by i1 pixels

    # this produces a weighted average between the two integer shifts weighted by the subpixel shift (frac)
    # so its a linear interpolation between the two integer shifts to get the subpixel shift.
    shifted = (1 - frac) * img_i0 + frac * img_i1 

    return shifted


# def circular_shift(batch, shift, dim=-1): NOTE: Our broken code. Shapes get through correctly but the output is incorrect when plotted
#     # TODO: fix this function. Shapes are correct
#     """
#     Circularly shift a tensor along one axis (default: longitude, axis=-1)
#     with wraparound and linear interpolation. Assumes 5D tensor with shape [B, T, C, H, W].

#     shift > 0 shifts left, shift < 0 shifts right.

#     Inputs:
#         batch: torch.Tensor with shape [B, T, C, H, W]
#         shift: torch.Tensor with shape [B, T]
#         dim: dimension to shift along (default: -1, longitude axis)
#     ------------------------------------------------------------------------------
#     Outputs:
#         shifted batch: torch.Tensor with shape [B, T, C, H, W]
#     ------------------------------------------------------------------------------
#     """

#     # Old code for non-batched shift:
#     # idx = (torch.arange(n, device=img.device, dtype=img.dtype) + shift) % n
#     # i0 = torch.floor(idx).long()
#     # i1 = (i0 + 1) % n
#     # frac = idx - i0

#     # Broadcast fractions along the other axis
#     # shape = [1] * img.ndim
#     # shape[dim] = n
#     # frac = frac.reshape(shape)

#     # img_i0 = torch.index_select(img, axis, i0)
#     # img_i1 = torch.index_select(img, axis, i1)
#     # return (1 - frac) * img_i0 + frac * img_i1

#     # New code for batched shift:
#     if dim < 0:
#         abs_dim = batch.ndim + dim

#     B, T, C, H, W = batch.shape
#     n = batch.shape[dim]

#     indices = torch.arange(n, device=batch.device, dtype=batch.dtype) # [n]
#     indices = indices.view(1, 1, n) # [1, 1, n]
#     indices = indices.expand(B, T, n) # [B, T, n]
#     shifted_indices = (indices + shift.unsqueeze(-1)) % n  # shape [B, T, n]

#     i0 = torch.floor(shifted_indices).long()
#     i1 = (i0 + 1) % n
#     frac = shifted_indices - i0 # shape [B, T, n]

#     # Flatten
#     i0 = i0.view(B * T * n)
#     i1 = i1.view(B * T * n)
#     frac = frac.view(B * T * n)
    
#     # Permute batch to put the shifting dimension last, then flatten B and T
#     rest_of_dims = list(range(2, len(batch.shape)))
#     rest_of_dims.remove(abs_dim)
#     batch_permuted = batch.permute(*rest_of_dims, 0, 1, abs_dim) # shape [B*T, C*H*W]
#     batch_flattened = batch_permuted.reshape(*batch_permuted.shape[:2], B*T*n) # shape [C,(H or W), B*T*n]

#     # Select and interpolate
#     img_i0 = torch.index_select(batch_flattened, -1, i0)
#     img_i1 = torch.index_select(batch_flattened, -1, i1)
#     shifted_img = (1 - frac) * img_i0 + frac * img_i1

#     # Reshape and permute back to original order
#     shifted_img_reshaped = shifted_img.reshape(*batch_permuted.shape) # shape [C,(H or W), B, T, n]
#     if n == W:
#         shifted_img_repermuted = shifted_img_reshaped.permute(2, 3, 0, 1, 4) # shape [B, T, C, H, W]
#     elif n == H:
#         shifted_img_repermuted = shifted_img_reshaped.permute(2, 3, 0, 4, 1) # shape [B, T, C, H, W]

#     return shifted_img_repermuted


# import torch

# def circular_shift(x, shifts, axis=-1): NOTE: ChatGPT
#     """
#     Circularly shift a batch of 4D tensors [T, C, H, W] along H or W,
#     with wraparound and linear interpolation.
    
#     Args:
#         x: torch.Tensor, shape [T, C, H, W]
#         shifts: torch.Tensor, shape [T], fractional shifts (float allowed)
#         axis: int, which axis to shift (2=H, 3=W)
    
#     Returns:
#         torch.Tensor, same shape as x
#     """
#     T, C, H, W = x.shape
#     n = H if axis == 2 else W
    
#     # Make index grid for target positions
#     idx = (torch.arange(n, device=x.device).unsqueeze(0) + shifts.unsqueeze(1)) % n  # [T, n]
#     i0 = torch.floor(idx).long()
#     i1 = (i0 + 1) % n
#     frac = (idx - i0).unsqueeze(1)  # [T,1,n]

#     # Expand dims for broadcasting with [T,C,H,W]
#     if axis == 2:  # shift along H
#         i0 = i0.unsqueeze(-1).expand(T, H, W)
#         i1 = i1.unsqueeze(-1).expand(T, H, W)
#         frac = frac.unsqueeze(-1)  # [T,1,H,1] → broadcast over W
#         out = (1 - frac) * torch.gather(x, 2, i0.unsqueeze(1).expand(T, C, H, W)) \
#             + frac * torch.gather(x, 2, i1.unsqueeze(1).expand(T, C, H, W))
#     elif axis == 3:  # shift along W
#         i0 = i0.unsqueeze(1).expand(T, W, H).transpose(1,2)  # [T,H,W]
#         i1 = i1.unsqueeze(1).expand(T, W, H).transpose(1,2)  # [T,H,W]
#         frac = frac.unsqueeze(1)  # [T,1,1,W] → broadcast over H
#         out = (1 - frac) * torch.gather(x, 3, i0.unsqueeze(1).expand(T, C, H, W)) \
#             + frac * torch.gather(x, 3, i1.unsqueeze(1).expand(T, C, H, W))
#     else:
#         raise ValueError("axis must be 2 (H) or 3 (W)")

#     return out