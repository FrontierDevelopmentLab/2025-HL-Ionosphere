import torch
import numpy as np
from ioncast import compute_sublunar_point, compute_subsolar_point
import datetime
from functools import lru_cache

import time
# TODO: convert the dynamic features to a dataset class, as it is wasteful to need to loop over the timestamps, make use of Gunes' new dataset class, + add on image generation + new dataset for the timestamp images
def timer(label):
    start = time.perf_counter()
    def end():
        elapsed = time.perf_counter() - start
        print(f"[{label}] took {elapsed:.4f} sec")
    return end

def haversine_distance(lat, lon, lat_grid, lon_grid, standardize=True):
    """
    Compute Haversine distances between N source points and a common lat/lon grid.
    
    Args:
        lat: np.ndarray of shape [N]
        lon: np.ndarray of shape [N]
        lat_grid: np.ndarray of shape [H, W]
        lon_grid: np.ndarray of shape [H, W]
        standardize: bool, whether to standardize the distances to [0, 1] and make the closest point 1 and the furthest point 0.
    
    Returns:
        distances: np.ndarray of shape [N, H, W]
    """
    # Convert all angles to radians
    lat = np.deg2rad(lat)[:, None, None] # [N, 1, 1]
    lon = np.deg2rad(lon)[:, None, None] # [N, 1, 1]
    lat_grid = np.deg2rad(lat_grid)[None, :, :]  # [1, H, W]
    lon_grid = np.deg2rad(lon_grid)[None, :, :]  # [1, H, W]

    dlon = lon_grid - lon
    dlat = lat_grid - lat

    a = np.sin(dlat / 2.0)**2 + np.cos(lat) * np.cos(lat_grid) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    if standardize:
        max_c = np.max(c)
        min_c = np.min(c)
        c = (max_c - c) / (max_c - min_c)
    return c  # [N, H, W]

# Day-in-month for non-leap year
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# TODO: Slight bug in parse fast, not accounting for leap years, i think we should convert this to use the datetime timestamps already generated in a previous step
def parse_fast(ts_str):
    year = int(ts_str[0:4])
    month = int(ts_str[5:7])
    day = int(ts_str[8:10])
    hour = int(ts_str[11:13])
    minute = int(ts_str[14:16])
    tod = (hour * 60 + minute) / (60 * 24)
    doy = (sum(DAYS_IN_MONTH[:month - 1]) + day + (hour / 24)) / 365 # Doesn't take into account leap years, but for our purposes this is fine
    return tod, doy

@lru_cache(maxsize=4) # Cache since no need to recompute every time
def get_lat_lon_grid(H, W):
    lat_vals = np.linspace(90, -90, H)                    # North to south
    lon_vals = np.linspace(-180, 180, W, endpoint=False)  # West to east
    return np.meshgrid(lon_vals, lat_vals)

def stack_features(
        sequence_batch, 
        n_img_datasets = 1, 
        model_type='GraphCast_forecast', # GraphCast_forecast, IonCastConvLSTM
        include_subsolar=True, 
        include_sublunar=True, 
        include_timestamp=True
    ): 
    """
    Sequence batch is a batch returned from a dataloader of a Sequences dataset 
    Sequences dataset structure:
    [Tensor[DSet 1 batch]    | Tensor[DSet 2 batch]   | ... | Tensor[DSet k batch]   | List[tuples(timestamps)]] |
    [torch.Size([B, T, F])   | torch.Size([B, T, F])  | ... | torch.Size([B, T, F])  | [T, B]                    |

    Example: (Default format of dataset order will be fist n datasets will be images, and rest of k-n datasets will be tabular)
    | Tensor[JPLD batch]               | Tensor[OMNIWeb batch]   | Tensor[Celestrak batch] | Tensor[Solar Index batch]  | List[tuples(timestamps)] |
    | torch.Size([B, T, C, H, W])      | torch.Size([B, T, F])   | torch.Size([B, T, F])   | torch.Size([B, T, F])      | [T, B]                   |
    | torch.Size([1, 10, 1, 180, 360]) | torch.Size([1, 10, 16]) | torch.Size([1, 10, 2])  | torch.Size([1, 10, 4])     | [10, 1]                  | 
    """

    assert n_img_datasets > 0, f"n_img_datasets {n_img_datasets} must be greater than 0"
    assert len(sequence_batch) > 0, f"sequence_batch of length {len(sequence_batch)} must not be empty"

    elapsed_time = timer("process dataset features")

    # List to hold all features stacked in shape [B, T, C, H, W]
    features_list = []
    
    # Split the dataset into image datasets, global feature datasets, and timestamps
    image_datasets = sequence_batch[:n_img_datasets] # torch.Size(torch.Size([B, T, C_i, H, W])
    global_param_datasets = sequence_batch[n_img_datasets:-1] # torch.Size(torch.Size([B, T, F_i])
    timestamps = sequence_batch[-1] # List[Tuple] -> [T, B]

    # Stack the datasets along the channel dimension
    stacked_imgs = torch.cat(image_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(C_i), H, W])

    # Get shapes
    assert len(stacked_imgs.shape) == 5, f"Expected first datasets to be image datasets with shape [B, T, C, H, W], got {stacked_imgs.shape}"
    B, T, C, H, W = stacked_imgs.shape 
    features_list.append(stacked_imgs)
    
    # Add global features to feature_list if they exist
    if global_param_datasets:
        stacked_globals = torch.cat(global_param_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(F_i)])
        B_g, T_g, _ = stacked_globals.shape
        stacked_globals = stacked_globals[:, :, :, None, None].repeat(1, 1, 1, H, W) # reshape to [B, T, C, H, W]
        assert T == T_g, "Mismatch in sequence length between image data and globals"
        assert B == B_g, "Mismatch in sequence length between image data and globals"
        features_list.append(stacked_globals)

    # If graphcast, requires batch size is 1
    if model_type == "GraphCast_forecast":
        assert B == 1, "Graphcast only allows a batch size of 1"

    elapsed_time()

    elapsed_time = timer("compute lat-lon grid")

    # Handle subsolar and sublunar points
    if include_subsolar or include_sublunar:
        # compute lat_grid and lon_grid matching shape H, W
        lon_grid, lat_grid = get_lat_lon_grid(H, W)

    elapsed_time()

    # Skip logic if no subsolar or sublunar points are requested
    if include_subsolar or include_sublunar:
        
        elapsed_time = timer("compute subsolar & sublunar lat long points")

        sublunar_list = [] 
        subsolar_list = [] 
        # For each timestamp in each sequence and batch, compute subsolar and sublunar points
        for batch_idx in range(B): # B will usually be 1 (for graphcast only)
            for seq_idx in range(T): # under default conditions T = 2 (number of frames of history to include)
                timestamp = timestamps[seq_idx][batch_idx]
                if include_subsolar:
                    subsolar_lat, subsolar_lon = compute_subsolar_point(timestamp)
                    subsolar_list.append([subsolar_lat, subsolar_lon])
                if include_sublunar:
                    sublunar_lat, sublunar_lon = compute_sublunar_point(timestamp)
                    sublunar_list.append([sublunar_lat, sublunar_lon]) 
        # Useful example for debugging maybe:
        #  # [T=2, B=4] [T_0B_0, T_0B_1, T_0B_2, T_0B_3, T_1B_0, T_1B_1, ...] (if looping over T then B)
        #  # [T=2, B=4] [T_0B_0, T_1B_0, T_0B_1, T_1B_1, T_0B_0, T_1B_0, T_0B_1, T_1B_1,  ...] (if looping over B then T)
        
        elapsed_time()

        elapsed_time = timer("compute subsolar & sublunar haversine images")
        

        # For each lat/lon pair, compute the Haversine distance to the grid
        if include_subsolar:
            subsolar_latlons = np.array(subsolar_list)
            subsolar_dist_map = haversine_distance(subsolar_latlons[:,0], subsolar_latlons[:,1], lat_grid, lon_grid, standardize=True) # [B x T, H, W]
            subsolar_dist_map = torch.from_numpy(subsolar_dist_map).float().reshape(B, T, 1, H, W) # [B, T, C=1, H, W]
            features_list.append(subsolar_dist_map)
            
        if include_sublunar:
            sublunar_latlons = np.array(sublunar_list)
            sublunar_dist_map = haversine_distance(sublunar_latlons[:,0], sublunar_latlons[:,1], lat_grid, lon_grid, standardize=True) # [B x T, H, W]
            sublunar_dist_map = torch.from_numpy(sublunar_dist_map).float().reshape(B, T, 1, H, W) # [B, T, C=1, H, W]
            features_list.append(sublunar_dist_map)
        elapsed_time()

    if include_timestamp: 
        elapsed_time = timer("compute timestamp encodings")
        # Transpose to shape (B, T)
        timestamps_TB = list(zip(*timestamps))  # Now shape (B, T)

        # Apply parse_fast and flatten to shape (B, T, C=4)
        timestamp_batch_features = []
        for ts_list in timestamps_TB:  # For each batch of sequences (B sequenc)
            timestamp_sequence_features = []
            for ts in ts_list: # For each timestamp in the batch
                # Parse timestamp
                tod, doy = parse_fast(ts)

                # Compute sine and cosine features of time of day and day of year (normalized to [0, 1])
                sin_tod = np.sin(2 * np.pi * tod)
                cos_tod = np.cos(2 * np.pi * tod)
                sin_doy = np.sin(2 * np.pi * doy)
                cos_doy = np.cos(2 * np.pi * doy)

                # timestamp_sequence_features is shape (T, 4)
                timestamp_sequence_features.append([sin_tod, cos_tod, sin_doy, cos_doy])
            # timestamp_batch_features is shape (B, T, 4)
            timestamp_batch_features.append(timestamp_sequence_features)

        # Reshape (B, T, C=4, H, W) (repeat along H and W dimensions) 
        time_tensor = torch.tensor(timestamp_batch_features)[:,:,:,None,None].repeat(1, 1, 1, H, W)
        features_list.append(time_tensor)
        
        elapsed_time()

    # Stack all features in features_list along the channel dimension
    stacked_features = torch.cat(features_list, dim=2) # [B, T, C_stacked, H, W]

    # If the model type is GraphCast_forecast, reshape to N, C, H, W format (flatten T and C together)
    if model_type == 'GraphCast_forecast':
        C_stacked = stacked_features.shape[2]

        # Convert image and global parameters to N, C, H, W format
        stacked_features = stacked_features.reshape(B, T * C_stacked, H, W)

    return stacked_features  # [B, C_total, H, W] or [B, T, C_total, H, W] (depends on model_type)




    # JPLD 1, + omni 16 + celstrak 2 = 19 -> 20 subsolar, 21 subsolar , 4x timestamp
    # [(00, 15, 30), 
