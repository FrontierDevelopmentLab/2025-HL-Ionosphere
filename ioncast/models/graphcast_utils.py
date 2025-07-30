import torch
import numpy as np
from ioncast import compute_sublunar_point, compute_subsolar_point

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
    lat = np.deg2rad(lat).reshape(-1, 1, 1)  # [N, 1, 1]
    lon = np.deg2rad(lon).reshape(-1, 1, 1)  # [N, 1, 1]
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

def parse_fast(ts_str):
    year = int(ts_str[0:4])
    month = int(ts_str[5:7])
    day = int(ts_str[8:10])
    hour = int(ts_str[11:13])
    minute = int(ts_str[14:16])
    tod = (hour * 60 + minute) / (60 * 24)
    doy = (sum(DAYS_IN_MONTH[:month - 1]) + day + hour / 24) / 365
    return tod, doy

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
    [Tensor[JPLD batch]               | Tensor[OMNIWeb batch]   | Tensor[Celestrak batch] | Tensor[Solar Index batch]  | List[tuples(timestamps)]] |
    [torch.Size([B, T, C, H, W])      | torch.Size([B, T, F])   | torch.Size([B, T, F])   | torch.Size([B, T, F])      | [T, B]            |
    [torch.Size([1, 10, 1, 180, 360]) | torch.Size([1, 10, 16]) | torch.Size([1, 10, 2])  | torch.Size([1, 10, 4])     | [10, 1]           | 
    """

    # Split the dataset into image datasets, global feature datasets, and timestamps
    image_datasets = sequence_batch[:n_img_datasets] # torch.Size(torch.Size([B, T, C_i, H, W])
    global_param_datasets = sequence_batch[n_img_datasets:-1] # torch.Size(torch.Size([B, T, F_i])
    timestamps = sequence_batch[-1] # List[Tuple] -> [T, B]

    # Stack the datasets along the channel dimension
    stacked_imgs = torch.cat(image_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(C_i), H, W])
    stacked_globals = torch.cat(global_param_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(F_i)])
    stacked_globals = stacked_globals[:, :, :, None, None].repeat(1, 1, 1, H, W) # reshape to [B, T, C, H, W]

    # Get shapes
    B, T, C, H, W = stacked_imgs.shape
    B_g, T_g, F, _, _ = stacked_globals.shape

    assert T == T_g, "Mismatch in sequence length between image data and globals"

    # If graphcast, requires batch size is 1
    if model_type == "Graphcast_forecast":
        assert B == B_g and B == 1, "Graphcast only allows a batch size of 1"
        

    # Handle subsolar and sublunar points
    dynamic_features = []
    if include_subsolar or include_sublunar:
        # compute lat_grid and lon_grid matching shape H, W
        lat_vals = np.linspace(90, -90, H)         # North to south
        lon_vals = np.linspace(-180, 180, W, endpoint=False)  # West to east

        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    # Skip logic if no subsolar or sublunar points are requested
    if include_subsolar or include_sublunar:
        sublunar_list = []
        subsolar_list = []
        # For each timestamp in each sequence and batch, compute subsolar and sublunar points
        for seq_idx in range(T): # under default conditions T = 2 (number of frames of history to include)
            for batch_idx in range(B): # B will usually be 1 (for graphcast only)
                timestamp = timestamps[seq_idx, batch_idx]
                if include_subsolar:
                    subsolar_lat, subsolar_lon = compute_subsolar_point(timestamp)
                    subsolar_list.append([subsolar_lat, subsolar_lon])
                if include_sublunar:
                    sublunar_lat, sublunar_lon = compute_sublunar_point(timestamp)
                    sublunar_list.append([sublunar_lat, sublunar_lon]) 

        # For each lat/lon pair, compute the Haversine distance to the grid
        if include_subsolar:
            subsolar_latlons = np.array(subsolar_list)
            subsolar_dist_map = haversine_distance(subsolar_latlons[:,0], subsolar_latlons[:,1], lat_grid, lon_grid, standardize=True)
            subsolar_dist_map = torch.tensor(subsolar_dist_map).reshape(T, B, H, W).permute(1, 0, 2, 3)
            dynamic_features.append(subsolar_dist_map)
            
        if include_sublunar:
            sublunar_latlons = np.array(sublunar_list)
            sublunar_dist_map = haversine_distance(sublunar_latlons[:,0], sublunar_latlons[:,1], lat_grid, lon_grid, standardize=True)
            sublunar_dist_map = torch.tensor(sublunar_dist_map).reshape(T, B, H, W).permute(1, 0, 2, 3) # []
            dynamic_features.append(sublunar_dist_map)

    if include_timestamp: 
        # Transpose to shape (B, T)
        timestamps_TB = list(zip(*timestamps))  # Now shape (B, T)

        # Apply parse_fast and flatten to shape (B, T, C=4)
        features = []
        for ts_list in timestamps_TB:  # For each batch of sequences (B sequenc)
            sample_features = []
            for ts in ts_list: # For each timestamp in the batch
                # Parse timestamp
                tod, doy = parse_fast(ts)

                # Compute sine and cosine features of time of day and day of year (normalized to [0, 1])
                sin_tod = np.sin(2 * np.pi * tod)
                cos_tod = np.cos(2 * np.pi * tod)
                sin_doy = np.sin(2 * np.pi * doy)
                cos_doy = np.cos(2 * np.pi * doy)

                sample_features.append([sin_tod, cos_tod, sin_doy, cos_doy])
            features.append(sample_features)  # sample_features is shape (T, 4), features is shape (B, T, 4)

        # Step 3: Convert to array: (B, 4*T)
        time_tensor = torch.tensor(np.array(features)).repeat(1, 1, 1, H, W) # shape (B, T, C=4, H, W)
        dynamic_features.append(time_tensor)

    # If there are dynamic features, stack them along the channel dimension
    if dynamic_features:
        dynamic_tensor = torch.cat(dynamic_features, dim=2) # [B, T, C_dyn, H, W]
        stacked_features = torch.cat([stacked_imgs, stacked_globals, dynamic_tensor], dim=2)  # [B, T, C_stacked = sum(C_i + F_i) + C_dyn , H, W]
    else:
        stacked_features = torch.cat([stacked_imgs, stacked_globals], dim=2)  # [B, T, C_stacked = sum(C_i + F_i), H, W]

    # If the model type is GraphCast_forecast, reshape to N, C, H, W format (flatten T and C together)
    if model_type == 'GraphCast_forecast':
        C_stacked = stacked_features.shape[2]

        # Convert image and global parameters to N, C, H, W format
        stacked_features = stacked_features.reshape(B, T * C_stacked, H, W)

    return stacked_features  # [B, C_total, H, W]