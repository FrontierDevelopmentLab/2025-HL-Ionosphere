import torch
import numpy as np
from physicsnemo.utils.graphcast.icosahedral_mesh import get_hierarchy_of_triangular_meshes_for_sphere
from ioncast import compute_sublunary_point, compute_subsolar_point

# NOTE: delete later if unused
# Convert mesh vertices (3D sphere coordinates) to lat/lon for interpolation 
def sphere_to_latlon(vertices):
    """Convert 3D unit sphere coordinates to lat/lon in degrees"""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Convert to lat/lon
    lat = torch.asin(z) * 180 / np.pi  # Latitude in degrees [-90, 90]
    lon = torch.atan2(y, x) * 180 / np.pi  # Longitude in degrees [-180, 180]
    
    return lat, lon

# # Function that takes in two lat, lon points and calculates the distance between them as arc length on a sphere
# def haversine_distance(lon1, lat1, lon_grid, lat_grid):
#     """
#     Calculate the great circle distance between two points
#     on the earth (specified in decimal degrees)
    
#     All args must be of equal length.    
#     """
#     lat, lon, lat_grid, lon_grid = np.deg2rad(lat), np.deg2rad(lon), np.deg2rad(lat_grid), np.deg2rad(lon_grid)
#     # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
#     dlon = lon_grid - lon1
#     dlat = lat_grid - lat1
    
#     a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat_grid) * np.sin(dlon/2.0)**2
    
#     c = 2 * np.arcsin(np.sqrt(a))
#     return c
def haversine_distance(lat, lon, lat_grid, lon_grid): # TODO: add parameter to have subsolar point be 0 vs 1 & standardize
    """
    Compute Haversine distances between N source points and a common lat/lon grid.
    
    Args:
        lat: np.ndarray of shape [N]
        lon: np.ndarray of shape [N]
        lat_grid: np.ndarray of shape [H, W]
        lon_grid: np.ndarray of shape [H, W]
    
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
    return c  # [N, H, W]

def process_to_grid_nodes(
        sequence_batch, 
        n_img_datasets = 1, 
        model_type='GraphCast_forecast', # GraphCast_forecast, IonCastConvLSTM
        include_subsolar=True, 
        include_sublunar=True, 
        include_timestamp=True
    ): 
    # TODO: include flatten parameter such that we can either return with shape BTCHW or BCHW depending on which model will use the outputs, for a bug free implementation of this, we can no longer assume B = 1 so should not have that assert stamtent
    # and need to ensure concats + any other operations occuring over dimensions happens on the correct dim. 
    # Sequence batch is a batch returned from a dataloader of a Sequences dataset 
    # Sequences dataset structure:
    # [Tensor[DSet1 batch].    | Tensor[DSet2 batch].   | ... | Tensor[DSet k batch]   | List[timestamps]] |
    # [torch.Size([B, T, F])   | torch.Size([B, T, F])  | ... | torch.Size([B, T, F])  | [T, B]            | 
    # 
    #
    # Example: (Default format of dataset order will be fist n datasets will be images, and rest of k-n datasets will be tabular)
    # [Tensor[JPLD batch]               | Tensor[OMNIWeb batch]   | Tensor[Celestrak batch] | Tensor[Solar Index batch]  | List[timestamps]] |
    # [torch.Size([B, T, C, H, W])      | torch.Size([B, T, F])   | torch.Size([B, T, F])   | torch.Size([B, T, F])      | [T, B]            |
    # [torch.Size([1, 10, 1, 180, 360]) | torch.Size([1, 10, 16]) | torch.Size([1, 10, 2])  | torch.Size([1, 10, 4])     | [10, 1]           | 
    #

    image_datasets = sequence_batch[:n_img_datasets] # torch.Size(torch.Size([B, T, C_i, H, W])
    global_param_datasets = sequence_batch[n_img_datasets:-1] # torch.Size(torch.Size([B, T, F_i])

    timestamps = sequence_batch[-1] # List[Tuple] -> [T, B]

    stacked_imgs = torch.cat(image_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(C_i), H, W])
    stacked_globals = torch.cat(global_param_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(F_i)])

    B, T, C, H, W = stacked_imgs.shape
    B_g, T_g, F = stacked_globals.shape

    # TODO: move to the end, after dynamic features are computed
    if model_type == 'GraphCast_forecast':
        assert B == B_g and B == 1, "Graphcast only allows a batch size of 1" # comment this assert statemnt (See TODO: at the start of function)
        assert T == T_g, "Mismatch in sequence length between image data and globals"
        
        # N, C, H, W
        # Convert image and global parameters to N, C, H, W format
        stacked_imgs = stacked_imgs.reshape(B, C*T, H, W) #  TODO: do all C*T / F*T reshaping at the end based on wether a passed in flatten flag is true or not
        stacked_globals = stacked_globals.reshape(B, F*T) 
        stacked_globals = stacked_globals[:, :, None, None].repeat(1, 1, H, W)

    dynamic_features = []

    if include_subsolar or include_sublunar:
        # compute lat_grid and lon_grid matching shape H, W
        lat_vals = np.linspace(90, -90, H)         # North to south
        lon_vals = np.linspace(-180, 180, W, endpoint=False)  # West to east

        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    sublunar_list = []
    subsolar_list = []
    # assuming batch size 1 # NOTE: this will be changed 
    for seq_idx in range(T): # under default conditions T = 2 (number of frames of history to include)
        for batch_idx in range(B): # B should be 1.
            timestamp = timestamps[seq_idx, batch_idx]
            if include_subsolar:
                subsolar_lat, subsolar_lon = compute_sublunary_point(timestamp)
                subsolar_list.append([subsolar_lat, subsolar_lon])
            if include_sublunar:
                sublunar_lat, sublunar_lon = compute_sublunary_point(timestamp)
                sublunar_list.append([sublunar_lat, sublunar_lon]) 

    if include_subsolar:
        subsolar_latlons = np.array(subsolar_list)
        subsolar_dist_map = haversine_distance(subsolar_latlons[:,0], subsolar_latlons[:,1], lat_grid, lon_grid)
        subsolar_dist_map = torch.tensor(subsolar_dist_map).reshape(T, B, H, W).permute(1, 0, 2, 3)
        dynamic_features.append(subsolar_dist_map)
    if include_sublunar:
        sublunar_latlons = np.array(sublunar_list)
        sublunar_dist_map = haversine_distance(sublunar_latlons[:,0], sublunar_latlons[:,1], lat_grid, lon_grid)
        sublunar_dist_map = torch.tensor(sublunar_dist_map).reshape(T, B, H, W).permute(1, 0, 2, 3) # []
        dynamic_features.append(sublunar_dist_map)

    # TODO: Create a include_timestamp for non-graphcast models
    if include_timestamp: # NOTE: double check tomorrow type of the timestamps (either datetime obj or pandas datetime)
        # Convert pandas timestamps to sine/cosine of the local time of day & sine/cosine of the of year progress (normalized to [0, 1))
        # Graphcast expects a sin(minute of the day) and cos(minute of the day) as well as sin(day of the year) and cos(day of the year)

        minute_of_day = lambda timestamp: timestamp.hour * 60 + timestamp.minute
        day_of_year = lambda timestamp: timestamp.day + timestamp.hour / 24 + timestamp.minute / (24 * 60) # Do we want to include the hours + mins?
        
        ts_time_of_day = list(map(minute_of_day, timestamps)) # this doenst quite work, it applies the function on the tuple
        ts_day_of_year = list(map(day_of_year, timestamps)) 

        # 
        
        

        # current shape
        sin_local_time_of_day = torch.sin(2 * np.pi * ts_time_of_day) 
        cos_local_time_of_day = torch.cos(2 * np.pi * ts_time_of_day) 
        sin_year_progress = torch.sin(2 * np.pi * ts_day_of_year)   
        cos_year_progress = torch.cos(2 * np.pi * ts_day_of_year)  

        # Create a tensor of shape B, 4*T, H, W (copy over H and W dimensions)
        time_tensor = torch.zeros(B, 4*T, H, W)
        time_tensor[:, 0, :, :] = sin_local_time_of_day # these tensors have shape [T,] currently I think so indexing will break here i think TOOD: go over tmo
        time_tensor[:, 1, :, :] = cos_local_time_of_day
        time_tensor[:, 2, :, :] = sin_year_progress
        time_tensor[:, 3, :, :] = cos_year_progress
        dynamic_features.append(time_tensor)

    if dynamic_features:
        dynamic_tensor = torch.cat(dynamic_features) # [B, C_dyn, H, W]
        stacked_features = torch.cat([stacked_imgs, stacked_globals, dynamic_tensor], dim=1)  # [B, C_stacked = sum(C_i + F_i) * T + C_dyn , H, W]
    else:
        stacked_features = torch.cat([stacked_imgs, stacked_globals], dim=1)  # [B, C_stacked = sum(C_i + F_i) * T, H, W]
    return stacked_features  # [B, C_total, H, W]