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

# Function that takes in two lat, lon points and calculates the distance between them as arc length on a sphere
def haversine_distance(lon1, lat1, lon_grid, lat_grid):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    """
    lat, lon, lat_grid, lon_grid = np.deg2rad(lat), np.deg2rad(lon), np.deg2rad(lat_grid), np.deg2rad(lon_grid)
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon_grid - lon1
    dlat = lat_grid - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat_grid) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    return c

def process_to_grid_nodes(sequence_batch, n_img_datasets = 1, include_subsolar=True, include_sublunar=True, include_timestamp=True): #
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

    stacked_imgs = torch.cat(image_datasets, dim=2) # torch.Size(torch.Size([B, T,sum(C_i), H, W])
    stacked_globals = torch.cat(global_param_datasets, dim=2) # torch.Size(torch.Size([B, T, Sum(F_i)])

    B, T, C, H, W = stacked_imgs.shape
    B_g, T_g, F = stacked_globals.shape
    assert B == B_g and B == 1, "Graphcast only allows a batch size of 1"
    assert T == T_g, "Mismatch in sequence length between image data and globals"

    stacked_imgs = stacked_imgs.reshape(B, C*T, H, W) 
    stacked_globals = stacked_globals.reshape(B, F*T) 
    stacked_globals = stacked_globals[:, :, None, None].repeat(1, 1, H, W)

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
                sublunar_lat, subluner_lon = compute_sublunary_point(timestamp)
                
                # sublunar_list.append([sublunar_lat, subluner_lon]) 

    if include_subsolar:
        subsolar_ = torch.tensor(subsolar_list).reshape(T, B, 2).permute(1, 0, 2)
    if include_sublunar:
        sublunar_ = torch.tensor(sublunar_list).reshape(T, B, 2).permute(1, 0, 2)

    if include_timestamp: # NOTE: double check tomorrow type of the timestamps (either datetime obj or pandas datetime)
        # Convert pandas timestamps to sine/cosine of the local time of day & sine/cosine of the of year progress (normalized to [0, 1))
        sin_local_time_of_day = torch.sin(2 * np.pi * timestamps[:, 0] / 24) # TODO: linnea check if indexing is correct
        cos_local_time_of_day = torch.cos(2 * np.pi * timestamps[:, 0] / 24)
        sin_year_progress = torch.sin(2 * np.pi * timestamps[:, 0] / 365)
        cos_year_progress = torch.cos(2 * np.pi * timestamps[:, 0] / 365)

        # Create a tensor of shape B, 4, H, W (copy over H and W dimensions)
        time_tensor = torch.zeros(B, 4, H, W)
        time_tensor[:, 0, :, :] = sin_local_time_of_day
        time_tensor[:, 1, :, :] = cos_local_time_of_day
        time_tensor[:, 2, :, :] = sin_year_progress
        time_tensor[:, 3, :, :] = cos_year_progress
