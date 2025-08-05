import torch
import numpy as np
from ioncast import compute_sublunar_point, compute_subsolar_point
import datetime
from functools import lru_cache


def stack_features(
        sequence_batch, 
        n_img_datasets = 1, 
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


    # elapsed_time = timer("process dataset features")

    # List to hold all features stacked in shape [B, T, C, H, W]
    features_list = []
    
    # Split the dataset into image datasets, global feature datasets, and timestamps
    image_datasets = sequence_batch[:n_img_datasets] # torch.Size(torch.Size([B, T, C_i, H, W])
    global_param_datasets = sequence_batch[n_img_datasets:-1] # torch.Size(torch.Size([B, T, F_i])
    _ = sequence_batch[-1] # List[Tuple] -> [T, B]
    
    if len(image_datasets[0].shape) == 4:
        image_datasets = [T.unsqueeze(0) for T in image_datasets] # if batch size is missing, the sequence_batch will be a tuple
        global_param_datasets = [T.unsqueeze(0) for T in global_param_datasets]

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

    # Stack all features in features_list along the channel dimension
    stacked_features = torch.cat(features_list, dim=2) # [B, T, C_stacked, H, W]

    return stacked_features  # [B, T, C_total, H, W]


