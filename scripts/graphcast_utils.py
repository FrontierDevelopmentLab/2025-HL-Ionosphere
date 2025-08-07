import torch

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


def calc_shapes_for_stack_features(seq_dataset_batch, aux_datasets, context_window, batched=True):
# Calculate the number of image-like features and make sure image-like datasets are before non-image-like datasets in the dataset list
    img_tensor_ndim = 4
    channel_idx = 1

    if batched: # this accounts for the offset caused by the inclusion of the batch dimension
        img_tensor_ndim += 1
        channel_idx += 1

    n_img_datasets = 0
    non_img_encountered_flag = False
    len_forcing_channel = 0
    forcing_channels = None

    # if 'sunmoon' is in the aux_datasets, we need to find the channel index it begins at
    if 'sunmoon' in aux_datasets: 
        sunmoon_idx = aux_datasets.index('sunmoon') + 1 # Add 1 to account for the JPLD dataset at index 0
        sunmoon_channel_idx = 0

    for idx, T in enumerate(seq_dataset_batch):
        if isinstance(T, torch.Tensor):
            # Check if the dataset is image-like (5D tensor) or non-image-like (3D tensor): [B, T, C, H, W] vs. [B, T, C]
            if len(T.shape) == img_tensor_ndim:
                if non_img_encountered_flag:
                    raise ValueError('All image-like datasets must be before non-image-like datasets in the dataset list')
                n_img_datasets += 1
            else:
                non_img_encountered_flag = True

            if 'sunmoon' in aux_datasets:
                # Get the index of the final channel before the sunmoon dataset
                if idx < sunmoon_idx:
                    sunmoon_channel_idx += T.shape[channel_idx] 

                # If we are at the sunmoon dataset, we need to get the number of channels in the sunmoon dataset
                elif idx == sunmoon_idx:
                    len_forcing_channel = T.shape[channel_idx] # Get the number of channels in the sunmoon dataset
                    forcing_channels = range(sunmoon_channel_idx, sunmoon_channel_idx + len_forcing_channel) # Get the channel indices for the forcing channels

    # Get the number of channels in the input and compute the number of features
    dummy_batch = stack_features(seq_dataset_batch, n_img_datasets=n_img_datasets)
    _, _, C, _, _ = dummy_batch.shape # B, T, C, H, W

    n_feats = context_window * C + len_forcing_channel

    return n_feats, C, forcing_channels, n_img_datasets