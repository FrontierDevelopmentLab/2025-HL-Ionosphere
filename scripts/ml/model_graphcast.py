import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional

from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

class IonCastGNN(nn.Module):
    """
    Wrapper for GraphCast from NVIDIA's physicsnemo package, a pytorch implementation. This wrapper
    will handle the input/output format and any necessary preprocessing/postprocessing.

    Parameters
    ----------
    multimesh_level: int, optional
        Level of the latent mesh, by default 6
    multimesh: bool, optional
        If the latent mesh is a multimesh, by default True
        If True, the latent mesh includes the nodes corresponding
        to the specified `mesh_level`and incorporates the edges from
        all mesh levels ranging from level 0 up to and including `mesh_level`.
    input_res: Tuple[int, int]
        Input resolution of the latitude-longitude grid
    input_dim_grid_nodes : int, optional
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim_grid_nodes : int, optional
        Final output dimensionality of the edge features, by default 227
    processor_type: str, optional
        The type of processor used in this model. Available options are
        'MessagePassing', and 'GraphTransformer', which correspond to the
        processors in GraphCast and GenCast, respectively.
        By default 'MessagePassing'.
    khop_neighbors: int, optional
        Number of khop neighbors used in the GraphTransformer.
        This option is ignored if 'MessagePassing' processor is used.
        By default 0.
    processor_layers : int, optional
        Number of processor layers, by default 16
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    activation_fn : str, optional
        Type of activation function, by default "silu"
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    use_cugraphops_encoder : bool, default=False
        Flag to select cugraphops kernels in encoder
    use_cugraphops_processor : bool, default=False
        Flag to select cugraphops kernels in the processor
    use_cugraphops_decoder : bool, default=False
        Flag to select cugraphops kernels in the decoder
    do_concat_trick : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    partition_size : int, default=1
        Number of process groups across which graphs are distributed. If equal to 1,
        the model is run in a normal Single-GPU configuration.
    partition_group_name : str, default=None
        Name of process group across which graphs are distributed. If partition_size
        is set to 1, the model is run in a normal Single-GPU configuration and the
        specification of a process group is not necessary. If partitition_size > 1,
        passing no process group name leads to a parallelism across the default
        process group. Otherwise, the group size of a process group is expected
        to match partition_size.
    use_lat_lon_partitioning : bool, default=False
        flag to specify whether all graphs (grid-to-mesh, mesh, mesh-to-grid)
        are partitioned based on lat-lon-coordinates of nodes or based on IDs.
    expect_partitioned_input : bool, default=False
        Flag indicating whether the model expects the input to be already
        partitioned. This can be helpful e.g. in multi-step rollouts to avoid
        aggregating the output just to distribute it in the next step again.
    global_features_on_rank_0 : bool, default=False
        Flag indicating whether the model expects the input to be present
        in its "global" form only on group_rank 0. During the input preparation phase,
        the model will take care of scattering the input accordingly onto all ranks
        of the process group across which the graph is partitioned. Note that only either
        this flag or expect_partitioned_input can be set at a time.
    produce_aggregated_output : bool, default=True
        Flag indicating whether the model produces the aggregated output on each
        rank of the procress group across which the graph is distributed or
        whether the output is kept distributed. This can be helpful e.g.
        in multi-step rollouts to avoid aggregating the output just to distribute
        it in the next step again.
    produce_aggregated_output_on_all_ranks : bool, default=True
        Flag indicating - if produce_aggregated_output is True - whether the model
        produces the aggregated output on each rank of the process group across
        which the group is distributed or only on group_rank 0. This can be helpful
        for computing the loss using global targets only on a single rank which can
        avoid either having to distribute the computation of a loss function.

    Note
    ----
    Based on these papers:
    - "GraphCast: Learning skillful medium-range global weather forecasting"
        https://arxiv.org/abs/2212.12794
    - "Forecasting Global Weather with Graph Neural Networks"
        https://arxiv.org/abs/2202.07575
    - "Learning Mesh-Based Simulation with Graph Networks"
        https://arxiv.org/abs/2010.03409
    - "MultiScale MeshGraphNets"
        https://arxiv.org/abs/2210.00612
    - "GenCast: Diffusion-based ensemble forecasting for medium-range weather"
        https://arxiv.org/abs/2312.15796
    """
    
    def __init__(
        self, 
        mesh_level: Optional[int] = 6,
        multimesh: bool = True,
        input_res: tuple = (721, 1440),
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 227,
        processor_type: str = "MessagePassing",
        khop_neighbors: int = 32,
        num_attention_heads: int = 4,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        use_cugraphops_encoder: bool = False,
        use_cugraphops_processor: bool = False,
        use_cugraphops_decoder: bool = False,
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        partition_size: int = 1,
        partition_group_name: Optional[str] = None,
        use_lat_lon_partitioning: bool = False,
        expect_partitioned_input: bool = False,
        global_features_on_rank_0: bool = False,
        produce_aggregated_output: bool = True,
        produce_aggregated_output_on_all_ranks: bool = True,
        device="cpu",
        context_window: int = 2,
    ):

        super().__init__()

        # Initialize the GraphCast model with the provided parameters
        # Note: for more info see https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/models/graphcast/graph_cast_net.py
        self.graph_cast = GraphCastNet(
            mesh_level=mesh_level,
            multimesh=multimesh,
            input_res=input_res,
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim_grid_nodes=output_dim_grid_nodes,
            processor_type=processor_type,
            khop_neighbors=khop_neighbors,
            num_attention_heads=num_attention_heads,
            processor_layers=processor_layers,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            activation_fn=activation_fn,
            norm_type=norm_type,
            use_cugraphops_encoder=use_cugraphops_encoder,
            use_cugraphops_processor=use_cugraphops_processor,
            use_cugraphops_decoder=use_cugraphops_decoder,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
            partition_size=partition_size,
            partition_group_name=partition_group_name,
            use_lat_lon_partitioning=use_lat_lon_partitioning,
            expect_partitioned_input=expect_partitioned_input,
            global_features_on_rank_0=global_features_on_rank_0,
            produce_aggregated_output=produce_aggregated_output,
            produce_aggregated_output_on_all_ranks=produce_aggregated_output_on_all_ranks,
        )

        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.processor_layers = processor_layers
        self.mesh_level = mesh_level
        self.processor_type = processor_type
        self.context_window = context_window
        
        # Move the model to the specified device
        self.graph_cast = self.graph_cast.to(device)

    def forward(self, input_grid):
        """
        Passes the input grid through the GraphCast model.
        Forward method of GraphCast reshapes the input from [B, C, H, W] to be H x W grid nodes with 
        C features, and passes the grid nodes through the encoder, processor, and decoder,
        returns the updated grid nodes, and reshapes the output back to [B, C, H, W].

        Parameters
        ----------
        input_grid : torch.Tensor
            Input tensor of shape (B, T, C, H, W), where:
            - B is the batch size (must be 1 for GraphCast, see physicsnemo documentation),
            - T is the sequence length (number of time steps),
            - C is the number of grid node features (input_dim_grid_nodes),
            - H is the height of the grid (n_lat),
            - W is the width of the grid (n_lon).

        Returns
        -------
        output_grid : torch.Tensor
            Output tensor of shape (B, C, H, W), where:
            - B is the batch size (1),
            - C is the number of grid node features (output_dim_grid_nodes),
            - H is the height of the grid (n_lat),
            - W is the width of the grid (n_lon).
        """
        B, T, C, H, W = input_grid.shape
        assert B == 1, "Batch size must be 1"

        # Reshape input to (B, T*C, H, W) for GraphCast
        input_grid = input_grid.reshape(B, T * C, H, W)
        # print(f"Input Grid type: {input_grid.dtype}")
        # print(f"forward input: {input_grid.shape}")

        # Pass through GraphCast
        output_grid = self.graph_cast(input_grid)

        return output_grid
    
    # def predict(self, data_context, prediction_window=4, ):
    #     """ 
    #     Forecasts the next time step given an input grid. 
    #     Duplication of the forward method to maintain consistency with the IonCastConvLSTM interface.
    #     The input grid is expected to be of shape (B, T, C, H, W), and the forward pass will reshape it to (B, T*C, H, W) for processing.

    #     Parameters
    #     ----------
    #     data_context : torch.Tensor
    #         Input tensor of shape (B, T, C, H, W), where:
    #         - B is the batch size, can be > 1
    #         - T is the context window length (number of time steps),
    #         - C is the number of grid node features (input_dim_grid_nodes),
    #         - H is the height of the grid (n_lat),
    #         - W is the width of the grid (n_lon).

    #     prediction_window : int, optional
    #         Number of time steps to predict autoregressively. Default is 4.
    #     """
        
    # #     # Create a masked grid and fill in :context_window with the context data
    # #     B, T, C, H, W = data_context.shape
    # #     masked_grid = torch.zeros(B, T + prediction_window, C, H, W).to(data_context.device) # [B, T + prediction_window, C, H, W]
    # #     masked_grid[:, :T, :, :, :] = data_context
    
    #     # for step in range(prediction_window): 
    #     #     # Pass context data through the model
    #     #     input_grid = masked_grid[:, step:T+step, :, :, :] # [B, T, C, H, W]
    #     #     step_output = self(input_grid) # [B, 1*C, H, W]
                
    #     #     # Fill the masked grid with the output of the model
    #     #     masked_grid[:, T+step, :, :, :] = step_output # [B, C, H, W]

    #     # return masked_grid
    
    # # def predict(self, data_context, prediction_window=4):
    #     B, T, C, H, W = data_context.shape
    #     device = data_context.device

    #     # We'll collect predictions here
    #     predictions = []

    #     # Initialize the masked grid with the context
    #     masked_grid = data_context.clone()  # shape (B, T, C, H, W)

    #     for step in range(prediction_window): 
    #         input_grid = masked_grid[:, -T:, :, :, :]  # Last T steps as context
    #         step_output = self(input_grid)  # shape (B, C, H, W)
    #         predictions.append(step_output.unsqueeze(1))  # shape (B, 1, C, H, W)

    #         # Avoid in-place: update masked_grid with new time step
    #         masked_grid = torch.cat([masked_grid, step_output.unsqueeze(1)], dim=1)  # Append along time

    #     return torch.cat([data_context, *predictions], dim=1)
    
    def predict(self, input_grid, context_window, forcing_channels=None):
        """ 
        Forecasts the next time step given an input grid. 
        Duplication of the forward method to maintain consistency with the IonCastConvLSTM interface.
        The input grid is expected to be of shape (B, T, C, H, W), and the forward pass will reshape it to (B, T*C, H, W) for processing.

        Parameters
        ----------
        data_context : torch.Tensor
            Input tensor of shape (B, T, C, H, W), where:
            - B is the batch size, can be > 1
            - T is the context window length (number of time steps),
            - C is the number of grid node features (input_dim_grid_nodes),
            - H is the height of the grid (n_lat),
            - W is the width of the grid (n_lon).

        context window : int
            Number of time steps to pass into the model for each autoregressive step.
        """

        B, T, C, H, W = input_grid.shape
        prediction_window = T - context_window
        if forcing_channels is None:
            forcing_channels = torch.zeros(C, dtype=torch.bool).to(input_grid.device) # [C] if no forcing channels are passed in, all channels are predicted autoregressively

        # if forcing_channels.max() > 1 or len(forcing_channels) != C:  # NOTE: not the biggest fan of this check, specifically the max check but idea is to check if forcing_channels is a list of indices or a boolean mask
        if not isinstance(forcing_channels, torch.Tensor) or forcing_channels.dtype != torch.bool or len(forcing_channels) != C:  
            # If forcing_channels is a list of indices, convert it to a boolean mask
            forcing_mask = torch.zeros(C, dtype=torch.bool).to(input_grid.device) # [C]
            forcing_mask[forcing_channels] = True
            forcing_channels = forcing_mask # convert forcing_channels to bool mask

        data_context = input_grid[:, :context_window, :, :, :] # [B, context_window, C, H, W]
        forcing_context = input_grid[:, :, forcing_channels, :, :] # [B, T, len(forcing_channels), H, W]
        masked_grid = torch.zeros_like(input_grid).to(data_context.device)
        masked_grid[:, :context_window, :, :, :] = data_context
        masked_grid[:, :, forcing_channels, :, :] = forcing_context # the ground truth forcing channels will be available for all time
                                                                    # steps as these are assumed to have an analytical solution so they are not masked out

        for step in range(prediction_window): 
            # Pass context data through the model
            # The reason to detach / clone these is to avoid feeding into the model tensors that were altered in-place 
            input_grid = masked_grid[:, step:context_window+step, :, :, :].detach().clone() # [B, context_window, C, H, W]
            step_output = self(input_grid) # [B, 1*C, H, W]
            
            # only fill in the non-forcing channels, if no forcing channels are passed in, all channels are filled with model output autoregressively
            masked_grid[:, context_window+step, ~forcing_channels, :, :] = step_output[:, ~forcing_channels, :, :] 

        return masked_grid 
    
    # def predict(self, data_context, prediction_window=4):
    #     B, T, C, H, W = data_context.shape
    #     device = data_context.device

    #     # We'll collect predictions here
    #     predictions = []

    #     # Initialize the masked grid with the context
    #     masked_grid = data_context.clone()  # shape (B, T, C, H, W)

    #     for step in range(prediction_window): 
    #         input_grid = masked_grid[:, -T:, :, :, :]  # Last T steps as context
    #         step_output = self(input_grid)  # shape (B, C, H, W)
    #         predictions.append(step_output.unsqueeze(1))  # shape (B, 1, C, H, W)

    #         # Avoid in-place: update masked_grid with new time step
    #         masked_grid = torch.cat([masked_grid, step_output.unsqueeze(1)], dim=1)  # Append along time

    #     return torch.cat([data_context, *predictions], dim=1)
                         
    # def predict(self, data_context, prediction_window=4):
    #     """ Forecasts the next time step given the context window. """
    #     # data_context shape: (batch_size, time_steps, channels, height, width)
    #     # time steps = context_window
    #     x, hidden_state = self(data_context) # inits hidden state
    #     x = x.unsqueeze(1)  # shape (batch_size, time_steps=1, channels, height, width)
    #     prediction = [x]
    #     for _ in range(prediction_window - 1):
    #         # Prepare the next input by appending the last prediction
    #         x, hidden_state = self(x, hidden_state=hidden_state)
    #         x = x.unsqueeze(1)  # shape (batch_size, time_steps=1, channels, height, width)
    #         prediction.append(x)
    #     prediction = torch.cat(prediction, dim=1)  # shape (batch_size, prediction_window, channels, height, width)
    #     return prediction

    def loss(self, grid_features, channel_list=None, context_window=None, n_steps=1):
        """ 
        Computes the loss for the IonCastGraph model. 
        In GraphCast the loss is https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/utils/graphcast/loss.py
        For vTEC predictions, we can use a simple MSE loss between the predicted and target grid nodes.
        For now, loss is computed between all node features (vTEC, F10.7, cos(lat), etc)- this
        is maybe not the best choice, and we might want to compute the loss only on the vTEC feature.

        Parameters
        ----------
        grid_features : torch.Tensor
            Input tensor of shape (B, T, C, H, W)
        channel_list : list, optional
            List of channels to compute the loss on. If None, the loss is computed on all channels.
        context_window : int, optional
            If provided, the model will use the context window idx as the target for the loss.
        prediction_window : int, optional
        """

        B, T, C, H, W = grid_features.shape
        if context_window is None:
            context_window = T - 1
        
        
        input_grid = grid_features[:, :context_window, :, :, :] # shape (B, context_window, C, H, W)
        all_targets = grid_features[:, context_window:, :, :, :] # shape (B, T - context_window, C, H, W)

        output_grid = self.predict(input_grid, prediction_window=n_steps) # shape (B, T, C, H, W) 
        predictions_grid = output_grid[:, context_window:, :, :, :]

        assert predictions_grid.shape[1] == n_steps, f"Expected predictions_grid to have {n_steps} time steps, got {predictions_grid.shape[1]}" # this should be true as long as predict is working properly i think, as predict forms tensor masked_grid of shape T+n_steps along context dim

        if channel_list is not None:
            # If specific channels are provided, select them
            print(predictions_grid.shape, all_targets.shape)
            assert predictions_grid.shape[1] == n_steps, f"Expected predictions_grid to have {n_steps} time steps, got {predictions_grid.shape[1]}"
            assert all_targets.shape[1] == n_steps, f"Expected all_targets to have {n_steps} time steps, got {all_targets.shape[1]}"
            predictions_grid = predictions_grid[:, :, channel_list, :, :]
            all_targets = all_targets[:, :, channel_list, :, :]

        recon_loss = nn.functional.mse_loss(predictions_grid, all_targets, reduction='sum') # Sum over all pixels and channels
        recon_loss = recon_loss / (all_targets.shape[0] * all_targets.shape[1] * all_targets.shape[2]) # Average over batch size, channels, and time steps

        # For simplicity, we can return just the reconstruction loss
        return recon_loss
    

    # def loss(self, grid_features, channel_list=None, context_window=None, n_steps=1):
    #     """ 
    #     Computes the loss for the IonCastGraph model. 
    #     In GraphCast the loss is https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/utils/graphcast/loss.py
    #     For vTEC predictions, we can use a simple MSE loss between the predicted and target grid nodes.
    #     For now, loss is computed between all node features (vTEC, F10.7, cos(lat), etc)- this
    #     is maybe not the best choice, and we might want to compute the loss only on the vTEC feature.

    #     Parameters
    #     ----------
    #     grid_features : torch.Tensor
    #         Input tensor of shape (B, C, H, W)
    #     channel_list : list, optional
    #         List of channels to compute the loss on. If None, the loss is computed on all channels.
    #     context_window : int, optional
    #         If provided, the model will use the context window idx as the target for the loss.
    #     prediction_window : int, optional
    #     """
    #     if n_steps > 1:
    #         raise NotImplementedError("Multi step forcasting loss not supported yet")
        
    #     B, T, C, H, W = grid_features.shape
    #     if context_window is None:
    #         context_window = T - 1
        
        
    #     input_grid = grid_features[:, :context_window, :, :, :] # shape (B, context_window, C, H, W)
    #     all_targets = grid_features[:, context_window:, :, :, :] # shape (B, T - context_window, C, H, W)
    #     target_grid = all_targets[:, 0, :, :, :] # shape (B, C, H, W)

    #     if n_steps > 1:
    #         # for more than 1 step forcasting input grid should be the masked grid with the last entry filled with the prev prediction
    #         # its masked to make sure we arent accidentally using ground truth features in multi step forcasting
    #         masked_grid = grid_features
    #         masked_grid[:, context_window:] = masked_grid[:, context_window:] * 0

    #     # Forward pass
    #     output_grid = self(input_grid) # shape (B, context_window, C, H, W)
    #     if channel_list is not None:
    #         # If specific channels are provided, select them
    #         output_grid = output_grid[:, channel_list, :, :]
    #         target_grid = target_grid[:, channel_list, :, :]

    #     recon_loss = nn.functional.mse_loss(output_grid, target_grid, reduction='sum') # Sum over all pixels and channels
    #     recon_loss = recon_loss / (target_grid.shape[0] * target_grid.shape[1]) # Average over batch size and channels

    #     # For simplicity, we can return just the reconstruction loss
    #     return recon_loss