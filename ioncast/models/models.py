import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a soft way. Taken from Handful of Trials """
    result_tensor = min + nn.functional.softplus(tensor - min)
    return result_tensor


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


class VAE(nn.Module):
    def __init__(self, z_dim=512, sigma_vae=False):
        super().__init__()
        self.z_dim = z_dim
        self.sigma_vae = sigma_vae
        self.encoder = self.get_encoder()
        test_input = torch.randn(1, 1, 180, 360)
        test_output = self.encoder(test_input)
        dim_after_encoder = test_output.nelement()
        self.fc1 = nn.Linear(dim_after_encoder, z_dim)
        self.fc2 = nn.Linear(dim_after_encoder, z_dim)
        self.fc3 = nn.Linear(z_dim, dim_after_encoder)
        self.decoder = self.get_decoder()
        
    def get_encoder(self):
        raise NotImplementedError()
    
    def get_decoder(self):
        raise NotImplementedError()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar
    
    def decode(self, z):
        x = self.fc3(z)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Needs revisit and better implementation
    def sample(self, mu=None, logvar=None, n=None):
        if mu is not None or logvar is not None:
            if not (mu is not None and logvar is not None):
                raise ValueError('Must specify both mu and logvar')
            if n is not None:
                raise ValueError('Cannot specify both mu/logvar and n')
        
        if n is not None:
            device = list(self.parameters())[0].device
            mu = torch.zeros(n, self.z_dim).to(device)
            logvar = torch.zeros(n, self.z_dim).to(device)
            
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def loss(self, x):
        x_recon, mu, logvar = self.forward(x)
        # Gunes <gunes@robots.ox.ac.uk>, July 2023
        # It's important to get this correct. There are correct and incorrect VAE loss implementations in the wild.
        # Computing the means of CE and KL independently and then summing up is incorrect and leads to
        # poor results especially when data space has significantly more components than the latent space.
        # More information:
        # - "Loss implementation details" in the sigma-VAE paper: https://arxiv.org/abs/2006.13202
        # - Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 https://arxiv.org/abs/1312.6114
        # Reconstruction loss summed over all data components (log-likelihood of data)
        if self.sigma_vae:
            # Based on https://github.com/orybkin/sigma-vae-pytorch
            log_sigma = ((x - x_recon) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
            log_sigma = softclip(log_sigma, -6)
            rec = gaussian_nll(x_recon, log_sigma, x).sum()
        else:
            rec = nn.functional.mse_loss(x_recon, x, reduction='sum')
        # KL divergence summed over all latent components
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # It is optional to divide the total by the batch size. I think it's better to do it here.
        batch_size = x.shape[0]
        return (rec + kl) / batch_size
    

class VAE1(VAE):
    def get_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten()
            )
    
    def get_decoder(self):
        return nn.Sequential(
            nn.Unflatten(1, (32, 12, 23)),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=(1,0), output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=(2,3), output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=(2,3), output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            )
    


class ConvLSTMCell(nn.Module):
    """The core ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # Convolution for input, forget, output, and gate gates
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """Initializes the hidden and cell states."""
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    A ConvLSTM layer that processes a sequence.
    
    Args:
        input_dim (int): Number of channels in input.
        hidden_dim (int): Number of hidden channels.
        kernel_size (tuple): Size of the convolutional kernel.
        num_layers (int): Number of ConvLSTM layers.
        batch_first (bool): If True, input and output tensors are provided as (B, T, C, H, W).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        
        # Create a list of ConvLSTM cells
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                               hidden_dim=hidden_dim,
                                               kernel_size=kernel_size,
                                               bias=bias))

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)
            
        B, T, _, H, W = x.size()
        
        # Initialize hidden state for each layer
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=B, image_size=(H, W))
        
        layer_output_list = []
        last_state_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # We return the output of the last layer and the hidden states of all layers
        return layer_output_list[-1], last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class IonCastConvLSTM(nn.Module):
    """The final model for sequence-to-one prediction."""
    def __init__(self, input_channels=1, output_channels=1, hidden_dim=128, num_layers=4, context_window=4, prediction_window=4):
        super().__init__()
        # A stack of ConvLSTM layers
        self.conv_lstm = ConvLSTM(input_dim=input_channels, 
                                  hidden_dim=hidden_dim, 
                                  kernel_size=(3, 3), 
                                  num_layers=num_layers, # Number of stacked layers
                                  batch_first=True)
        
        # Final 1x1 convolution to get the desired number of output channels
        self.final_conv = nn.Conv2d(in_channels=hidden_dim, 
                                    out_channels=output_channels, 
                                    kernel_size=(1, 1))
        
        self.context_window = context_window  # Number of time steps in the input sequence during training
        self.prediction_window = prediction_window  # Number of time steps to predict during training

    def forward(self, x, hidden_state=None):
        # x shape: (B, T, C, H, W)
        
        # Pass through ConvLSTM
        # We only need the last hidden state to make the prediction
        _, hidden_state = self.conv_lstm(x, hidden_state=hidden_state)
        
        # Get the last hidden state of the last layer
        last_hidden_state = hidden_state[-1][0] # h_n of the last layer
        
        # Pass through the final convolution
        output = self.final_conv(last_hidden_state)
        
        return output, hidden_state
    
    def predict(self, data_context, prediction_window=4):
        """ Forecasts the next time step given the context window. """
        # data_context shape: (batch_size, time_steps, channels=1, height, width)
        # time steps = context_window
        x, hidden_state = self(data_context) # inits hidden state
        x = x.unsqueeze(1)  # shape (batch_size, time_steps=1, channels=1, height, width)
        prediction = [x]
        for _ in range(prediction_window - 1):
            # Prepare the next input by appending the last prediction
            x, hidden_state = self(x, hidden_state=hidden_state)
            x = x.unsqueeze(1)  # shape (batch_size, time_steps=1, channels=1, height, width)
            prediction.append(x)
        prediction = torch.cat(prediction, dim=1)  # shape (batch_size, prediction_window, channels=1, height, width)
        return prediction

    def loss(self, data, context_window=4):
        """ Computes the loss for the IonCastConvLSTM model. """
        # data shape: (batch_size, time_steps, channels=1, height, width)
        # time steps = context_window + prediction_window

        data_context = data[:, :context_window, :, :, :] # shape (batch_size, context_window, channels=1, height, width)
        data_target = data[:, context_window, :, :, :] # shape (batch_size, channels=1, height, width)

        # Forward pass
        data_predict, _ = self(data_context) # shape (batch_size, channels=1, height, width)
        recon_loss = nn.functional.mse_loss(data_predict, data_target, reduction='sum')
        
        # For simplicity, we can return just the reconstruction loss
        return recon_loss / data.size(0)
    
from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

class IonCastGraph(nn.Module):
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

    def forward(self, input_grid):
        """
        Passes the input grid through the GraphCast model.
        Forward method of GraphCast reshapes the input from [B, C, H, W] to be H x W grid nodes with 
        C features, and passes the grid nodes through the encoder, processor, and decoder,
        returns the updated grid nodes, and reshapes the output back to [B, C, H, W].

        Parameters
        ----------
        input_grid : torch.Tensor
            Input tensor of shape (B, C, H, W), where:
            - B is the batch size (must be 1 for GraphCast, see physicsnemo documentation),
            - C is the number of grid node features,
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
        B, C, H, W = input_grid.shape
        assert B == 1, "Batch size must be 1"

        # Pass through GraphCast
        output_grid = self.graph_cast(input_grid)

        return output_grid
    
    def predict(self, input_grid):
        """ 
        Forecasts the next time step given an input grid. 
        Duplication of the forward method to maintain consistency with the IonCastConvLSTM interface.
        """
        return self(input_grid)

    def loss(self, input_grid, target_grid, channel_list=None):
        """ 
        Computes the loss for the IonCastGraph model. 
        In GraphCast the loss is https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/utils/graphcast/loss.py
        For vTEC predictions, we can use a simple MSE loss between the predicted and target grid nodes.
        For now, loss is computed between all node features (vTEC, F10.7, cos(lat), etc)- this
        is maybe not the best choice, and we might want to compute the loss only on the vTEC feature.

        Parameters
        ----------
        input_grid : torch.Tensor
            Input tensor of shape (B, C, H, W)
        target_grid : torch.Tensor
            Target tensor of shape (B, C, H, W)
        channel_list : list, optional
            List of channels to compute the loss on. If None, the loss is computed on all channels.
            Use indexing to select specific channels from the input and target grids.
        """
        
        # Forward pass
        output_grid = self(input_grid) # shape (batch_size, channels=1, height, width)
        if channel_list is not None:
            # If specific channels are provided, select them
            output_grid = output_grid[:, channel_list, :, :]
            target_grid = target_grid[:, channel_list, :, :]
        else:
            recon_loss = nn.functional.mse_loss(output_grid, target_grid, reduction='sum')

        # For simplicity, we can return just the reconstruction loss
        return recon_loss / input_grid.size(0)
