import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """The core ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, dropout=0.0):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolution for input, forget, output, and gate gates with circular padding
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              padding_mode='circular',  # Use circular padding
                              bias=self.bias)
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout)

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

        # Apply dropout to the output hidden state
        h_next = self.dropout(h_next)

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
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True, dropout=0.0):
        super(ConvLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        
        # Create a list of ConvLSTM cells and GroupNorm layers
        self.cell_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                               hidden_dim=hidden_dim,
                                               kernel_size=kernel_size,
                                               bias=bias,
                                               dropout=dropout if i < self.num_layers - 1 else 0.0))
            # Add GroupNorm for all but the last layer's output
            if i < self.num_layers - 1:
                # Use GroupNorm with 1 group, which is equivalent to LayerNorm over channels
                self.norm_list.append(nn.GroupNorm(num_groups=1, num_channels=hidden_dim))


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
                # Apply GroupNorm to the hidden state before passing to the next layer
                if layer_idx < self.num_layers - 1:
                    h = self.norm_list[layer_idx](h)
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
    """The final model for sequence-to-sequence prediction."""
    def __init__(self, input_channels=17, output_channels=17, hidden_dim=128, num_layers=6, context_window=4, prediction_window=4, dropout=0.25):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.dropout = dropout

        # A stack of ConvLSTM layers
        self.conv_lstm = ConvLSTM(input_dim=input_channels, 
                                  hidden_dim=hidden_dim, 
                                  kernel_size=(5, 5), 
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout)
        
        # Final 1x1 convolution to get the desired number of output channels
        self.final_conv = nn.Conv2d(in_channels=hidden_dim, 
                                  out_channels=output_channels, 
                                  kernel_size=(1, 1))

    def forward(self, x, hidden_state=None):
        # x shape: (B, T, C, H, W)
        
        # Pass through ConvLSTM. layer_output is a sequence of hidden states from the last layer.
        layer_output, hidden_state = self.conv_lstm(x, hidden_state)
        # layer_output shape: (B, T, hidden_dim, H, W)
        
        # To apply the final 2D convolution, we need to reshape the output.
        # We treat the batch and time dimensions as one.
        B, T, C, H, W = layer_output.shape
        layer_output_flat = layer_output.view(B * T, C, H, W)
        
        # Pass each time step's hidden state through the final convolution
        output_flat = self.final_conv(layer_output_flat)
        
        # Reshape back to the sequence format
        output = output_flat.view(B, T, self.output_channels, H, W)
        
        return output, hidden_state

    def loss(self, data, jpld_channel_index=0, jpld_weight=1.0):
        """ Computes a weighted MSE loss for the IonCastConvLSTM model. """
        # data shape: (B, T_total, C, H, W)
        # For seq-to-seq, input is steps 0 to T-1, target is steps 1 to T
        data_input = data[:, :-1, :, :, :]
        data_target = data[:, 1:, :, :, :]

        # Forward pass
        data_predict, _ = self(data_input)

        # Calculate per-element squared error
        elementwise_loss = nn.functional.mse_loss(data_predict, data_target, reduction='none')

        # Create a weight tensor for the channels
        weights = torch.ones_like(data_target[0, 0, :, :, :]).unsqueeze(0).unsqueeze(0) # Shape (1, 1, C, H, W)
        weights[:, :, jpld_channel_index, :, :] = jpld_weight

        # Apply weights and calculate the mean
        loss = torch.mean(weights * elementwise_loss)
        
        # Calculate RMSE for diagnostics (unweighted)
        with torch.no_grad():
            rmse = torch.sqrt(nn.functional.mse_loss(data_predict, data_target, reduction='mean'))
            jpld_rmse = torch.sqrt(nn.functional.mse_loss(data_predict[:, :, jpld_channel_index, :, :], 
                                                          data_target[:, :, jpld_channel_index, :, :], reduction='mean'))
        
        return loss, rmse, jpld_rmse

    def predict(self, data_context, prediction_window=4):
        """ Forecasts the next time steps given the context window. """
        # data_context shape: (B, T_context, C, H, W)
        
        # First, process the context to get the initial hidden state
        _, hidden_state = self(data_context)
        
        # Get the last frame of the context as the first input for prediction
        next_input = data_context[:, -1:, :, :, :]
        
        prediction = []
        for _ in range(prediction_window):
            # Predict one step ahead
            output, hidden_state = self(next_input, hidden_state=hidden_state)
            
            # The output is the prediction, use it as the next input
            next_input = output
            prediction.append(output)
            
        prediction = torch.cat(prediction, dim=1)
        return prediction