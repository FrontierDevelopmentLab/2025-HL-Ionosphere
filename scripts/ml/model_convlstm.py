import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """The core ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
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
    def __init__(self, input_channels=17, output_channels=17, hidden_dim=128, num_layers=4, context_window=4, prediction_window=4):
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
        # data_context shape: (batch_size, time_steps, channels, height, width)
        # time steps = context_window
        x, hidden_state = self(data_context) # inits hidden state
        x = x.unsqueeze(1)  # shape (batch_size, time_steps=1, channels, height, width)
        prediction = [x]
        for _ in range(prediction_window - 1):
            # Prepare the next input by appending the last prediction
            x, hidden_state = self(x, hidden_state=hidden_state)
            x = x.unsqueeze(1)  # shape (batch_size, time_steps=1, channels, height, width)
            prediction.append(x)
        prediction = torch.cat(prediction, dim=1)  # shape (batch_size, prediction_window, channels, height, width)
        return prediction

    def loss(self, data, context_window=4):
        """ Computes the loss for the IonCastConvLSTM model. """
        # data shape: (batch_size, time_steps, channels=1, height, width)
        # time steps = context_window + prediction_window

        data_context = data[:, :context_window, :, :, :] # shape (batch_size, context_window, channels=1, height, width)
        data_target = data[:, context_window, :, :, :] # shape (batch_size, channels, height, width)

        # Forward pass
        data_predict, _ = self(data_context) # shape (batch_size, channels=1, height, width)
        recon_loss = nn.functional.mse_loss(data_predict, data_target, reduction='sum')  # Sum over all pixels and channels
        recon_loss = recon_loss / (data_target.shape[0] * data_target.shape[1]) # Average over batch size and channels
        return recon_loss