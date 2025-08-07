# model_s2convlstm.py

import torch
import torch.nn as nn
from s2cnn import S2Conv

class S2ConvLSTMCell(nn.Module):
    """The core S2ConvLSTM cell."""
    def __init__(self, input_dim, hidden_dim, bandwidth, bias=True):
        super(S2ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bandwidth = bandwidth
        self.bias = bias

        # A single spherical convolution for input, forget, output, and cell gates
        self.conv = S2Conv(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            b_in=self.bandwidth,
            b_out=self.bandwidth,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
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
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class S2ConvLSTM(nn.Module):
    """A multi-layer Spherical ConvLSTM."""
    def __init__(self, input_dim, hidden_dim, bandwidth, num_layers, batch_first=True, bias=True):
        super(S2ConvLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(S2ConvLSTMCell(input_dim=cur_input_dim,
                                                 hidden_dim=hidden_dim,
                                                 bandwidth=bandwidth,
                                                 bias=bias))

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        B, T, _, H, W = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=B, image_size=(H, W))

        cur_layer_input = x
        last_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            last_state_list.append([h, c])

        return cur_layer_input, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.cell_list]


class IonCastS2ConvLSTM(nn.Module):
    """The final S2ConvLSTM model, matching the API of IonCastConvLSTM."""
    def __init__(self, input_channels=17, output_channels=17, hidden_dim=64, num_layers=4, bandwidth=90, context_window=4, prediction_window=4):
        super().__init__()
        self.conv_lstm = S2ConvLSTM(input_dim=input_channels,
                                    hidden_dim=hidden_dim,
                                    bandwidth=bandwidth,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        # Final S2Conv to get the desired output channels
        self.final_conv = S2Conv(in_channels=hidden_dim,
                                 out_channels=output_channels,
                                 b_in=bandwidth,
                                 b_out=bandwidth)
        
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.bandwidth = bandwidth # Added for saving/loading

    def forward(self, x, hidden_state=None):
        _, hidden_state = self.conv_lstm(x, hidden_state=hidden_state)
        last_hidden_state = hidden_state[-1][0]
        output = self.final_conv(last_hidden_state)
        return output, hidden_state

    def predict(self, data_context, prediction_window=4):
        """Auto-regressive forecasting, identical to the original model's logic."""
        x, hidden_state = self(data_context)
        
        # The first input to the next step needs all 17 channels
        next_input = x.unsqueeze(1)
        
        predictions = [next_input]
        for _ in range(prediction_window - 1):
            x, hidden_state = self(next_input, hidden_state=hidden_state)
            next_input = x.unsqueeze(1)
            predictions.append(next_input)
            
        return torch.cat(predictions, dim=1)

    def loss(self, data, context_window=4):
        """Computes the loss for a single next-step prediction."""
        data_context = data[:, :context_window, ...]
        data_target = data[:, context_window, ...]
        
        data_predict, _ = self(data_context)
        
        # Your original loss focuses on the primary channel (0). Let's keep that.
        recon_loss = nn.functional.mse_loss(data_predict[:, 0, ...], data_target[:, 0, ...])
        
        return recon_loss