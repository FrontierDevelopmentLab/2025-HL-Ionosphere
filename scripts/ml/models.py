import torch
import torch.nn as nn
import numpy as np


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