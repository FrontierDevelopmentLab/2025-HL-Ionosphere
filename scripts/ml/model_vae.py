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
    