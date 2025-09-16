import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

########################################
# 1. SFNO Core 
########################################

class SphericalFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes_lat, modes_lon):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_lat, modes_lon, 2) * 0.01
        )
    def compl_mul2d(self, x, w):
        w = torch.view_as_complex(w)
        return torch.einsum("bchw,cohw->bohw", x, w)
    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        x_ft = x_ft[:, :, :self.weight.shape[2], :self.weight.shape[3]]
        out_ft = self.compl_mul2d(x_ft, self.weight)
        out_ft_full = torch.zeros(B, self.weight.shape[1], H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft_full[:, :, :self.weight.shape[2], :self.weight.shape[3]] = out_ft
        return torch.fft.irfft2(out_ft_full, s=(H, W), norm="ortho")

class MCDropout(nn.Dropout):
    def __init__(self, p=0.35):
        super().__init__(p)
        self.mc = False
    def forward(self, x):
        return F.dropout(x, self.p, training=self.training or self.mc)

class SFNOBlock(nn.Module):
    def __init__(self, width, modes_lat, modes_lon, dropout=0.0, mc_dropout=False):
        super().__init__()
        self.fourier = SphericalFourierLayer(width, width, modes_lat, modes_lon)
        self.linear = nn.Conv2d(width, width, 1)
        self.dropout = MCDropout(dropout)
        self.dropout.mc = mc_dropout
        self.norm = nn.LayerNorm([width])
        self.act = nn.GELU()
    def set_mc_dropout(self, mc=True):
        self.dropout.mc = mc
    def forward(self, x):
        x1 = self.fourier(x)
        x2 = self.linear(x)
        out = x1 + x2
        out = self.act(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0, 3, 1, 2)
        return out

########################################
# 2. General Fourier Harmonic Positional Encoding 
########################################

def add_fourier_positional_encoding(
    x, coord_grids, n_harmonics=4, harmonics_range=(1, 4), concat_original_coords=True
):
    """
    x: (B, C, H, W)
    coord_grids: list of [H, W] numpy arrays, e.g. [lat_grid, lon_grid, ...]
    n_harmonics: number of Fourier harmonics per coordinate
    harmonics_range: tuple, (min_harmonic, max_harmonic)
    """
    B, C, H, W = x.shape
    channels = [x]
    for coord in coord_grids:
        # Normalize coord to [-pi, pi]
        coord_tensor = torch.tensor(coord, dtype=torch.float32, device=x.device)
        coord_norm = (coord_tensor - coord_tensor.min()) / (coord_tensor.max() - coord_tensor.min())
        coord_norm = coord_norm * 2 * np.pi - np.pi  # [-pi, pi]
        if concat_original_coords:
            coord_exp = coord_norm.unsqueeze(0).expand(B, -1, -1, -1)
            channels.append(coord_exp)
        # Harmonic stack
        for k in range(harmonics_range[0], harmonics_range[1]+1):
            for fn in [torch.sin, torch.cos]:
                hval = fn(k * coord_norm).unsqueeze(0).expand(B, -1, -1, -1)
                channels.append(hval)
    return torch.cat(channels, dim=1)

########################################
# 3. SFNO Model (unchanged except input stacking outside model)
########################################

class SphericalFourierNeuralOperatorModel(nn.Module):
    def __init__(
        self,
        in_channels: int, trunk_width: int = 64, trunk_depth: int = 6,
        modes_lat: int = 32, modes_lon: int = 64,
        aux_dim: int = 0, 
        tasks: tuple = ('vtec',), 
        out_shapes: dict = {'vtec': (1, 'grid')},
        probabilistic: bool = True, dropout: float = 0.2, mc_dropout: bool = False,
        n_sunlocked_heads: int = 360
    ):
        super().__init__()
        self.trunk_width = trunk_width
        self.tasks = tasks
        self.out_shapes = out_shapes
        self.in_proj = nn.Conv2d(in_channels, trunk_width, 1)
        self.blocks = nn.ModuleList([
            SFNOBlock(trunk_width, modes_lat, modes_lon, dropout=dropout, mc_dropout=mc_dropout)
            for _ in range(trunk_depth)
        ])
        self.aux_proj = nn.Linear(aux_dim, trunk_width) if aux_dim > 0 else None
        self.probabilistic = probabilistic
        self.n_sunlocked_heads = n_sunlocked_heads
        # This supports only one task for now, but ready for multiple.
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            out_dim, _ = self.out_shapes[task]
            self.out_dim = out_dim  # <--- ADDED, assumes one task
            out_channels = out_dim * (2 if self.probabilistic else 1)
            self.heads[task] = nn.ModuleList([
                nn.Conv2d(trunk_width, out_channels, 1) for _ in range(n_sunlocked_heads)
            ])

    def set_mc_dropout(self, mc=True):
        for b in self.blocks:
            b.set_mc_dropout(mc)

    def forward(self, x, sunlocked_lon_grid, aux=None):
        # x: (B, in_channels, H, W)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        #Add a line from Linnea here related to smoothing with some Gauss Party Stuff / import torchvision.transforms.functional as F
        #x = F.gaussian_blur(x, kernel_size=(5, 5), sigma=(1.0, 1.0))
        if self.aux_proj is not None and aux is not None:
            aux_emb = self.aux_proj(aux).unsqueeze(-1).unsqueeze(-1)
            x = x + aux_emb
        B, _, H, W = x.shape

        # Make sure sunlocked_lon_grid is a torch tensor, and on correct device
        if not torch.is_tensor(sunlocked_lon_grid):
            sunlocked_lon_grid = torch.tensor(sunlocked_lon_grid, device=x.device)
        # Ensure shape (B, H, W)
        if sunlocked_lon_grid.dim() == 2:
            sunlocked_lon_grid = sunlocked_lon_grid.unsqueeze(0).expand(B, -1, -1)
        sunlocked_lon_grid = sunlocked_lon_grid.clamp(0, self.n_sunlocked_heads - 1)

        outs = {}
        for task in self.tasks:
            out_dim, _ = self.out_shapes[task]
            out_channels = out_dim * (2 if self.probabilistic else 1)
            outputs = torch.zeros((B, out_channels, H, W), device=x.device)
            for head_idx in range(self.n_sunlocked_heads):
                for b in range(B):
                    mask_b = (sunlocked_lon_grid[b] == head_idx)  # (H, W)
                    if mask_b.sum() == 0:
                        continue
                    x_sub = x[b:b+1, :, :, :]  # (1, trunk_width, H, W)
                    out_sub = self.heads[task][head_idx](x_sub)  # (1, out_channels, H, W)
                    outputs[b, :, mask_b] = out_sub[0, :, mask_b]
            if self.probabilistic:
                mu, logvar = torch.split(outputs, out_dim, dim=1)
                outs[task] = (mu, logvar)
            else:
                outs[task] = outputs
        return outs


    @staticmethod
    def gaussian_nll(mu, log_sigma, x):
        return 0.5 * ((x - mu) / log_sigma.exp()).pow(2) + log_sigma + 0.5 * np.log(2 * np.pi)

    def loss(self, batch, sunlocked_lon_grid, context_window=4, reduction="mean"):
        data_context = batch[:, :context_window, ...]
        data_target  = batch[:, context_window, ...]
        B, T, C, H, W = data_context.shape
        x = data_context.reshape(B, T*C, H, W)
        outputs = self.forward(x, sunlocked_lon_grid)
        task = self.tasks[0]
        if self.probabilistic:
            mu, logvar = outputs[task]
            nll = self.gaussian_nll(mu, logvar, data_target)
            loss = nll.mean() if reduction == "mean" else nll.sum()
        else:
            y_pred = outputs[task]
            loss = F.mse_loss(y_pred, data_target, reduction=reduction)
        return loss

    def predict(self, data_context, sunlocked_lon_grid, prediction_window=4, aux=None):
        B, T, C, H, W = data_context.shape
        cur_input = data_context.reshape(B, T*C, H, W)
        preds = []
        task = self.tasks[0]
        for _ in range(prediction_window):
            out = self.forward(cur_input, sunlocked_lon_grid, aux=aux)
            if self.probabilistic:
                mu, logvar = out[task]
                preds.append(mu.unsqueeze(1))
                cur_input = torch.cat([cur_input[:, C:], mu], dim=1)
            else:
                y_pred = out[task]
                preds.append(y_pred.unsqueeze(1))
                cur_input = torch.cat([cur_input[:, C:], y_pred], dim=1)
        preds = torch.cat(preds, dim=1)
        return preds

########################################
# 4. Latitude Band Ensemble (smooth blend)
########################################

class LatBandSFNOEnsemble(nn.Module):
    def __init__(self, band_edges, base_model_args):
        super().__init__()
        self.bands = band_edges
        self.models = nn.ModuleList([
            SphericalFourierNeuralOperatorModel(**base_model_args) for _ in self.bands
        ])
        self.band_centers = [(b[0] + b[1]) / 2. for b in band_edges]

    def forward(self, x, lat_grid, sunlocked_lon_grid, aux=None):
        B, _, H, W = x.shape
        out = 0
        total_weight = 0
        lat_grid_expanded = lat_grid.unsqueeze(0) if lat_grid.ndim == 2 else lat_grid
        for i, (lat_min, lat_max) in enumerate(self.bands):
            center = self.band_centers[i]
            width = (lat_max - lat_min)
            band_weight = 1.0 - torch.abs(lat_grid_expanded - center) / (width / 2)
            band_weight = torch.clamp(band_weight, min=0.0)
            mask = (lat_grid_expanded >= lat_min) & (lat_grid_expanded < lat_max)
            band_weight = band_weight * mask
            if band_weight.sum() == 0:
                continue
            x_in = x.clone()
            x_in *= band_weight
            out += self.models[i](x_in, sunlocked_lon_grid, aux=aux) * band_weight
            total_weight += band_weight
        out = out / (total_weight + 1e-6)
        return out

########################################
# 5. Deep Ensemble Wrapper
########################################

class DeepEnsemble(nn.Module):
    def __init__(self, base_model_class, n_ensemble, base_model_args):
        super().__init__()
        self.models = nn.ModuleList([
            base_model_class(**base_model_args) for _ in range(n_ensemble)
        ])
    def set_mc_dropout(self, mc=True):
        for model in self.models:
            model.set_mc_dropout(mc)
    def forward(self, *args, **kwargs):
        return [model(*args, **kwargs) for model in self.models]
    def predict(self, *args, **kwargs):
        return torch.stack([model.predict(*args, **kwargs) for model in self.models], dim=0)