import torch

def mae_loss(output, target):
    return torch.mean(torch.abs(output - target))
