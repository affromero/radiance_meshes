import torch
import numpy as np

def contract_mean_std(x, std):
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    mask = x_mag_sq <= 1
    z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    # det_13 = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2)) ** (1 / 3)
    det_13 = (torch.pow(2 * x_mag_sqrt - 1, 1/3) / x_mag_sqrt) ** 2

    std = torch.where(mask[..., 0], std, det_13[..., 0] * std)
    return z, std
