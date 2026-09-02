"""PSNR / SSIM for both numpy arrays (analysis notebooks) and torch tensors (training loop)."""
import numpy as np
import torch
from scipy import ndimage


def psnr_np(a, b, data_range=1.0):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim_np(a, b, win=7, data_range=1.0):
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a = ndimage.uniform_filter(a, win)
    mu_b = ndimage.uniform_filter(b, win)
    sigma_a = ndimage.uniform_filter(a ** 2, win) - mu_a ** 2
    sigma_b = ndimage.uniform_filter(b ** 2, win) - mu_b ** 2
    sigma_ab = ndimage.uniform_filter(a * b, win) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2))
    return float(ssim_map.mean())


@torch.no_grad()
def psnr_torch(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return (20 * np.log10(data_range) - 10 * torch.log10(mse)).mean().item()


@torch.no_grad()
def batch_ssim_torch(pred, target, data_range=1.0):
    """Loops to numpy SSIM per-sample -- fine for validation-sized batches, not for training-time use."""
    vals = []
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    for i in range(p.shape[0]):
        vals.append(ssim_np(p[i, 0], t[i, 0], data_range=data_range))
    return float(np.mean(vals))
