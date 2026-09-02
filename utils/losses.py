"""Fixed (non-adaptive) loss functions used as ablation baselines."""
import torch


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def gen_charbonnier_loss(pred, target, p=0.845, eps=1e-3):
    """Fixed-exponent generalized Charbonnier, p from the GLOBAL residual fit (Notebook 02)."""
    return ((pred - target) ** 2 + eps ** 2).pow(p / 2).mean()
