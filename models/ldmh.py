"""
Local Distribution Mixture Head (LDMH) -- COMBINED final version.

Combines BOTH confirmed-needed fixes into one file:
  1. Sigmoid-based beta bounding (fixes beta-collapse -- CONFIRMED working
     in real training: betas stayed diverse through epoch 6, no premature
     freezing at the boundary, unlike the old clamp-based version).
  2. Entropy-based load-balance penalty (fixes dead-component/usage-
     collapse -- the MSE version at bal_weight=0.05 was too weak; real
     training showed usage collapsing to [0.0, 0.0, 0.999] by epoch 7
     even while betas stayed diverse, proving these are two SEPARATE
     problems needing two separate fixes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDistributionMixtureHead(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, n_components=3, beta_min=0.3, beta_max=2.5):
        super().__init__()
        self.K = n_components
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.GroupNorm(8, base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.GroupNorm(8, base_ch), nn.GELU(),
        )
        self.weight_head = nn.Conv2d(base_ch, n_components, 1)

        init_betas = torch.linspace(beta_min + 0.1, beta_max - 0.1, n_components)
        init_fractions = (init_betas - beta_min) / (beta_max - beta_min)
        self.log_beta = nn.Parameter(torch.logit(init_fractions))
        self.log_scale = nn.Parameter(torch.zeros(n_components))

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.weight_head(feat)                 # [B, K, H, W]
        mix_weights = F.softmax(logits, dim=1)
        # Fix 1: smooth sigmoid bound, no dead-gradient zone (confirmed working)
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.log_beta)
        scale = F.softplus(self.log_scale) + 1e-3
        return mix_weights, beta, scale, feat

    @torch.no_grad()
    def get_component_stats(self, mix_weights, beta, scale):
        K = beta.shape[0]
        stats = {"beta": beta.detach().cpu().tolist(), "scale": scale.detach().cpu().tolist()}
        mean_weight = mix_weights.mean(dim=(0, 2, 3))
        stats["mean_usage"] = mean_weight.detach().cpu().tolist()
        dominant = mix_weights.argmax(dim=1)
        counts = torch.bincount(dominant.flatten(), minlength=K)
        stats["dominant_pixel_counts"] = counts.detach().cpu().tolist()
        pdist = torch.pdist(beta.unsqueeze(1))
        stats["min_beta_pairwise_dist"] = pdist.min().item() if K > 1 else float("nan")
        return stats


def mixture_gen_gaussian_nll(residual, mix_weights, beta, scale, eps=1e-6):
    K = beta.shape[0]
    r = residual.abs()
    if r.dim() == 3:
        r = r.unsqueeze(1)
    beta_ = beta.view(1, K, 1, 1)
    scale_ = scale.view(1, K, 1, 1)
    log_norm = torch.log(beta_) - torch.log(2 * scale_) - torch.lgamma(1.0 / beta_)
    log_comp = log_norm - (r / scale_).clamp(min=eps).pow(beta_)
    log_mix = torch.logsumexp(torch.log(mix_weights + eps) + log_comp, dim=1)
    return -log_mix.mean()


def beta_diversity_penalty(beta, margin=0.3):
    K = beta.shape[0]
    if K < 2:
        return torch.tensor(0.0, device=beta.device)
    pdist = torch.pdist(beta.unsqueeze(1))
    penalty = F.relu(margin - pdist)
    return penalty.mean()


def load_balance_penalty_entropy(mix_weights):
    """Fix 2: entropy-based balance penalty, stronger than MSE -- pushes
    harder specifically against near-zero usage than MSE-to-uniform did."""
    K = mix_weights.shape[1]
    mean_usage = mix_weights.mean(dim=(0, 2, 3)).clamp(min=1e-8)
    entropy = -(mean_usage * mean_usage.log()).sum()
    max_entropy = torch.log(torch.tensor(float(K), device=mix_weights.device))
    return max_entropy - entropy


class MixtureFiLMProjector(nn.Module):
    def __init__(self, n_components, feat_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_components, feat_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch * 2, 3, padding=1),
        )

    def forward(self, mix_weights):
        gamma, shift = self.net(mix_weights).chunk(2, dim=1)
        return gamma, shift