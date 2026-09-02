"""Shared dataset utilities: loading, pairing, and a paired GT/NoisyLR Dataset class."""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def list_npy(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.npy")))
    return [p for p in paths if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]


def match_pairs(gt_dir, noisy_dir):
    gt_files = {os.path.basename(p): p for p in list_npy(gt_dir)}
    noisy_files = {os.path.basename(p): p for p in list_npy(noisy_dir)}
    common = sorted(set(gt_files) & set(noisy_files))
    return [(gt_files[k], noisy_files[k]) for k in common]


def load_npy(path):
    return np.load(path).astype(np.float32)


def downsample(img, out_hw):
    """Area-average downsample to match a target (H, W)."""
    H, W = img.shape[:2]
    oh, ow = out_hw
    fh, fw = H // oh, W // ow
    trimmed = img[:oh * fh, :ow * fw]
    return trimmed.reshape(oh, fh, ow, fw).mean(axis=(1, 3))


def split_pairs(pairs, val_frac=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_frac))
    val_idx = set(idx[:n_val])
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val


class PairedRestorationDataset(Dataset):
    """
    Returns (noisy_LR, gt_HR) tensors, both shape [1, H, W].
    gt_HR stays at its native (higher) resolution; noisy_LR stays at its native
    (lower) resolution -- the model is expected to upsample. Also returns a
    downsampled-GT tensor (matching noisy_LR resolution) for residual-based
    losses/analysis that need same-resolution targets.

    If augment=True, applies a randomly chosen geometric transform (flip
    horizontal, flip vertical, or 90/180/270 rotation) IDENTICALLY to noisy,
    gt, and gt_ds so they remain spatially aligned. Intensity is never
    altered -- geometric-only, so it doesn't distort the fitted noise
    statistics from Notebooks 02/03. Use augment=True for the training
    split and augment=False for validation/test.
    """
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _apply_transform(t, flip_h, flip_v, rot_k):
        if flip_h:
            t = torch.flip(t, dims=[-1])
        if flip_v:
            t = torch.flip(t, dims=[-2])
        if rot_k:
            t = torch.rot90(t, k=rot_k, dims=[-2, -1])
        return t

    def __getitem__(self, idx):
        gt_path, noisy_path = self.pairs[idx]
        gt = load_npy(gt_path)
        noisy = load_npy(noisy_path)
        gt_ds = downsample(gt, noisy.shape[:2])

        gt_t = torch.from_numpy(gt).unsqueeze(0).float()
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        gt_ds_t = torch.from_numpy(gt_ds).unsqueeze(0).float()

        if self.augment:
            flip_h = bool(np.random.rand() < 0.5)
            flip_v = bool(np.random.rand() < 0.5)
            rot_k = int(np.random.randint(0, 4))  # 0,1,2,3 -> 0/90/180/270 degrees
            gt_t = self._apply_transform(gt_t, flip_h, flip_v, rot_k)
            noisy_t = self._apply_transform(noisy_t, flip_h, flip_v, rot_k)
            gt_ds_t = self._apply_transform(gt_ds_t, flip_h, flip_v, rot_k)

        return noisy_t, gt_t, gt_ds_t


class PairedPatchDataset(Dataset):
    """
    Random-crop patch dataset for training (patch_size applies to the NoisyLR
    side; the GT crop is taken at `scale`x the size/location). Falls back to
    using the whole image if it's already <= patch_size (as is the case for
    this project's native 128x128 NoisyLR images). Applies random horizontal/
    vertical flip augmentation when train=True; deterministic center-crop
    (no flips) when train=False, for stable validation.
    """
    def __init__(self, pairs, patch_size=128, scale=2, train=True, seed=42):
        self.pairs = pairs
        self.patch_size = patch_size
        self.scale = scale
        self.train = train
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, noisy_path = self.pairs[idx]
        gt = load_npy(gt_path)
        noisy = load_npy(noisy_path)
        H, W = noisy.shape
        ps_h, ps_w = min(self.patch_size, H), min(self.patch_size, W)

        if H > ps_h or W > ps_w:
            if self.train:
                top = self.rng.randint(0, H - ps_h + 1)
                left = self.rng.randint(0, W - ps_w + 1)
            else:
                top, left = (H - ps_h) // 2, (W - ps_w) // 2
        else:
            top, left = 0, 0

        noisy_patch = noisy[top:top + ps_h, left:left + ps_w]
        gt_top, gt_left = top * self.scale, left * self.scale
        gt_patch = gt[gt_top:gt_top + ps_h * self.scale, gt_left:gt_left + ps_w * self.scale]

        if self.train:
            if self.rng.rand() < 0.5:
                noisy_patch, gt_patch = np.fliplr(noisy_patch).copy(), np.fliplr(gt_patch).copy()
            if self.rng.rand() < 0.5:
                noisy_patch, gt_patch = np.flipud(noisy_patch).copy(), np.flipud(gt_patch).copy()

        gt_ds_patch = downsample(gt_patch, noisy_patch.shape[:2])

        noisy_t = torch.from_numpy(noisy_patch).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_patch).unsqueeze(0).float()
        gt_ds_t = torch.from_numpy(gt_ds_patch).unsqueeze(0).float()
        return noisy_t, gt_t, gt_ds_t
