"""
Shared, full-featured training loop used by 05_Model_Training.ipynb (and
reusable from 06_Ablation_Study.ipynb). Implements: AdamW, cosine LR
schedule, mixed precision, gradient clipping, early stopping (on val PSNR),
per-epoch CSV logging, TensorBoard logging, best+last checkpointing
(with optimizer state, epoch, best PSNR/SSIM, full history), and
loss/PSNR/SSIM curve plotting.

UPDATED: now includes beta_diversity_penalty + load_balance_penalty to fix
the component-collapse issue (all K betas converging to the clamp ceiling,
one component going unused), plus per-epoch component-health logging.
"""
import os
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.losses import charbonnier_loss, gen_charbonnier_loss
from utils.metrics import psnr_torch, batch_ssim_torch
from models.ldmh import mixture_gen_gaussian_nll, beta_diversity_penalty, load_balance_penalty_entropy

try:
    import lpips as lpips_lib
    _LPIPS_IMPORT_OK = True
except ImportError:
    _LPIPS_IMPORT_OK = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lpips_model(device):
    if not _LPIPS_IMPORT_OK:
        print("WARNING: `lpips` package not installed (pip install lpips). "
              "LPIPS will be logged as NaN.")
        return None
    try:
        model = lpips_lib.LPIPS(net="alex").to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    except Exception as e:
        print(f"WARNING: could not initialize LPIPS ({e}). LPIPS will be logged as NaN.")
        return None


@torch.no_grad()
def compute_lpips(pred, target, lpips_model):
    if lpips_model is None:
        return float("nan")
    pred_c = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    target_c = target.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
    return lpips_model(pred_c, target_c).mean().item()


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def compute_loss(model_type, loss_type, pred_out, gt, gt_ds, noisy, global_beta, aux_weight,
                  div_weight=0.1, bal_weight=0.3):
    """model_type in {'plain','ldmh'}; loss_type in {'charbonnier','gen_charbonnier','mixture'}."""
    if model_type == "plain":
        pred = pred_out
        if loss_type == "charbonnier":
            loss = charbonnier_loss(pred, gt)
        elif loss_type == "gen_charbonnier":
            loss = gen_charbonnier_loss(pred, gt, p=global_beta)
        else:
            raise ValueError(f"loss_type={loss_type} invalid for model_type=plain")
        return loss, pred
    else:  # ldmh
        pred, mix_weights, beta, scale = pred_out
        recon = charbonnier_loss(pred, gt)
        lr_residual = noisy - gt_ds
        mix_nll = mixture_gen_gaussian_nll(lr_residual, mix_weights, beta, scale)

        # Fixes the component-collapse issue diagnosed after the original run.
        div_penalty = beta_diversity_penalty(beta, margin=0.3)
        bal_penalty = load_balance_penalty_entropy(mix_weights)

        loss = recon + aux_weight * mix_nll + div_weight * div_penalty + bal_weight * bal_penalty
        return loss, pred


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    model_type="plain",          # "plain" or "ldmh"
    loss_type="charbonnier",     # "charbonnier" | "gen_charbonnier" | "mixture"
    global_beta=0.845,
    aux_weight=1.0,
    div_weight=0.1,              # NEW: weight on the beta-diversity penalty
    bal_weight=0.3,              # raised from 0.05 -- MSE was too weak vs. NLL magnitude
    epochs=300,
    batch_size=8,
    lr=2e-4,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    patience=30,
    seed=42,
    ckpt_dir="../results/checkpoints",
    log_dir="../results/logs",
    device=None,
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("../results/curves", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    lpips_model = get_lpips_model(device)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))

    csv_path = os.path.join(ckpt_dir, f"{model_name}_history.csv")
    logger = CSVLogger(csv_path, fieldnames=[
        "epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "val_lpips",
        "lr", "time_sec"
    ])

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [],
               "val_lpips": [], "lr": [], "time_sec": []}

    best_psnr = -float("inf")
    best_ssim = -float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    training_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []

        for noisy, gt, gt_ds in train_loader:
            noisy, gt, gt_ds = noisy.to(device), gt.to(device), gt_ds.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, enabled=use_amp):
                pred_out = model(noisy)
                loss, _ = compute_loss(model_type, loss_type, pred_out, gt, gt_ds, noisy,
                                        global_beta, aux_weight, div_weight, bal_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        scheduler.step()

        # ---- validation ----
        model.eval()
        val_losses, psnrs, ssims, lpips_vals = [], [], [], []
        with torch.no_grad():
            for noisy, gt, gt_ds in val_loader:
                noisy, gt, gt_ds = noisy.to(device), gt.to(device), gt_ds.to(device)
                pred_out = model(noisy)
                loss, pred = compute_loss(model_type, loss_type, pred_out, gt, gt_ds, noisy,
                                           global_beta, aux_weight, div_weight, bal_weight)
                val_losses.append(loss.item())
                pred_clamped = pred.clamp(0, 1)
                psnrs.append(psnr_torch(pred_clamped, gt))
                ssims.append(batch_ssim_torch(pred_clamped, gt))
                lpips_vals.append(compute_lpips(pred_clamped, gt, lpips_model))

        epoch_time = time.time() - t0
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_psnr = float(np.mean(psnrs))
        val_ssim = float(np.mean(ssims))
        val_lpips = float(np.nanmean(lpips_vals)) if lpips_model is not None else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["val_lpips"].append(val_lpips)
        history["lr"].append(current_lr)
        history["time_sec"].append(epoch_time)

        logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "val_psnr": val_psnr, "val_ssim": val_ssim, "val_lpips": val_lpips,
                     "lr": current_lr, "time_sec": epoch_time})

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/PSNR", val_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", val_ssim, epoch)
        writer.add_scalar("Metrics/LPIPS", val_lpips, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(f"[{model_name}] epoch {epoch}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"PSNR={val_psnr:.3f}  SSIM={val_ssim:.4f}  LPIPS={val_lpips:.4f}  "
              f"lr={current_lr:.2e}  t={epoch_time:.1f}s")

        # NEW: per-epoch component-health check, only meaningful for LDMH models
        if model_type == "ldmh":
            with torch.no_grad():
                sample_noisy, _, _ = next(iter(val_loader))
                sample_noisy = sample_noisy.to(device)
                sample_out = model(sample_noisy)
                _, sample_mix_w, sample_beta, sample_scale = sample_out
                comp_stats = model.ldmh.get_component_stats(sample_mix_w, sample_beta, sample_scale)
                print(f"    [component check] beta={[round(b,3) for b in comp_stats['beta']]}  "
                      f"usage={[round(u,3) for u in comp_stats['mean_usage']]}  "
                      f"min_pairwise_dist={comp_stats['min_beta_pairwise_dist']:.3f}")

        improved = val_psnr > best_psnr
        if improved:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_psnr": best_psnr,
                "best_val_ssim": best_ssim,
                "history": history,
                "config": {
                    "model_type": model_type, "loss_type": loss_type,
                    "global_beta": global_beta, "aux_weight": aux_weight,
                    "div_weight": div_weight, "bal_weight": bal_weight,
                    "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                    "grad_clip_norm": grad_clip_norm, "seed": seed,
                },
            }, os.path.join(ckpt_dir, f"{model_name}.pth"))
        else:
            epochs_since_improvement += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "history": history,
        }, os.path.join(ckpt_dir, f"{model_name}_last.pth"))

        if epochs_since_improvement >= patience:
            print(f"[{model_name}] Early stopping at epoch {epoch} "
                  f"(no val_PSNR improvement for {patience} epochs).")
            break

    total_time = time.time() - training_start
    writer.close()

    _plot_curves(model_name, history)

    print(f"\n{'='*60}\n[{model_name}] TRAINING COMPLETE\n{'='*60}")
    print(f"Best Epoch:          {best_epoch}")
    print(f"Best Val PSNR:       {best_psnr:.3f}")
    print(f"Best Val SSIM:       {best_ssim:.4f}")
    print(f"Training Time:       {total_time/60:.1f} min")
    print(f"Checkpoint Location: {os.path.join(ckpt_dir, model_name + '.pth')}")
    print(f"CSV log:             {csv_path}")
    print(f"TensorBoard log:     {os.path.join(log_dir, model_name)}")

    return {
        "model_name": model_name, "best_epoch": best_epoch, "best_psnr": best_psnr,
        "best_ssim": best_ssim, "training_time_sec": total_time, "history": history,
        "checkpoint_path": os.path.join(ckpt_dir, f"{model_name}.pth"),
    }


def _plot_curves(model_name, history, out_dir="../results/curves"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name}: Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(history["val_psnr"], color="green")
    axes[1].set_title(f"{model_name}: Val PSNR"); axes[1].set_xlabel("epoch")

    axes[2].plot(history["val_ssim"], color="purple")
    axes[2].set_title(f"{model_name}: Val SSIM"); axes[2].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_curves.png"), dpi=150)
    plt.close(fig)