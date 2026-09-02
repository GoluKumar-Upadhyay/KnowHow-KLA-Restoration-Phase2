#!/usr/bin/env python3
"""
run.py -- entry point 
(AI-Based Restoration of Degraded Images), SEMICON Hackathon 2026.

USAGE (required, positional, exactly as specified):
    python run.py <input-dir> <output-dir>

Example:
    python run.py /path/to/degraded /path/to/restored


"""
import argparse
import os
import sys
import time
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.restoration_net import DistributionMixtureRestorationNet

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "checkpoint.pth"
)


def list_input_files(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    paths = [p for p in paths if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]
    return paths


def load_model(checkpoint_path, device):
    model = DistributionMixtureRestorationNet(base_ch=32, n_components=3,
                                                n_lr_blocks=4, n_hr_blocks=2, use_film=True)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Expected the trained model at models/checkpoint.pth relative to run.py."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt.get("epoch", "unknown")


def print_environment_info(device):
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Device selected: {device}")
    print("=" * 60)


@torch.no_grad()
def run_batch(model, batch_np, device, use_half):
    """batch_np: numpy array [B, H, W]. Returns restored numpy array [B, H2, W2]."""
    x = torch.from_numpy(batch_np).unsqueeze(1).to(device, non_blocking=True)
    if use_half and device == "cuda":
        x = x.half()
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        restored, _, _, _ = model(x)
    restored = restored.float().cpu().numpy()[:, 0]  # [B,H2,W2]
    return restored


def sanitize_output(arr):
    """Enforces the spec's hard output contract: no NaN/Inf, values in [0,1]."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Restore degraded semiconductor inspection images.")
    parser.add_argument("input_dir", type=str, help="Directory containing degraded .npy input images.")
    parser.add_argument("output_dir", type=str, help="Directory to write restored .npy output images.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="Path to the trained model checkpoint (default: models/checkpoint.pth).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--half", action="store_true", help="FP16 inference on supported GPUs.")
    parser.add_argument("--device", type=str, default=None, help="Force 'cuda' or 'cpu'. Default: auto-detect.")
    parser.add_argument("--save_grid", action="store_true",
                         help="Also save a combined comparison PNG for visual review (off by default).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print_environment_info(device)

    use_half = args.half
    if use_half and device == "cpu":
        print("NOTE: --half ignored on CPU (no benefit, some ops unsupported).\n")
        use_half = False

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading model from: {args.checkpoint}")
    model, ckpt_epoch = load_model(args.checkpoint, device)
    if use_half:
        model = model.half()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded (checkpoint epoch: {ckpt_epoch}, {n_params:,} parameters).")

    input_paths = list_input_files(args.input_dir)
    if len(input_paths) == 0:
        print(f"No .npy files found in {args.input_dir} -- nothing to do.")
        return
    print(f"\nFound {len(input_paths)} input images in {args.input_dir}")
    print(f"Batch size: {args.batch_size}  |  FP16: {use_half}\n")

    grid_buffer = []
    n_processed = 0
    n_failed = 0
    per_image_times = []
    t_start = time.time()

    for batch_start in range(0, len(input_paths), args.batch_size):
        batch_paths = input_paths[batch_start: batch_start + args.batch_size]
        batch_arrays, valid_paths = [], []
        for p in batch_paths:
            try:
                arr = np.load(p).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                batch_arrays.append(arr)
                valid_paths.append(p)
            except Exception as e:
                print(f"  [SKIP] Failed to load {p}: {e}")
                n_failed += 1

        if not batch_arrays:
            continue

        batch_np = np.stack(batch_arrays, axis=0)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            restored_batch = run_batch(model, batch_np, device, use_half)
        except Exception as e:
            print(f"  [ERROR] Batch starting at {valid_paths[0]} failed: {e}")
            n_failed += len(valid_paths)
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        batch_time = time.time() - t0
        per_image_times.extend([batch_time / len(valid_paths)] * len(valid_paths))

        for p, degraded_np, restored_np in zip(valid_paths, batch_np, restored_batch):
            restored_np = sanitize_output(restored_np)
            # Written DIRECTLY into output_dir, same filename as input -- per spec.
            out_path = os.path.join(args.output_dir, os.path.basename(p))
            np.save(out_path, restored_np)

            if args.save_grid:
                grid_buffer.append((os.path.basename(p), degraded_np, restored_np))

            n_processed += 1

        if n_processed % (args.batch_size * 5) < args.batch_size:
            print(f"  processed {n_processed}/{len(input_paths)}")

    if args.save_grid and grid_buffer:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(grid_buffer)
            fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
            if n == 1:
                axes = axes.reshape(1, 2)
            for row, (name, deg, res) in enumerate(grid_buffer):
                axes[row, 0].imshow(deg, cmap="gray"); axes[row, 0].set_title(f"Noisy: {name}", fontsize=9); axes[row, 0].axis("off")
                axes[row, 1].imshow(res, cmap="gray", vmin=0, vmax=1); axes[row, 1].set_title(f"Restored: {name}", fontsize=9); axes[row, 1].axis("off")
            plt.tight_layout()
            grid_path = os.path.join(args.output_dir, "_comparison_grid.png")
            plt.savefig(grid_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved optional comparison grid: {grid_path}")
        except ImportError:
            print("  [NOTE] --save_grid requested but matplotlib is unavailable; skipped (does not affect .npy outputs).")

    total_time = time.time() - t_start

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Images processed:      {n_processed}")
    print(f"Images failed:         {n_failed}")
    print(f"Total wall-clock time: {total_time:.2f} sec")
    if per_image_times:
        mean_ms = np.mean(per_image_times) * 1000
        throughput = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        print(f"Mean per-image latency: {mean_ms:.2f} ms")
        print(f"Throughput:             {throughput:.2f} images/sec")
    print(f"Output directory:      {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()