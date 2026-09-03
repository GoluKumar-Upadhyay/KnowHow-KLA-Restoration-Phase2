
"""Restore degraded semiconductor images with the trained LDMH-FiLM model."""

import argparse
import glob
import os
import sys
import time

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.restoration_net import DistributionMixtureRestorationNet


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_input_files(input_dir):
    """Return sorted supported inputs, excluding macOS metadata files."""
    paths = []
    for ext in (".npy",) + IMAGE_EXTS:
        paths.extend(glob.glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return [
        path
        for path in sorted(set(paths))
        if "__MACOSX" not in path and not os.path.basename(path).startswith("._")
    ]


def load_input_array(path, target_size=128):
    """Load a grayscale input as float32; image files are normalized to [0, 1]."""
    if os.path.splitext(path)[1].lower() == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale array, got shape {array.shape}")
        return array

    image = Image.open(path).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_model(checkpoint_path, device):
    """Create the final architecture and load the saved trained weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = DistributionMixtureRestorationNet(
        base_ch=32,
        n_components=3,
        n_lr_blocks=4,
        n_hr_blocks=2,
        use_film=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint.get("epoch", "unknown")


def print_environment_info(device):
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"Device selected: {device}")
    print("=" * 60)


@torch.no_grad()
def run_batch(model, batch_np, device, use_half):
    """Restore one [B, H, W] batch and return arrays of the same batch shape."""
    inputs = torch.from_numpy(batch_np).unsqueeze(1).to(device, non_blocking=True)
    if use_half and device == "cuda":
        inputs = inputs.half()
    with torch.autocast(device_type=device, enabled=device == "cuda"):
        restored, _, _, _ = model(inputs)
    return restored.float().clamp(0, 1).cpu().numpy()[:, 0]


def save_restored_png(restored_np, output_path):
    """Save a normalized restored array as an 8-bit grayscale PNG."""
    restored_uint8 = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(restored_uint8, mode="L").save(output_path, format="PNG")


def save_comparison_grid(pairs, out_path):
    """Save a two-column noisy/restored comparison grid."""
    n_images = len(pairs)
    fig, axes = plt.subplots(n_images, 2, figsize=(6, 3 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, 2)

    for row, (name, degraded, restored) in enumerate(pairs):
        axes[row, 0].imshow(degraded, cmap="gray")
        axes[row, 0].set_title(f"Noisy: {name}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(restored, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Restored: {name}", fontsize=9)
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore degraded semiconductor images using LDMH-FiLM."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing degraded images.")
    parser.add_argument("--output_dir", required=True, help="Directory for restored PNG images.")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/DistributionMixtureRestorationNet.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Images per batch.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA.")
    parser.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'.")
    parser.add_argument("--grid_size", type=int, default=20, help="Pairs per comparison grid.")
    parser.add_argument("--no_grid", action="store_true", help="Do not save comparison grids.")
    return parser.parse_args()


def resolve_device(requested_device):
    device = requested_device.lower() if requested_device else "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cuda", "cpu"):
        raise ValueError("--device must be either 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def save_grid(grid_buffer, output_dir, grid_index):
    output_path = os.path.join(output_dir, f"comparison_grid_{grid_index:04d}.png")
    try:
        save_comparison_grid(grid_buffer, output_path)
        print(f"  saved {output_path} ({len(grid_buffer)} images)")
    except Exception as error:
        print(f"  [WARNING] Failed to save comparison grid: {error}")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.grid_size < 1:
        raise ValueError("--grid_size must be >= 1")

    device = resolve_device(args.device)
    use_half = args.half and device == "cuda"
    if args.half and not use_half:
        print("NOTE: --half is disabled because the selected device is CPU.")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    print_environment_info(device)
    print(f"\nLoading model from: {args.checkpoint}")
    model, checkpoint_epoch = load_model(args.checkpoint, device)
    if use_half:
        model = model.half()
    print(f"Model loaded successfully. Checkpoint epoch: {checkpoint_epoch}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    input_paths = list_input_files(args.input_dir)
    if not input_paths:
        print("\nNo supported input files found.")
        return

    print(f"\nFound {len(input_paths)} input images.")
    print(f"Input directory:  {os.path.abspath(args.input_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Batch size: {args.batch_size}; FP16: {use_half}; Comparison grid: {not args.no_grid}\n")

    n_processed = n_failed = grid_index = 0
    per_image_times, grid_buffer = [], []
    start_time = time.time()

    for batch_start in range(0, len(input_paths), args.batch_size):
        batch_paths = input_paths[batch_start : batch_start + args.batch_size]
        batch_arrays, valid_paths = [], []
        for path in batch_paths:
            try:
                batch_arrays.append(load_input_array(path))
                valid_paths.append(path)
            except Exception as error:
                print(f"  [SKIP] Failed to load {path}: {error}")
                n_failed += 1
        if not batch_arrays:
            continue

        try:
            batch_np = np.stack(batch_arrays)
        except Exception as error:
            print(f"  [ERROR] Could not stack batch: {error}")
            n_failed += len(valid_paths)
            continue

        if device == "cuda":
            torch.cuda.synchronize()
        batch_time_start = time.time()
        try:
            restored_batch = run_batch(model, batch_np, device, use_half)
        except Exception as error:
            print(f"  [ERROR] Batch starting at {valid_paths[0]} failed: {error}")
            n_failed += len(valid_paths)
            continue
        if device == "cuda":
            torch.cuda.synchronize()
        batch_time = time.time() - batch_time_start
        per_image_times.extend([batch_time / len(valid_paths)] * len(valid_paths))

        for path, degraded_np, restored_np in zip(valid_paths, batch_np, restored_batch):
            filename = os.path.splitext(os.path.basename(path))[0] + ".png"
            output_path = os.path.join(args.output_dir, filename)
            try:
                save_restored_png(restored_np, output_path)
            except Exception as error:
                print(f"  [ERROR] Failed to save {output_path}: {error}")
                n_failed += 1
                continue

            if not args.no_grid:
                grid_buffer.append((os.path.basename(path), degraded_np, restored_np))
                if len(grid_buffer) >= args.grid_size:
                    grid_index += 1
                    save_grid(grid_buffer, args.output_dir, grid_index)
                    grid_buffer = []
            n_processed += 1

        if n_processed % (args.batch_size * 5) < args.batch_size:
            print(f"  processed {n_processed}/{len(input_paths)}")

    if not args.no_grid and grid_buffer:
        save_grid(grid_buffer, args.output_dir, grid_index + 1)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Images found:             {len(input_paths)}")
    print(f"Images processed:         {n_processed}")
    print(f"Images failed:            {n_failed}")
    print(f"Total wall-clock time:    {total_time:.2f} sec")
    if per_image_times:
        mean_ms = np.mean(per_image_times) * 1000
        print(f"Mean per-image latency:   {mean_ms:.2f} ms")
        print(f"Throughput:               {1000 / mean_ms:.2f} images/sec")
    print(f"Restored PNG images:      {os.path.abspath(args.output_dir)}")
    if not args.no_grid:
        print(f"Comparison grid PNG(s):   {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
