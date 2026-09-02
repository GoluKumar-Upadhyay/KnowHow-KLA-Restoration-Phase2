#!/usr/bin/env python3
"""
Standalone inference script -- the judged, distributable deliverable.

Loads the trained proposed model ONCE, then processes every degraded image
in --input_dir, restoring it and saving the restored results as PNG images
to --output_dir.

Accepts:
    .npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff

Input behavior:
    - .npy files are loaded directly as float32 arrays.
    - Image files are converted to grayscale.
    - Non-128x128 images are resized to 128x128.

Output behavior:
    - Every restored image is saved as a grayscale PNG.
    - Model output is converted from [0,1] float32 to [0,255] uint8.
    - Optional comparison grids are also generated.

Runs on CPU or CUDA GPU.
Device is auto-detected unless --device is explicitly specified.

USAGE:

    python inference.py --input_dir /path/to/degraded \
                       --output_dir /path/to/restored

    python inference.py --input_dir X \
                       --output_dir Y \
                       --checkpoint path/to/model.pth \
                       --batch_size 16 \
                       --half

    python inference.py --input_dir X \
                       --output_dir Y \
                       --grid_size 10

    python inference.py --input_dir X \
                       --output_dir Y \
                       --no_grid
"""

import argparse
import os
import sys
import time
import glob

import numpy as np
import torch

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Import proposed model
# ---------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.restoration_net import DistributionMixtureRestorationNet


# ---------------------------------------------------------------------
# Supported input formats
# ---------------------------------------------------------------------

IMAGE_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


# ---------------------------------------------------------------------
# Find all input files
# ---------------------------------------------------------------------

def list_input_files(input_dir):
    """
    Find all supported files in input_dir.

    Supported:
        .npy
        .png
        .jpg
        .jpeg
        .bmp
        .tif
        .tiff

    Both lowercase and uppercase extensions are supported.
    """

    paths = []

    for ext in (".npy",) + IMAGE_EXTS:

        # Lowercase
        paths.extend(
            glob.glob(
                os.path.join(input_dir, f"*{ext}")
            )
        )

        # Uppercase
        paths.extend(
            glob.glob(
                os.path.join(input_dir, f"*{ext.upper()}")
            )
        )

    # Remove duplicates and sort
    paths = sorted(set(paths))

    # Ignore macOS metadata files
    paths = [
        p for p in paths
        if "__MACOSX" not in p
        and not os.path.basename(p).startswith("._")
    ]

    return paths


# ---------------------------------------------------------------------
# Load input image / numpy array
# ---------------------------------------------------------------------

def load_input_array(path, target_size=128):
    """
    Load a degraded input.

    .npy:
        Loaded directly as float32.

    Image formats:
        Converted to grayscale and normalized to [0,1].
        Resized to 128x128 if necessary.

    Returns:
        numpy array [H,W], float32
    """

    ext = os.path.splitext(path)[1].lower()

    # ---------------------------------------------------------------
    # NumPy input
    # ---------------------------------------------------------------

    if ext == ".npy":

        arr = np.load(path).astype(np.float32)

        # If array has channels, use first channel
        if arr.ndim == 3:
            arr = arr[..., 0]

        # Validate dimensions
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D grayscale array, got shape {arr.shape}"
            )

        return arr

    # ---------------------------------------------------------------
    # Image input
    # ---------------------------------------------------------------

    else:

        img = Image.open(path).convert("L")

        # Resize to model input size
        if img.size != (target_size, target_size):

            img = img.resize(
                (target_size, target_size),
                Image.BILINEAR
            )

        # Convert to numpy
        arr = np.array(img).astype(np.float32)

        # Normalize uint8 image to [0,1]
        arr = arr / 255.0

        return arr


# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------

def load_model(checkpoint_path, device):

    """
    Construct and load the trained
    DistributionMixtureRestorationNet.
    """

    model = DistributionMixtureRestorationNet(
        base_ch=32,
        n_components=3,
        n_lr_blocks=4,
        n_hr_blocks=2,
        use_film=True
    )

    # Check checkpoint
    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Load model weights
    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    # Move model to selected device
    model = model.to(device)

    # Evaluation mode
    model.eval()

    return model, ckpt.get("epoch", "unknown")


# ---------------------------------------------------------------------
# Environment information
# ---------------------------------------------------------------------

def print_environment_info(device):

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)

    print(
        f"PyTorch version: {torch.__version__}"
    )

    print(
        f"CUDA available:  {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA version:    {torch.version.cuda}"
        )

        print(
            f"GPU:             {torch.cuda.get_device_name(0)}"
        )

        print(
            f"cuDNN version:   {torch.backends.cudnn.version()}"
        )

    print(
        f"Device selected: {device}"
    )

    print("=" * 60)


# ---------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------

@torch.no_grad()
def run_batch(model, batch_np, device, use_half):

    """
    batch_np:
        numpy array [B,H,W]

    Returns:
        restored numpy array [B,H,W]
    """

    # ---------------------------------------------------------------
    # NumPy -> PyTorch
    # ---------------------------------------------------------------

    x = torch.from_numpy(
        batch_np
    ).unsqueeze(1).to(
        device,
        non_blocking=True
    )

    # ---------------------------------------------------------------
    # FP16 input
    # ---------------------------------------------------------------

    if use_half and device == "cuda":

        x = x.half()

    # ---------------------------------------------------------------
    # Model inference
    # ---------------------------------------------------------------

    with torch.autocast(
        device_type=device,
        enabled=(device == "cuda")
    ):

        restored, _, _, _ = model(x)

    # ---------------------------------------------------------------
    # Convert output
    # ---------------------------------------------------------------

    restored = (
        restored
        .float()
        .clamp(0, 1)
        .cpu()
        .numpy()
        [:, 0]
    )

    return restored


# ---------------------------------------------------------------------
# Save restored image as PNG
# ---------------------------------------------------------------------

def save_restored_png(restored_np, output_path):
    """
    Convert model output from:

        float32 [0,1]

    to:

        uint8 [0,255]

    and save as grayscale PNG.
    """

    restored_uint8 = np.clip(
        restored_np * 255.0,
        0,
        255
    ).astype(np.uint8)

    img = Image.fromarray(
        restored_uint8,
        mode="L"
    )

    img.save(
        output_path,
        format="PNG"
    )


# ---------------------------------------------------------------------
# Save comparison grid
# ---------------------------------------------------------------------

def save_comparison_grid(pairs, out_path):

    """
    pairs:
        list of

        (
            filename,
            degraded_np,
            restored_np
        )

    Creates:

        [Noisy | Restored]

    for each image.
    """

    n = len(pairs)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(6, 3 * n)
    )

    # Handle single-image grid
    if n == 1:

        axes = axes.reshape(1, 2)

    # ---------------------------------------------------------------
    # Draw images
    # ---------------------------------------------------------------

    for row, (
        name,
        degraded,
        restored
    ) in enumerate(pairs):

        # -----------------------------------------------------------
        # Degraded
        # -----------------------------------------------------------

        axes[row, 0].imshow(
            degraded,
            cmap="gray"
        )

        axes[row, 0].set_title(
            f"Noisy: {name}",
            fontsize=9
        )

        axes[row, 0].axis("off")

        # -----------------------------------------------------------
        # Restored
        # -----------------------------------------------------------

        axes[row, 1].imshow(
            restored,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 1].set_title(
            f"Restored: {name}",
            fontsize=9
        )

        axes[row, 1].axis("off")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    # ===============================================================
    # Arguments
    # ===============================================================

    parser = argparse.ArgumentParser(
        description=(
            "Restore degraded semiconductor inspection "
            "images using DistributionMixtureRestorationNet."
        )
    )

    # ---------------------------------------------------------------
    # Input directory
    # ---------------------------------------------------------------

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Directory containing degraded input images "
            "(.npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff)."
        )
    )

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Directory where restored PNG images "
            "and optional comparison grids are saved."
        )
    )

    # ---------------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "results/checkpoints/"
            "DistributionMixtureRestorationNet.pth"
        ),
        help="Path to trained proposed-model checkpoint."
    )

    # ---------------------------------------------------------------
    # Batch size
    # ---------------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=(
            "Number of images processed per batch."
        )
    )

    # ---------------------------------------------------------------
    # Half precision
    # ---------------------------------------------------------------

    parser.add_argument(
        "--half",
        action="store_true",
        help=(
            "Use FP16 inference on CUDA GPUs "
            "for higher throughput."
        )
    )

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Force device: 'cuda' or 'cpu'. "
            "Default: automatically detect CUDA."
        )
    )

    # ---------------------------------------------------------------
    # Grid size
    # ---------------------------------------------------------------

    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help=(
            "Number of image pairs per comparison grid PNG."
        )
    )

    # ---------------------------------------------------------------
    # Disable comparison grid
    # ---------------------------------------------------------------

    parser.add_argument(
        "--no_grid",
        action="store_true",
        help=(
            "Skip comparison grid generation. "
            "Only restored PNG images are saved."
        )
    )

    args = parser.parse_args()

    # ===============================================================
    # Validate arguments
    # ===============================================================

    if args.batch_size < 1:

        raise ValueError(
            "--batch_size must be >= 1"
        )

    if args.grid_size < 1:

        raise ValueError(
            "--grid_size must be >= 1"
        )

    # ===============================================================
    # Select device
    # ===============================================================

    if args.device is not None:

        device = args.device.lower()

    else:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # Validate device
    if device not in ("cuda", "cpu"):

        raise ValueError(
            "--device must be either 'cuda' or 'cpu'"
        )

    # Prevent CUDA selection if unavailable
    if device == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA was explicitly requested, "
            "but CUDA is not available."
        )

    # ===============================================================
    # Environment
    # ===============================================================

    print_environment_info(device)

    # ===============================================================
    # FP16
    # ===============================================================

    use_half = args.half

    if use_half and device == "cpu":

        print(
            "\nNOTE: --half requested but device is CPU."
        )

        print(
            "FP16 is disabled because CPU FP16 "
            "provides no useful benefit.\n"
        )

        use_half = False

    # ===============================================================
    # CUDA optimization
    # ===============================================================

    if device == "cuda":

        torch.backends.cudnn.benchmark = True

    # ===============================================================
    # Create output directory
    # ===============================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ===============================================================
    # Load model ONCE
    # ===============================================================

    print(
        f"\nLoading model from: {args.checkpoint}"
    )

    model, ckpt_epoch = load_model(
        args.checkpoint,
        device
    )

    # Convert model to FP16 if requested
    if use_half:

        model = model.half()

    # Number of parameters
    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model loaded successfully."
    )

    print(
        f"Checkpoint epoch: {ckpt_epoch}"
    )

    print(
        f"Parameters: {n_params:,}"
    )

    # ===============================================================
    # Find input images
    # ===============================================================

    input_paths = list_input_files(
        args.input_dir
    )

    if len(input_paths) == 0:

        print(
            f"\nNo supported input files found in:"
        )

        print(
            f"{args.input_dir}"
        )

        print(
            "Supported formats:"
        )

        print(
            ".npy, .png, .jpg, .jpeg, .bmp, .tif, .tiff"
        )

        return

    # ===============================================================
    # Input information
    # ===============================================================

    print(
        f"\nFound {len(input_paths)} input images."
    )

    print(
        f"Input directory:  "
        f"{os.path.abspath(args.input_dir)}"
    )

    print(
        f"Output directory: "
        f"{os.path.abspath(args.output_dir)}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"FP16: {use_half}"
    )

    print(
        f"Comparison grid: {not args.no_grid}"
    )

    print()

    # ===============================================================
    # Statistics
    # ===============================================================

    n_processed = 0
    n_failed = 0

    per_image_times = []

    grid_buffer = []

    grid_file_idx = 1

    t_start = time.time()

    # ===============================================================
    # Process images in batches
    # ===============================================================

    for batch_start in range(
        0,
        len(input_paths),
        args.batch_size
    ):

        batch_paths = input_paths[
            batch_start:
            batch_start + args.batch_size
        ]

        batch_arrays = []
        valid_paths = []

        # -----------------------------------------------------------
        # Load batch
        # -----------------------------------------------------------

        for p in batch_paths:

            try:

                arr = load_input_array(p)

                batch_arrays.append(arr)

                valid_paths.append(p)

            except Exception as e:

                print(
                    f"  [SKIP] Failed to load "
                    f"{p}: {e}"
                )

                n_failed += 1

        # Nothing valid in this batch
        if not batch_arrays:

            continue

        # -----------------------------------------------------------
        # Stack batch
        # -----------------------------------------------------------

        try:

            batch_np = np.stack(
                batch_arrays,
                axis=0
            )

        except Exception as e:

            print(
                f"  [ERROR] Could not stack batch: {e}"
            )

            n_failed += len(valid_paths)

            continue

        # -----------------------------------------------------------
        # GPU synchronization
        # -----------------------------------------------------------

        if device == "cuda":

            torch.cuda.synchronize()

        # -----------------------------------------------------------
        # Start timing
        # -----------------------------------------------------------

        t0 = time.time()

        # -----------------------------------------------------------
        # Run inference
        # -----------------------------------------------------------

        try:

            restored_batch = run_batch(
                model,
                batch_np,
                device,
                use_half
            )

        except Exception as e:

            print(
                f"  [ERROR] Batch starting at "
                f"{valid_paths[0]} failed: {e}"
            )

            n_failed += len(valid_paths)

            continue

        # -----------------------------------------------------------
        # GPU synchronization
        # -----------------------------------------------------------

        if device == "cuda":

            torch.cuda.synchronize()

        # -----------------------------------------------------------
        # Batch timing
        # -----------------------------------------------------------

        batch_time = time.time() - t0

        per_image_times.extend(
            [
                batch_time / len(valid_paths)
            ]
            * len(valid_paths)
        )

        # ===========================================================
        # Save each restored image
        # ===========================================================

        for (
            p,
            degraded_np,
            restored_np
        ) in zip(
            valid_paths,
            batch_np,
            restored_batch
        ):

            # -------------------------------------------------------
            # Original filename without extension
            # -------------------------------------------------------

            base = os.path.splitext(
                os.path.basename(p)
            )[0]

            # -------------------------------------------------------
            # Output PNG path
            # -------------------------------------------------------

            png_path = os.path.join(
                args.output_dir,
                base + ".png"
            )

            # -------------------------------------------------------
            # Save restored PNG
            # -------------------------------------------------------

            try:

                save_restored_png(
                    restored_np,
                    png_path
                )

            except Exception as e:

                print(
                    f"  [ERROR] Failed to save "
                    f"{png_path}: {e}"
                )

                n_failed += 1

                continue

            # -------------------------------------------------------
            # Comparison grid
            # -------------------------------------------------------

            if not args.no_grid:

                grid_buffer.append(
                    (
                        os.path.basename(p),
                        degraded_np,
                        restored_np
                    )
                )

                # ---------------------------------------------------
                # Save full grid
                # ---------------------------------------------------

                if len(grid_buffer) >= args.grid_size:

                    out_path = os.path.join(
                        args.output_dir,
                        f"comparison_grid_"
                        f"{grid_file_idx:04d}.png"
                    )

                    try:

                        save_comparison_grid(
                            grid_buffer,
                            out_path
                        )

                        print(
                            f"  saved {out_path} "
                            f"({len(grid_buffer)} images)"
                        )

                    except Exception as e:

                        print(
                            f"  [WARNING] Failed to save "
                            f"comparison grid: {e}"
                        )

                    grid_buffer = []

                    grid_file_idx += 1

            # -------------------------------------------------------
            # Count successful image
            # -------------------------------------------------------

            n_processed += 1

        # ===========================================================
        # Progress
        # ===========================================================

        if (
            n_processed
            % (args.batch_size * 5)
            < args.batch_size
        ):

            print(
                f"  processed "
                f"{n_processed}/"
                f"{len(input_paths)}"
            )

    # ===============================================================
    # Save remaining comparison grid
    # ===============================================================

    if (
        not args.no_grid
        and grid_buffer
    ):

        out_path = os.path.join(
            args.output_dir,
            f"comparison_grid_"
            f"{grid_file_idx:04d}.png"
        )

        try:

            save_comparison_grid(
                grid_buffer,
                out_path
            )

            print(
                f"  saved {out_path} "
                f"({len(grid_buffer)} images)"
            )

        except Exception as e:

            print(
                f"  [WARNING] Failed to save "
                f"final comparison grid: {e}"
            )

    # ===============================================================
    # Total time
    # ===============================================================

    total_time = (
        time.time()
        - t_start
    )

    # ===============================================================
    # Final report
    # ===============================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "INFERENCE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Images found:             "
        f"{len(input_paths)}"
    )

    print(
        f"Images processed:         "
        f"{n_processed}"
    )

    print(
        f"Images failed:            "
        f"{n_failed}"
    )

    print(
        f"Total wall-clock time:    "
        f"{total_time:.2f} sec"
    )

    # ---------------------------------------------------------------
    # Timing
    # ---------------------------------------------------------------

    if per_image_times:

        mean_ms = (
            np.mean(per_image_times)
            * 1000
        )

        throughput = (
            1000.0 / mean_ms
            if mean_ms > 0
            else float("inf")
        )

        print(
            f"Mean per-image latency:   "
            f"{mean_ms:.2f} ms"
        )

        print(
            f"Throughput:               "
            f"{throughput:.2f} images/sec"
        )

    # ---------------------------------------------------------------
    # Output location
    # ---------------------------------------------------------------

    print(
        f"Restored PNG images:      "
        f"{os.path.abspath(args.output_dir)}"
    )

    if not args.no_grid:

        print(
            f"Comparison grid PNG(s):   "
            f"{os.path.abspath(args.output_dir)}"
        )

    print(
        "=" * 60
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    main()