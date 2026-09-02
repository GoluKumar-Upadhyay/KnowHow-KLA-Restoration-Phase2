
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from pathlib import Path


PROJECT_ROOT = Path("..").resolve()

DEGRADED_DIR = PROJECT_ROOT / "Test_NoisyLR" / "NoisyLR"
GT_DIR = PROJECT_ROOT / "train" / "train" / "GT"
TRAIN_DEGRADED_DIR = PROJECT_ROOT / "train" / "train" / "NoisyLR"

FILE_EXT = "*.npy"
MAX_IMAGES = 1000                                     
SATURATION_MARGIN = 0.001                            
                                                       



def load_image(path):
    """Load a single image as a float64 numpy array, preserving raw values."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        try:
            from PIL import Image
            img = Image.open(path)
            arr = np.array(img)
        except ImportError:
            raise RuntimeError(
                "Pillow not installed and file is not .npy. "
                "Install with: pip install Pillow --break-system-packages"
            )
    return arr.astype(np.float64)


def analyze_dataset(image_dir, ext_pattern, max_images=None):
    paths = sorted(glob.glob(os.path.join(image_dir, ext_pattern)))
    
    paths = [
        p for p in paths
        if "__MACOSX" not in p and not os.path.basename(p).startswith("._")
    ]
    if max_images:
        paths = paths[:max_images]

    if len(paths) == 0:
        print(f"[WARNING] No files found in {image_dir} matching {ext_pattern}")
        return None

    print(f"Found {len(paths)} files. Analyzing up to {len(paths)}...\n")

    per_image_stats = []
    all_pixel_samples = []  
    raw_dtype_seen = set()

    for i, p in enumerate(paths):
        try:
            arr = load_image(p)
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
            continue

        raw_dtype_seen.add(str(arr.dtype))
        flat = arr.flatten()
        if not np.all(np.isfinite(flat)):
            n_bad = int(np.sum(~np.isfinite(flat)))
            print(f"[WARNING] {p}: {n_bad} NaN/Inf values found — dropping them for stats")
            flat = flat[np.isfinite(flat)]
        if i == 0:
            print(f"  (sample shape: {arr.shape}, dtype: {arr.dtype})")
        vmin, vmax = float(flat.min()), float(flat.max())
        dynamic_range = vmax - vmin if vmax > vmin else 1.0

        
        near_sat_thresh = vmax - SATURATION_MARGIN * dynamic_range
        n_near_sat = int(np.sum(flat >= near_sat_thresh))
        pct_near_sat = 100.0 * n_near_sat / flat.size

        
        n_exact_max = int(np.sum(flat == vmax))
        pct_exact_max = 100.0 * n_exact_max / flat.size

        per_image_stats.append({
            "path": p,
            "dtype": str(arr.dtype),
            "min": vmin,
            "max": vmax,
            "pct_near_saturation": pct_near_sat,
            "pct_exact_max": pct_exact_max,
            "n_pixels": flat.size,
        })

        # subsample pixels for the aggregate histogram (cap memory)
        if flat.size > 20000:
            idx = np.random.choice(flat.size, 20000, replace=False)
            all_pixel_samples.append(flat[idx])
        else:
            all_pixel_samples.append(flat)

        if (i + 1) % 25 == 0 or (i + 1) == len(paths):
            print(f"  processed {i+1}/{len(paths)}")

    all_pixels = np.concatenate(all_pixel_samples)
    return per_image_stats, all_pixels, raw_dtype_seen


def print_summary(per_image_stats, all_pixels, raw_dtype_seen, label):
    print(f"\n{'='*60}\nSUMMARY: {label}\n{'='*60}")
    dtypes = raw_dtype_seen
    mins = [s["min"] for s in per_image_stats]
    maxs = [s["max"] for s in per_image_stats]
    near_sat = [s["pct_near_saturation"] for s in per_image_stats]
    exact_max = [s["pct_exact_max"] for s in per_image_stats]

    print(f"dtype(s) seen:              {dtypes}")
    print(f"global min / max:           {min(mins):.4f} / {max(maxs):.4f}")
    print(f"mean per-image min / max:   {np.mean(mins):.4f} / {np.mean(maxs):.4f}")
    print(f"std of per-image max:       {np.std(maxs):.6f}  "
          f"(near-zero std => maxima cluster on a repeated value => clipping signal)")
    print(f"mean %% pixels near max:     {np.mean(near_sat):.4f}%%")
    print(f"mean %% pixels == exact max: {np.mean(exact_max):.4f}%%  "
          f"(elevated => pile-up at ceiling => clipping signal)")

    return {
        "dtypes": dtypes,
        "max_values": np.array(maxs),
        "exact_max_pct": np.array(exact_max),
        "near_sat_pct": np.array(near_sat),
    }


def make_plots(all_pixels, per_image_stats, out_path="q1_diagnostic_plots.png"):
    maxs = np.array([s["max"] for s in per_image_stats])
    exact_max_pct = np.array([s["pct_exact_max"] for s in per_image_stats])
    near_sat_pct = np.array([s["pct_near_saturation"] for s in per_image_stats])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) aggregate histogram
    axes[0, 0].hist(all_pixels, bins=200, color="steelblue")
    axes[0, 0].set_title("Aggregate pixel-value histogram (degraded images)")
    axes[0, 0].set_xlabel("pixel value")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_yscale("log")

    # (b) CDF zoomed near the top
    sorted_vals = np.sort(all_pixels)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    top_cut = np.percentile(all_pixels, 90)
    mask = sorted_vals >= top_cut
    axes[0, 1].plot(sorted_vals[mask], cdf[mask])
    axes[0, 1].set_title("CDF, top 10% of pixel-value range\n(a vertical jump = pile-up = clipping)")
    axes[0, 1].set_xlabel("pixel value")
    axes[0, 1].set_ylabel("CDF")

    # (c) per-image max scatter
    axes[1, 0].scatter(range(len(maxs)), maxs, s=8, color="darkorange")
    axes[1, 0].set_title("Per-image maximum value\n(clustering on one value = clipping)")
    axes[1, 0].set_xlabel("image index")
    axes[1, 0].set_ylabel("max pixel value")

    # (d) exact-max pixel percentage per image
    axes[1, 1].hist(exact_max_pct, bins=40, color="seagreen")
    axes[1, 1].set_title("%% pixels exactly at per-image max, across images\n"
                          "(mass away from 0%% = pile-up = clipping)")
    axes[1, 1].set_xlabel("%% of pixels at exact max")
    axes[1, 1].set_ylabel("number of images")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved diagnostic plots to: {out_path}")


def verdict(summary):
    """
    Heuristic decision rule:
      - If per-image max values cluster tightly (low relative std) AND/OR
        a meaningful fraction of pixels sit exactly at the max repeatedly
        across images -> CLIPPING signature present.
      - Otherwise -> NO strong clipping signature; overshoot looks continuous.
    """
    maxs = summary["max_values"]
    exact_max_pct = summary["exact_max_pct"]

    rel_std = np.std(maxs) / (np.mean(np.abs(maxs)) + 1e-8)
    mean_exact_max_pct = np.mean(exact_max_pct)

    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"Relative std of per-image max values: {rel_std:.5f}")
    print(f"Mean %% of pixels exactly at max:       {mean_exact_max_pct:.4f}%%")

    clipped = (rel_std < 0.01) or (mean_exact_max_pct > 0.05)

    if clipped:
        print("\n>>> EVIDENCE SUPPORTS CLIPPING (censored observations).")
        print(">>> Maxima cluster tightly across images and/or a non-trivial")
        print(">>> fraction of pixels sit exactly at the ceiling repeatedly.")
        print(">>> -> Proceed with the Tobit / censored-likelihood (CESS-Net) framing.")
    else:
        print("\n>>> NO STRONG CLIPPING SIGNATURE DETECTED.")
        print(">>> Per-image maxima vary continuously and there is no pile-up")
        print(">>> at a repeated ceiling value.")
        print(">>> -> The 'censored observation' framing is likely NOT justified as-is.")
        print(">>> -> Reframe as heavy-tailed / multiplicative-outlier restoration:")
        print(">>>    values are displaced, not destroyed -- argue existing L1/L2")
        print(">>>    losses are miscalibrated for these outliers, rather than")
        print(">>>    arguing information is lost.")
    print(f"{'='*60}\n")
    return clipped


if __name__ == "__main__":
    if not os.path.isdir(DEGRADED_DIR):
        print(f"[ERROR] DEGRADED_DIR does not exist: {DEGRADED_DIR}")
        print("Edit DEGRADED_DIR (and FILE_EXT) at the top of this script, then re-run.")
        raise SystemExit(1)

    per_image_stats, all_pixels, dtypes = analyze_dataset(
        DEGRADED_DIR, FILE_EXT, MAX_IMAGES
    )
    summary = print_summary(per_image_stats, all_pixels, dtypes, "DEGRADED IMAGES")
    make_plots(all_pixels, per_image_stats)
    verdict(summary)

    if GT_DIR and os.path.isdir(GT_DIR):
        print("\nAlso analyzing ground-truth images for comparison...")
        gt_stats, gt_pixels, gt_dtypes = analyze_dataset(GT_DIR, FILE_EXT, MAX_IMAGES)
        print_summary(gt_stats, gt_pixels, gt_dtypes, "GROUND TRUTH IMAGES")
        print("\nCompare GT max/dtype range to degraded range above --")
        print("degraded values exceeding GT range is expected per the problem statement;")
        print("what matters for Q1 is whether THOSE excess values pile up (clipped)")
        print("or spread continuously (not clipped).")
