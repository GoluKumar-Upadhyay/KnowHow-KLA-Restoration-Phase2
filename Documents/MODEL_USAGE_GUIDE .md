# Model Usage Guide

**DistributionMixtureRestorationNet** — trained restoration model for
degraded semiconductor inspection images.

This guide explains how to load, run, and verify the trained model on
CPU and NVIDIA GPU systems. Complete `ENVIRONMENT_SETUP.md` first.

---

## TL;DR — 3 Steps

```bash
# 1. Verify the checkpoint loads
python -c "import torch; from models.restoration_net import DistributionMixtureRestorationNet; m = DistributionMixtureRestorationNet(use_film=True); m.load_state_dict(torch.load('results/checkpoints/DistributionMixtureRestorationNet.pth', map_location='cpu')['model_state_dict']); print('OK')"

```
 
## Setup
```bash
pip install -r requirements.txt
```
No internet access, API keys, additional model downloads, or manual configuration are required at run time -- the checkpoint is included in `models/` and loaded automatically.
 
## Run
```bash
python run.py <input-dir> <output-dir>
```
**Example:**
```bash
python run.py /path/to/degraded /path/to/restored
```
 


# 3. Check the output folder for restored .npy files
```
Everything below expands on these three steps, with hardware-specific tuning.

---

## 1. Required Files

| Item | Location |
|---|---|
| Model implementation | `models/` |
| Utility code | `utils/` |
| Trained checkpoint | `results/checkpoints/DistributionMixtureRestorationNet.pth` |
| Inference script | `inference.py` (project root) |
| Interactive testing notebook | `notebooks/Proposed_Model_Testing.ipynb` |

**Both the checkpoint AND the `models/`/`utils/` code are required together.**
The checkpoint stores the trained weights and training metadata
(`model_state_dict`, `epoch`, `best_val_psnr`, `best_val_ssim`); it does
not store the model architecture itself — the code in `models/` must be
present to reconstruct the network before the weights can be loaded into it.

---

## 2. Command-Line Inference

```bash
python inference.py --input_dir <path_to_degraded_images> --output_dir <path_to_save_results>
```
**Example:**
```bash
python inference.py --input_dir ./test_data/NoisyLR --output_dir ./restored_output
```
The script auto-detects CUDA when available (override with `--device`).
It prints: PyTorch version, CUDA availability, GPU name, images
processed/failed, total time, mean latency, throughput, and output
location. Restored images are saved as `.npy` files, same filenames as
the input, values normalized to `[0, 1]`.

### Optional Arguments

| Argument | Purpose | Default |
|---|---|---|
| `--checkpoint <path>` | Use a different trained checkpoint | `results/checkpoints/DistributionMixtureRestorationNet.pth` |
| `--batch_size <N>` | Images processed per batch | `16` |
| `--half` | FP16 inference — **supported NVIDIA GPUs only** | off |
| `--device cpu` / `--device cuda` | Force a specific device | auto-detect |

---

## 3. Interactive Verification (Notebook)

```
notebooks/Proposed_Model_Testing.ipynb
```
--
```
notebooks/Quick_Load_And_Predict .ipynb
```
Run top to bottom. It loads the checkpoint, displays its metadata, shows
before/after restoration examples with intensity-histogram comparisons,
computes PSNR/SSIM where ground truth is available, and batch-processes
a full test set. It uses the identical model code as `inference.py` — a
visual companion, not a separate implementation.

### Minimal Load-and-Predict Cell

For a quick, standalone check without running the full notebook above,
paste this single cell into any Jupyter notebook opened at the project
root (or `notebooks/`, adjusting the path prefix accordingly):

```python
import sys, os
sys.path.insert(0, "..")   # remove this line if running from the project root, not notebooks/

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.restoration_net import DistributionMixtureRestorationNet

# ---- 1. Load the trained model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DistributionMixtureRestorationNet(base_ch=32, n_components=3,
                                            n_lr_blocks=4, n_hr_blocks=2, use_film=True)
checkpoint = torch.load("results/checkpoints/DistributionMixtureRestorationNet.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device).eval()
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
      f"(best PSNR={checkpoint['best_val_psnr']:.3f}, best SSIM={checkpoint['best_val_ssim']:.4f})")

# ---- 2. Load one degraded image (.npy, 128x128, float32) ----
IMAGE_PATH = "path/to/one/degraded_image.npy"   # <-- set this to a real file
degraded = np.load(IMAGE_PATH).astype(np.float32)

# ---- 3. Predict (restore) ----
with torch.no_grad():
    x = torch.from_numpy(degraded).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    restored, mix_weights, beta, scale = model(x)
    restored_np = restored[0, 0].clamp(0, 1).cpu().numpy()

print("Input shape:", degraded.shape, " Output shape:", restored_np.shape)

# ---- 4. Visualize ----
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(degraded, cmap="gray"); axes[0].set_title("Degraded (input)"); axes[0].axis("off")
axes[1].imshow(restored_np, cmap="gray"); axes[1].set_title("Restored (output)"); axes[1].axis("off")
plt.tight_layout()
plt.show()
```
This is the minimum code needed to load the model and get a restored
image back — useful for judges who want to inspect the model
interactively without running the full test/benchmark notebooks. This
same four-step pattern (load model → load image → predict → visualize)
is also available as a standalone notebook:
`notebooks/Quick_Load_And_Predict.ipynb`.

---

## 4. Hardware-Specific Instructions

### CPU (any machine, no GPU required)
```bash
python inference.py --input_dir <in> --output_dir <out> --device cpu --batch_size 1
```
**Measured on the development machine:** 15.35 images/sec (Full model),
65.1 ms/image, single image at a time. Actual speed on your machine will
vary with CPU model — expect substantially slower inference than GPU
regardless. Do not use `--half` on CPU; most CPUs gain nothing from it
and some operations may be unsupported.

### NVIDIA GPU — General
```bash
python inference.py --input_dir <in> --output_dir <out> --batch_size 16
```
Add `--half` if the GPU supports FP16 (all NVIDIA GPUs from the Pascal
generation onward, i.e. GTX 10-series and newer, do). The correct batch
size depends on available VRAM — reduce it if you hit an out-of-memory error.

### Verified Configuration: NVIDIA RTX 3050 (4GB, laptop)
This exact configuration was benchmarked during development:

| Setting | Measured Result |
|---|---|
| Batch 1, FP32 | 54.89 img/s (18.22 ms/image), 47.1 MB peak GPU memory |
| Batch 16, FP32 | 108.08 img/s |
| Batch 16, FP16 (`--half`) | 186.15 img/s (1.72x speedup over FP32) |

```bash
python inference.py --input_dir <in> --output_dir <out> --batch_size 16 --half
```
If CUDA out-of-memory occurs on a 4GB card, reduce to `--batch_size 4` or `2`.

### NVIDIA H100 (datacenter GPU)
```bash
python inference.py --input_dir <in> --output_dir <out> --batch_size 64 --half
```
**Not benchmarked on this hardware** — the batch size above is a
starting-point example, not a measured optimum. The model is lightweight
(116,138 parameters, 4.84 GFLOPs at 128x128 input, both measured via
`09_Inference_Benchmark.ipynb`), so H100's advantage will come primarily
from running larger batches in parallel rather than from per-image
speed. Increase `--batch_size` and measure throughput directly on the
target system to find the actual optimal setting; do not assume the
RTX 3050 figures above scale linearly to H100.

### IoT / Edge Devices (Jetson, Raspberry Pi, other ARM boards)
```bash
python inference.py --input_dir <in> --output_dir <out> --device cpu --batch_size 1
```
The model can potentially run on any device with a working PyTorch
installation for that architecture — **this depends on the specific
device, its OS, available RAM, and whether a compatible PyTorch build
exists for it, and must be verified per device rather than assumed.**
We have not tested this repository on IoT/edge hardware. The model's
small parameter count makes it a plausible candidate for such
deployment, but this is an expectation based on model size, not a
verified result.

**Not implemented in this repository:** ONNX export, INT8 quantization,
or TensorRT conversion — any of these would likely help further on
constrained hardware but have not been built, tested, or benchmarked
here, and should not be assumed to work without separate verification.

---

## 5. Verify the Checkpoint Loads Correctly

```bash
python -c "
import torch
from models.restoration_net import DistributionMixtureRestorationNet

model = DistributionMixtureRestorationNet(use_film=True)
checkpoint = torch.load('results/checkpoints/DistributionMixtureRestorationNet.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

print('Model loaded successfully.')
print('Trained epoch:', checkpoint['epoch'])
print('Best validation PSNR:', checkpoint['best_val_psnr'])
print('Best validation SSIM:', checkpoint['best_val_ssim'])
"
```
Expected: `Model loaded successfully.` with no errors, followed by the
three metadata lines.

## 6. Verify CUDA (GPU systems only)

```bash
nvidia-smi
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"
```
Expect `CUDA available: True` and a real GPU name. If `False` on a
GPU-equipped machine, reinstall PyTorch using the CUDA-enabled command
in `ENVIRONMENT_SETUP.md`.

---

## 7. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'models'` | Script run from the wrong directory | Run `inference.py` from the project root |
| `No .npy files found in <dir>` | Input path wrong or empty | Verify the path exists and contains `.npy` files |
| `CUDA available: False` on a GPU machine | CPU-only PyTorch build installed | Reinstall using the CUDA command in `ENVIRONMENT_SETUP.md` |
| CUDA out-of-memory | Batch size too large for available VRAM | Reduce `--batch_size` |
| Very slow GPU inference | Small batch size or FP32-only | Increase `--batch_size`; add `--half` if supported |
| Checkpoint load error / key mismatch | Model code version doesn't match checkpoint | Ensure `models/ldmh.py` matches the version used to train this checkpoint |
| LPIPS download failure (training only, not inference) | No internet access on first use | Not required for `inference.py`; only affects training-time logging |

---

## 8. Compatibility Summary

```
Trained checkpoint (weights only)
            |
            v
   models/ + utils/ code  ------>  reconstructs the network
            |
            v
   Same checkpoint loads on:
   +---------+--------------+---------------+
   |   CPU       |  NVIDIA GPU  |  H100 GPU |
   |environment  | environment  |environment|
   +---------+--------------+---------------+
            |
            v
   Restored image (identical output, different speed)
```
The same trained checkpoint produces the same restoration on any of
these targets — hardware changes inference **speed**, not the learned
parameters. Minor numerical differences (typically far below visual
significance) can occur across different hardware, PyTorch versions,
CUDA versions, and FP16 vs. FP32 precision.

---

