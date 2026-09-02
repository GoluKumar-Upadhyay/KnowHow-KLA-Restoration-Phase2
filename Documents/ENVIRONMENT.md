# Environment Setup Guide

## Requirements
---

## 1. System Requirements

### Software

- Python 3.10+
- pip
- Git (recommended)

### Main Python Libraries

- PyTorch
- torchvision
- NumPy
- SciPy
- pandas
- Matplotlib
- Pillow
- scikit-image
- scikit-learn
- tqdm
- TensorBoard
- LPIPS
- THOP (optional)

---


## 1. Core dependencies (required on every setup)

```bash
pip install numpy scipy pandas matplotlib pillow jupyter
```

## 2. PyTorch (pick ONE based on your hardware)

### CPU only (no GPU)
```bash
pip install torch torchvision
```
Works everywhere, but training/inference will be slow (300-epoch full-scale training is not practical on CPU — use for testing/debugging only).

### GPU — consumer NVIDIA 
- RTX 3050
- RTX 3060
- RTX 4060
- RTX 4090
- A100
- H100

install the PyTorch CUDA build appropriate for the target environment.
For example, a CUDA 12.1 PyTorch build can be installed using:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Requires an NVIDIA driver supporting CUDA 12.1+. Verify with `nvidia-smi` before installing.

### GPU — datacenter NVIDIA H100 (e.g. judge/cluster environment)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
H100 requires a recent CUDA build (12.4+) and a recent PyTorch release (2.3+) for full Tensor Core / FP16 support. If the environment already has CUDA preinstalled, match the `cuXXX` suffix above to the installed CUDA version instead of assuming.

## 3. Optional (used by specific notebooks)

```bash
pip install lpips tensorboard thop
```
| Package | Used for | Skip if... |
|---|---|---|
| `lpips` | Perceptual loss metric during training (`utils/train.py`) | You don't need LPIPS logging — training still runs without it (logs `NaN`) |
| `tensorboard` | Training curve logging | You only need the printed console metrics |
| `thop` | FLOPs counting (`09_Inference_Benchmark.ipynb`) | A manual FLOP counter is used automatically as a fallback |

`lpips`'s AlexNet backbone weights download automatically on first use — requires internet access once.

## 4. Verify your setup

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 5. Minimum hardware

| Setup | RAM | VRAM | Notes |
|---|---|---|---|
| CPU only | 8 GB | — | Slow; smoke-testing only |
| Consumer GPU (RTX 3050 4GB+) | 8 GB | 4 GB | Verified working; batch size 8, ~68s/epoch |
| Datacenter GPU (H100) | 16 GB+ | 80 GB | Use `--batch_size` and `--half` flags in `inference.py` for max throughput |

## 6. Run inference (after installation)

```bash
python inference.py --input_dir <path_to_degraded_npy_folder> --output_dir <path_to_output_folder> --batch_size 16 --half
```
`--half` (FP16) is optional but recommended on GPU — gives ~1.7x throughput at negligible quality cost (verified on RTX 3050).

## Recommended Hardware

| Hardware | Inference | Training | Notes |
|---|---|---|---|
| CPU | Yes | Possible | Very slow for full training |
| RTX 3050 4GB | Yes | Yes | Use small batches/AMP when required |
| RTX 3060 12GB | Yes | Yes | More training memory |
| RTX 4090 24GB | Yes | Yes | Faster training |
| NVIDIA A100 | Yes | Yes | Suitable for large-scale training |
| NVIDIA H100 | Yes | Yes | Suitable for large-scale training |