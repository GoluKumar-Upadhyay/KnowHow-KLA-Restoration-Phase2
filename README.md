<div align="center">
  <h1>🔍 KnowHow</h1>
  <h3>Distribution-Adaptive Restoration of Degraded Semiconductor Inspection Images</h3>
  <p><i>A Statistical Investigation and Corrected Mixture-Density Restoration Network</i></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Task-Image%20Restoration-brightgreen" alt="Task" />
    <img src="https://img.shields.io/badge/Hackathon-KLA%202026-orange" alt="Hackathon" />
  </p>
</div>

---

## 1. Phase 2 Submission

This repository contains the complete Phase 2 implementation of our solution for:

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

### Phase 2 Deliverables

- ✅ Final trained LDMH + FiLM model
- ✅ Standalone inference pipeline
- ✅ Reproducible evaluation workflow
- ✅ Statistical degradation analysis
- ✅ Ablation study
- ✅ Validation results and visual outputs
- ✅ Research documentation
- ✅ Final presentation and demo video

> Phase 2 extends the initial concept into a fully implemented and experimentally validated restoration system.

## 📊 Phase 2 Evidence

| Evidence | Phase 2 Result |
|---|---|
| Statistical Analysis | Non-Gaussian, heavy-tailed residual behavior |
| Spatial Analysis | Significant variation across local image regions |
| Proposed Architecture | Local Distribution Mixture Head + FiLM |
| Ablation | Progressive improvement across tested configurations |
| Best PSNR | **28.206 dB** |
| Best SSIM | **0.7542** |
| Lowest LPIPS | **0.2540** |
| Reproducibility | Standalone inference without source modification |

---

# 2. Final Solution

The submitted model is:

**DistributionMixtureRestorationNet**

The model combines:

```text
Degraded 128 × 128 Image
          │
          ▼
Local Distribution Mixture Head (LDMH)
          │
          ▼
FiLM Feature Conditioning
          │
          ▼
Compact NAFNet-style Restoration Backbone
          │
          ▼
PixelShuffle ×2
          │
          ▼
Restored 256 × 256 Image

````

### Model configuration

- LDMH components: **K = 3**
- Learned Generalized-Gaussian prototypes
- FiLM feature conditioning
- Compact NAFNet-style restoration backbone
- PixelShuffle ×2 upsampling
- Parameters: **116,138**

The final trained checkpoint is:

```
results/checkpoints/DistributionMixtureRestorationNet.pth

```

The checkpoint is included in the repository and is used automatically by `inference.py`.

**Training is not required to run the submitted model.**

---

# 3. Quick Start — Evaluation

The repository is designed so that the submitted model can be evaluated without modifying the source code.

## 3.1 Clone the repository

```
git clone https://github.com/GoluKumar-Upadhyay/KnowHow-KLA-Restoration-Phase2.git
cd KnowHow-KLA-Restoration-Phase2
```

## 3.2 Install dependencies

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
 


The script automatically loads:

```
results/checkpoints/DistributionMixtureRestorationNet.pth
```

No source-code modification or manual checkpoint-path editing is required.


# 4. Evaluation / Inference Pipeline

`inference.py` is the standalone evaluation entry point.

It performs the complete inference pipeline:

```
Input Directory
      │
      ▼
Load degraded images
      │
      ▼
Preprocessing
      │
      ▼
CPU → GPU
      │
      ▼
Model inference
      │
      ▼
GPU → CPU
      │
      ▼
Post-processing
      │
      ▼
Output Directory
```

The script:

1. Reads all valid input images from the supplied directory.
2. Loads the trained checkpoint automatically.
3. Selects CUDA when an NVIDIA GPU is available.
4. Performs restoration and 2× super-resolution.
5. Writes one restored output for each input image.
6. Preserves input filenames.

---

# 5. Input and Output Specification

## Input

The evaluation pipeline is designed for the KLA degraded test data.

Expected input:

```
Format: .npy
Type: float32
Channels: Grayscale
Resolution: 128 × 128
```

The degraded data contains the image degradation described by the challenge:

```
Speckle noise
+
Additive Gaussian noise
+
Spatial downsampling
```

## Output

For each valid input file, the inference script produces the corresponding restored output:

```
Input:
test_001.npy

Output:
test_001.npy
```

Output specification:

```
Format: .npy
Type: float32
Resolution: 256 × 256
Value range: [0, 1]
```

---

# 6. Environment and Dependencies

The Python package environment used for the submitted model is recorded in:

```
requirements.txt
```

It was generated using:

```
pip freeze > requirements.txt
```

The exact hardware/software setup and NVIDIA GPU installation instructions are documented in:

```
Documents/ENVIRONMENT_SETUP.md
```

Detailed inference usage and troubleshooting are documented in:

```
Documents/MODEL_USAGE_GUIDE.md
```

The project was developed and efficiency measurements were performed on an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

The competition's final inference benchmark is performed by KLA on its evaluation H100 GPU.

---

## Optional interactive prediction

For a quick visual check of a single image, use:

`notebooks/Quick_Load_And_Predict.ipynb`

The notebook provides a self-contained cell for:

- loading the trained checkpoint,
- loading a `.npy`, `.png`, `.jpg`, or `.jpeg` image,
- running restoration,
- displaying the degraded and restored images side by side.

This notebook is optional and is not part of the standalone evaluation pipeline.

# 7. Repository Structure

```
KnowHow-KLA-Restoration/
│
├── README.md
├── requirements.txt
├── inference.py
├── Documents/Research.docx                              <-- Detailed research documentation 
│
├── Documents/
│   ├── ENVIRONMENT_SETUP.md
│   ├── MODEL_USAGE_GUIDE.md
│   └── Research.docx
│
├── models/
│   ├── backbone.py
│   ├── ldmh.py
│   └── restoration_net.py
│
├── utils/
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   └── train.py
│
├── notebooks/
│   ├── 03_Local_Distribution_Analysis.ipynb
│   ├── 04_LDMH_Design.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Ablation_Study.ipynb
│   ├── 07_Comparison_and_OOD.ipynb
│   ├── 08_Visualization.ipynb
│   ├── 09_Inference_Benchmark.ipynb
│   ├── 01_Data_Analysis-1 .py 
    |── 01_Data_Analysis_2 .ipynb
    |── 01_Data_Analysis_3 .ipynb
    |── 02_Local_Distribution_Analysis.ipynb
    |── validates the statistical claim _ original _data.ipynb  
    |── validates the statistical claim_sybthesized .ipynb
    |── Proposed_Model_Testing.ipynb
│   └── Quick_Load_And_Predict.ipynb
|   |
│
├── results/
│   └── checkpoints/
│       └── DistributionMixtureRestorationNet.pth
│
└── Data_Analysis_Images/
```

### Key files

| File / DirectoryPurpose                  |                                            |
| ---------------------------------------- | ------------------------------------------ |
| `inference.py`                           | Standalone evaluation/inference script     |
| `requirements.txt`                       | Python environment dependencies            |
| `models/`                                | Restoration model implementation           |
| `utils/`                                 | Data, loss, metrics and training utilities |
| `results/checkpoints/`                   | Submitted trained checkpoint               |
| `notebooks/05_Model_Training.ipynb`      | Training pipeline                          |
| `notebooks/06_Ablation_Study.ipynb`      | Controlled ablation experiments            |
| `notebooks/07_Comparison_and_OOD.ipynb`  | Robustness and OOD experiments             |
| `notebooks/09_Inference_Benchmark.ipynb` | Inference efficiency experiments           |
| `Documents/ENVIRONMENT_SETUP.md`              | Environment and hardware setup            |
| `Documents/MODEL_USAGE_GUIDE.md`         | Detailed inference usage and troubleshooting            |
| `Documents/Research.docx`                | Detailed research documentation            |
| `notebooks/Quick_Load_And_Predict.ipynb` | Optional single-image interactive restoration demo            |

---

# 8. Training and Reproduction

Training is not required for evaluation because the final checkpoint is included.

The complete training process can be reproduced using:

```
notebooks/05_Model_Training.ipynb
```

Four models were trained under the same full-scale protocol:

| ModelConfiguration                  |                                         |
| ----------------------------------- | --------------------------------------- |
| `BaselineNet`                       | Backbone + plain Charbonnier loss       |
| `Baseline_GenCharbonnier`           | Backbone + fixed global heavy-tail loss |
| `LDMH`                              | Mixture loss + LDMH, FiLM disabled      |
| `DistributionMixtureRestorationNet` | Mixture loss + LDMH + FiLM              |

### Training configuration

| ParameterConfiguration |                  |
| ---------------------- | ---------------- |
| Optimizer              | AdamW            |
| LR schedule            | Cosine annealing |
| Mixed precision        | Enabled          |
| Gradient clipping      | Enabled          |
| Early stopping         | Patience 30      |
| Maximum epochs         | 300              |
| Batch size             | 8                |
| Seed                   | 42               |
| Training pairs         | 2,880            |
| Validation pairs       | 320              |
| Augmentation           | Flip / rotation  |

### LDMH configuration

| ParameterValue      |            |
| ------------------- | ---------- |
| Components          | 3          |
| Beta range          | [0.3, 2.5] |
| Softmax temperature | 1.5        |
| Diversity margin    | 0.3        |
| Mixture NLL weight  | 1.0        |
| Diversity weight    | 0.1        |
| Load-balance weight | 0.3        |

---

# 9. Research Summary

The project followed an **evidence-before-architecture** approach.

Instead of assuming a conventional Gaussian/L1/L2 restoration formulation, the degradation was statistically investigated first.

```
Real Data
   ↓
Hypothesis
   ↓
Statistical Analysis
   ↓
Finding
   ↓
Architecture / Loss Design
   ↓
Ablation
   ↓
Final Model

```

## 9.1 Censored-observation hypothesis

An initial hypothesis treated values exceeding the true image range as possible sensor clipping.

Analysis of:

- Pixel-value histograms
- Per-image maximum behavior
- Exact-maximum pixel pile-up

found no clipping signature.

The hypothesis was therefore rejected.

---

## 9.2 Global residual distribution

The residual was defined as:

```
Residual = NoisyLR - downsampled(GT)
```

Four candidate distributions were fitted to:

```
400 matched image pairs
6,553,600 residual pixels
```

Candidates:

- Gaussian
- Laplace
- Student-t
- Generalized Gaussian

The Generalized Gaussian achieved the best AIC with:

```
β = 0.845
```

Its AIC was approximately 54,700 better than the next-best candidate.

---

## 9.3 Local distribution analysis

The global distribution was further investigated using per-patch analysis.

Across:

```
51,200 valid 32 × 32 patches
```

the estimated local Generalized-Gaussian parameter was:

```
Mean β = 1.451
Std β  = 0.428
```

A one-way ANOVA showed significant spatial variation:

```
F = 18.06
p < 1e-300
```

Local beta also correlated with:

```
Edge density   r = -0.388
Local variance r = -0.550
Entropy        r = -0.430
```

with:

```
p < 1e-300
```

This finding motivated the use of a local mixture-based degradation representation rather than a single fixed global distribution.

---

## 9.4 Mixture-component collapse and correction

An initial implementation of the mixture model collapsed during training.

Two components converged to the clamp boundary:

```
β = 3.0
```

and one component became unused.

Direct gradient analysis identified a zero-gradient region caused by the hard `clamp()` operation.

The correction replaced the hard clamp with:

```
Smooth sigmoid reparameterization
```

and introduced:

```
Entropy-based load balancing
```

The corrected model converged to:

```
β ≈ [1.76, 2.20, 2.50]

Usage ≈ [0.19, 0.40, 0.41]
```

with stable behavior from approximately epoch 60 through epoch 200.

---

# 10. Final Ablation Results

## Full-scale validation

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| BaselineNet | 28.120 | 0.7513 | 0.2633 |
| Global Gen. Charbonnier | 28.124 | 0.7521 | 0.2609 |
| LDMH loss only | 28.139 | 0.7530 | 0.2614 |
| **Full LDMH + FiLM (Proposed)** | **28.206** | **0.7542** | **0.2540** |

**Key finding:** The full LDMH + FiLM model achieved the best result across all three validation metrics.

---

# 11. Per-Degradation-Type Robustness

Evaluation was performed on 30 held-out images.

| ModelSpeckleGaussianDownsampleCombined |            |            |            |            |
| -------------------------------------- | ---------- | ---------- | ---------- | ---------- |
| Baseline                               | 24.580     | 28.096     | 28.449     | 24.218     |
| + Global GC                            | 24.635     | **28.203** | **28.649** | **24.256** |
| LDMH                                   | 24.579     | 28.077     | 28.528     | 24.218     |
| **Full (corrected)**                   | **24.661** | 27.989     | 28.307     | 24.241     |

The corrected Full model achieved the highest PSNR on the speckle-only condition.

The fixed global loss remained strongest on three of the four tested conditions. Therefore, the advantage of the proposed model is reported as **degradation-specific rather than uniform**.

---

# 12. Inference Efficiency

Measured on:

```
NVIDIA RTX 3050 Laptop GPU
4 GB VRAM
```

| ModelParametersGFLOPsBatch 1Batch 16 FP16 |         |       |             |              |
| ----------------------------------------- | ------- | ----- | ----------- | ------------ |
| Baseline                                  | 86,945  | 3.898 | 59.33 img/s | 200.84 img/s |
| Full                                      | 116,138 | 4.844 | 54.89 img/s | 186.15 img/s |

Batch-16 FP16 throughput was approximately **1.72×** batch-1 throughput for both models.

These are local RTX 3050 measurements. They are **not H100 benchmark results**.

---

# 13. OOD Supporting Evaluation

A supporting OOD experiment was performed on:

```
10 external microscopy images
```

from the Carinthia-S dataset with synthetic degradation applied.

Results:

```
Mean PSNR = 35.87 dB
Std PSNR  = 2.64 dB

Mean SSIM = 0.746
Std SSIM  = 0.025
```

This experiment is treated as **supporting evidence only**, not proof of full domain generalization, because of the small sample size and external-domain setup.

No external dataset was used for model training.

---

# 14. Dataset

Training and competition evaluation use the KLA-provided data.

Training set:

```
3,200 matched GT + NoisyLR pairs
```

Split:

```
2,880 training
320 validation
```

with:

```
Seed = 42
```

No public/external dataset was used for model training.

---

# 15. Known Limitations

- **Statistical significance:** Results are from single training runs; small differences (0.01–0.1 dB) are not tested across multiple seeds.
- **Reduced-scale ablation:** Favored the simpler baseline at a small scale.
- **External comparison:** No end-to-end comparison against external architectures (e.g., Restormer, SwinIR).
- **Deployment:** ONNX export and TensorRT conversion are not implemented.
- **Mixture Usage:** The corrected LDMH component usage is not perfectly uniform (≈ 19% / 40% / 41%).

---

# 16. Detailed Research Documentation

For the complete scientific investigation, statistical derivations, model formulation, experiments and development history, see:

```
Documents/Research.docx
```

Relevant experiment notebooks:

```
notebooks/03_Local_Distribution_Analysis.ipynb
notebooks/04_LDMH_Design.ipynb
notebooks/05_Model_Training.ipynb
notebooks/06_Ablation_Study.ipynb
notebooks/07_Comparison_and_OOD.ipynb
notebooks/08_Visualization.ipynb
notebooks/09_Inference_Benchmark.ipynb
```

---

# 17. References

1. T. Kumar, R. Brennan, A. Mileo and M. Bendechache,
   **"Image Data Augmentation Approaches: A Comprehensive Survey and Future Directions,"** IEEE Access, 2024.
2. L. Zhai, Y. Wang, S. Cui and Y. Zhou,
   **"A Comprehensive Review of Deep Learning-Based Real-World Image Restoration,"** IEEE Access, 2023.
3. J. Terven, D. M. Cordova-Esparza, J. A. Romero-González et al.,
   **"A Comprehensive Survey of Loss Functions and Metrics in Deep Learning,"** Artificial Intelligence Review, 2025.
4. L. Chen, X. Chu, X. Zhang and J. Sun,
   **"Simple Baselines for Image Restoration,"** ECCV 2022.
5. T. Amemiya,
   **"Regression Analysis when the Dependent Variable is Truncated Normal,"** Econometrica, 1973.
6. **KLA Hackathon Problem Statement and provided GT + NoisyLR paired dataset.**

---

<div align="center">
  <i>Developed with ❤️ for KLA Hackathon 2026</i>
</div>
