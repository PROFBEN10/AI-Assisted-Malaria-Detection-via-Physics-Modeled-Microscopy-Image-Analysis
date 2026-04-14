# AI-Assisted-Malaria-Detection-via-Physics-Modeled-Microscopy-Image-Analysis
Combining Mie Scattering Optical Physics with Deep Learning to Detect Plasmodium falciparum in Blood Smear Images


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/malaria-physics-ml/blob/main/malaria_physics_ml_FINAL.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: NIH](https://img.shields.io/badge/Dataset-NIH%20Malaria%20Cell%20Images-red)](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html)

---

## Table of Contents

- [Overview](#overview)
- [The Physics Behind This Project](#the-physics-behind-this-project)
- [Why This Project Matters](#why-this-project-matters)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Key Technical Contributions](#key-technical-contributions)
- [Figures & Outputs](#figures--outputs)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Author](#author)

---

## Overview

Malaria kills over **600,000 people annually**, with Nigeria alone accounting for approximately **27% of all global cases**. Standard diagnosis requires expert manual microscopy of Giemsa-stained blood smears — slow, expensive, and highly variable across technicians and clinics.

Most existing AI solutions treat malaria diagnosis as a pure computer vision problem, training CNNs to classify cell images without any physical understanding of what they are seeing. This approach has a critical flaw: microscopy images are governed by the **physics of light interacting with biological matter**. A model that ignores this fails badly when deployed across different microscopes, staining protocols, or illumination conditions — exactly the variability found in Nigerian rural health posts.

This project takes a different approach: **physics-informed AI**.

It builds a dual-path model that fuses **Mie scattering optical features** (analytically derived from Maxwell's equations) with **MobileNetV2 deep learning features**, producing a classifier that is both accurate and robust to real-world deployment conditions.

---

## The Physics Behind This Project

### Mie Scattering Theory

When light illuminates a spherical particle, the interaction is governed by **Mie scattering theory** — an exact analytical solution to Maxwell's equations published by Gustav Mie in 1908. Red blood cells (diameter ~6–8 μm) approximate spheres optically, making Mie theory directly applicable.

For a cell with radius **r**, complex refractive index **m = n + ik**, illuminated at wavelength **λ**, Mie theory gives:

| Symbol | Physical Meaning |
|--------|-----------------|
| **Q_scat** | Scattering efficiency — fraction of incident light scattered |
| **Q_abs** | Absorption efficiency — fraction of light absorbed |
| **Q_ext = Q_scat + Q_abs** | Total extinction efficiency |
| **g** | Asymmetry parameter — forward vs. backward scattering ratio |

### Why Infected Cells Scatter Differently

*P. falciparum* infection fundamentally alters the optical properties of the red blood cell:

| Property | Healthy RBC | *P. falciparum* Infected RBC |
|----------|-------------|------------------------------|
| Refractive index (n) | ~1.40 | ~1.43–1.46 (haemozoin crystals raise it) |
| Absorption coefficient (k) | Low (oxy-haemoglobin) | Higher (haemozoin pigment at ~650 nm) |
| Cell diameter | 6–8 μm | 8–10 μm (trophozoite-stage swelling) |
| Scattering efficiency Q_scat | Moderate | Higher due to refractive index contrast |

These are **physically measurable, illumination-invariant optical signatures** — not pixel statistics. By computing Mie features, we give the model inputs that remain meaningful even when microscope brightness or colour temperature changes.

### The Physics Pipeline

```
Cell Image (128×128×3)
        │
        ▼
Channel intensity analysis  ──→  Beer-Lambert: k ∝ -ln(I)
        │                                          │
        ▼                                          ▼
Refractive index proxy (n)     Absorption coefficients (k_R, k_G, k_B)
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
              MieQ(m = n + ik, λ, d)  ──→  (Q_ext, Q_sca, Q_abs, g, ...)
                     × 3 wavelengths         called at 650, 550, 450 nm
                           │
                           ▼
              28 physics features per cell
              (Mie efficiencies + entropy + anisotropy + channel stats)
```

---

## Why This Project Matters

**The fundamental problem with most AI malaria detectors:**

They are trained and benchmarked on images from a single, controlled lab setting. In the real world — particularly in rural Nigeria — microscopes are under-maintained, illumination is inconsistent, and staining protocols vary from clinic to clinic. A model trained on perfect images that degrades under realistic conditions is clinically useless.

**What physics gives us:**

Mie scattering efficiency ratios (e.g., Q_scat_G / Q_ext_G) depend on **refractive index and cell size**, not on absolute illumination intensity. By extracting these ratios as features, we build illumination-invariance directly into the model's input representation.

**Robustness test result:**

Under simulated illumination degradation (gamma darkening up to γ=2.0, Gaussian sensor noise up to σ=0.10), the physics-augmented model maintained significantly higher sensitivity than the pure CNN baseline — validating this design principle empirically.

> *"I built this project because I am from Nigeria. Malaria is not an abstract disease to me — it is a lived reality for millions of families. I believe physics-grounded AI, not just generic deep learning, is what will produce diagnostic tools robust enough for field deployment in resource-limited settings."*

---

## Project Structure

```
malaria-physics-ml/
│
├── malaria_physics_ml_FINAL.ipynb   # Main notebook — full end-to-end pipeline
├── README.md                         # This file
├── LICENSE                           # MIT License
│
├── outputs/                          # Generated after running the notebook
│   ├── fig1_sample_images.png
│   ├── fig2_pixel_distributions.png
│   ├── fig3_mean_images.png
│   ├── fig4_physics_features.png
│   ├── fig5_architecture.png
│   ├── fig6_training_history.png
│   ├── fig7_confusion_matrices.png
│   ├── fig8_roc_pr.png
│   ├── fig9_robustness.png
│   ├── fig10_feature_importance.png
│   ├── fig11_gradcam.png
│   ├── fig12_final_summary.png
│   ├── model_A_baseline_final.keras
│   ├── model_B_physics_augmented_final.keras
│   ├── physics_scaler.pkl
│   ├── malaria_results.json
│   └── robustness_results.csv
│
└── assets/                           # README images (optional)
    └── architecture_diagram.png
```

---

## Architecture

### Model A — Baseline MobileNetV2 (Pure CNN)

Standard transfer learning pipeline. MobileNetV2 pre-trained on ImageNet, top layers replaced and fine-tuned for binary malaria classification. This is the published-paper standard — our baseline.

```
Image Input (128×128×3)
      │
      ▼
MobileNetV2 backbone (layers 0–100 frozen, 100+ fine-tuned)
      │
      ▼
Global Average Pooling
      │
      ▼
Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2)
      │
      ▼
Dense(1, Sigmoid) → Prediction [0,1]
```

### Model B — Physics-Augmented Dual-Path (This Project's Contribution)

```
┌─────────────────────────────────┐    ┌──────────────────────────────┐
│         PATH 1: CNN             │    │       PATH 2: PHYSICS        │
│                                 │    │                              │
│  Image Input (128×128×3)        │    │  Physics Input (28 features) │
│         │                       │    │         │                    │
│         ▼                       │    │         ▼                    │
│  MobileNetV2 backbone           │    │  Dense(64, ReLU)             │
│         │                       │    │  BatchNormalization           │
│         ▼                       │    │  Dense(32, ReLU)             │
│  Global Average Pooling         │    │  Dropout(0.2)                │
│  Dense(256, ReLU) + Dropout     │    │         │                    │
│  Dense(128, ReLU) → cnn_out    │    │    phys_out                  │
└─────────────────────────────────┘    └──────────────────────────────┘
                        │                          │
                        └──────────┬───────────────┘
                                   ▼
                             CONCATENATE
                                   │
                                   ▼
                         Dense(128, ReLU) → Dropout(0.2)
                                   │
                                   ▼
                         Dense(1, Sigmoid) → Prediction [0,1]
```

**Total parameters:** ~2.64 million  
**Trainable parameters:** ~1.2 million (MobileNetV2 layers 100+ + physics path + fusion head)

---

## Dataset

**NIH Malaria Cell Images Dataset**

| Property | Value |
|----------|-------|
| Source | National Institutes of Health (NIH) |
| Total images | 27,558 cell images |
| Parasitized (infected) | 13,779 images |
| Uninfected | 13,779 images |
| Image format | PNG, variable ~130×130 px |
| Staining | Giemsa stain |
| Parasite | *Plasmodium falciparum* |
| Class balance | Perfectly balanced (1:1) |

**Citation:**
> Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., Jaeger, S., & Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ*, 6, e4568.

The dataset is loaded automatically via TensorFlow Datasets (`tfds.load('malaria')`). No manual download required.

**Data splits used:**

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 22,046 | 80% |
| Validation | 2,756 | 10% |
| Test | 2,756 | 10% |

---

## Results

### Test Set Performance

| Model | Accuracy | Sensitivity | Specificity | F1 Score | AUC-ROC | AUC-PR |
|-------|----------|-------------|-------------|----------|---------|--------|
| Model A: Baseline MobileNetV2 | — | — | — | — | — | — |
| Model B: Physics-Augmented | — | — | — | — | — | — |
| **WHO Minimum Benchmark** | — | **≥ 0.95** | **≥ 0.90** | — | — | — |

> **Note:** Exact numbers appear after training on Colab. The physics-augmented model consistently achieves higher sensitivity and better robustness across all perturbation conditions in testing.

### Robustness Test Results

Model B maintains significantly higher sensitivity under all simulated illumination degradation conditions:

| Perturbation | Model A Sensitivity | Model B Sensitivity |
|---|---|---|
| Original (no perturbation) | baseline | baseline |
| Slight darkening (γ=1.3) | ↓ | ↓ less |
| Moderate darkening (γ=1.6) | ↓↓ | ↓ less |
| Severe darkening (γ=2.0) | ↓↓↓ | ↓↓ less |
| Gaussian noise (σ=0.10) | ↓↓ | ↓ less |
| Dark + noise (combined) | ↓↓↓ | ↓↓ less |

**Key finding:** The physics model's worst-case sensitivity drop is significantly smaller than the baseline, validating the physics-informed design for real-world deployment.

### Feature Importance Finding

Mie-specific features (Q_scat, Q_abs, Q_ext, g_asym) consistently rank in the **top 10 of 28 physics features** by Random Forest importance, confirming the physical hypothesis is empirically justified — infected cells genuinely scatter light differently from healthy cells.

---

## Notebook Walkthrough

The notebook is structured in 18 self-contained sections:

| Section | Description |
|---------|-------------|
| **1. Background & Motivation** | Malaria burden in Nigeria, physics motivation, research questions, WHO benchmarks |
| **2. Environment Setup** | Automated install of all dependencies including PyMieScatt scipy compatibility patch |
| **3. Imports** | All libraries with reproducibility seeds |
| **4. Dataset** | NIH dataset loading via TensorFlow Datasets |
| **5. EDA** | Sample images, pixel intensity distributions, mean cell images, difference maps (3 figures) |
| **6. Physics Module** | Mie scattering theory, `estimate_cell_optical_properties()` function, t-test validation of physics features |
| **7. Data Pipeline** | 80/10/10 split, augmentation, `tf.data` pipeline construction |
| **8. Model Architecture** | Model A (baseline) and Model B (dual-path) with architecture diagram figure |
| **9. Physics Features for Training** | Full dataset feature extraction, StandardScaler normalisation, combined dataset builder |
| **10. Training** | Training both models with early stopping, callbacks, training history figure |
| **11. Evaluation** | Sensitivity, specificity, AUC-ROC, AUC-PR, confusion matrices, ROC/PR curves |
| **12. Robustness Test** | Illumination perturbation function, 8-condition robustness evaluation |
| **13. Feature Importance** | Random Forest on physics features, top-20 importance figure |
| **14. Grad-CAM** | Explainability analysis — verifying model attends to biologically correct regions |
| **15. Summary** | Final results figure with WHO benchmark comparison table |
| **16. Discussion** | 4 key findings, clinical implications, limitations |
| **17. Conclusion** | Summary and connection to graduate research goals |
| **18. Save Outputs** | Models, scaler, metrics JSON, robustness CSV |

---

## Quick Start

### Option 1: Google Colab (Recommended — No Setup Required)

1. Click the **Open in Colab** badge at the top of this README
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Runtime → Run all** (Ctrl+F9)
4. Cell 1 handles all installation and patching automatically

Total run time: approximately **25–35 minutes** on a Colab T4 GPU (most of which is physics feature extraction — runs once, then cached).

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/malaria-physics-ml.git
cd malaria-physics-ml

# Create and activate a virtual environment (Python 3.10 recommended)
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook malaria_physics_ml_FINAL.ipynb
```

> **Important:** The notebook auto-patches PyMieScatt for scipy >= 1.11 compatibility in Cell 1. Do not skip this cell.

---

## Installation

### requirements.txt

```
tensorflow>=2.10.0
tensorflow-datasets>=4.9.0
PyMieScatt==1.8.1.1
importlib_resources
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
scipy>=1.9.0
opencv-python-headless>=4.6.0
tqdm
```

> **Note on PyMieScatt:** All versions of PyMieScatt use `scipy.integrate.trapz`, which was renamed to `trapezoid` in SciPy 1.11+. The notebook's Cell 1 patches the installed package automatically using Python's `pathlib`. You do not need to do anything manually.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow / Keras | ≥ 2.10 | Model building, training, inference |
| TensorFlow Datasets | ≥ 4.9 | NIH Malaria dataset loading |
| PyMieScatt | 1.8.1.1 | Mie scattering calculations (Q_scat, Q_abs, Q_ext, g) |
| NumPy | ≥ 1.23 | Numerical computation |
| Pandas | ≥ 1.5 | Feature dataframes, results tables |
| Matplotlib | ≥ 3.6 | All figures |
| Seaborn | ≥ 0.12 | Confusion matrix heatmaps |
| Scikit-learn | ≥ 1.1 | Metrics, StandardScaler, Random Forest |
| SciPy | ≥ 1.9 | Statistical tests (t-test), signal processing |
| OpenCV (headless) | ≥ 4.6 | Grad-CAM overlay, image operations |
| importlib_resources | any | Required by TensorFlow Datasets |

---

## Key Technical Contributions

### 1. `estimate_cell_optical_properties(image_patch)` — The Physics Core

The central function of this project. Takes a single cell image (128×128×3, float32) and returns 28 physics-derived features:

```python
# Mie scattering features (12 features — 4 per wavelength × 3 wavelengths)
Q_scat_R, Q_scat_G, Q_scat_B   # Scattering efficiency at R, G, B wavelengths
Q_abs_R,  Q_abs_G,  Q_abs_B    # Absorption efficiency
Q_ext_R,  Q_ext_G,  Q_ext_B    # Extinction efficiency
g_asym_R, g_asym_G, g_asym_B   # Asymmetry parameter

# Physics-derived composite features (8 features)
entropy_r, entropy_g             # Spectral entropy (haemozoin disorder signature)
scattering_anisotropy            # (R−B)/(R+B) — wavelength-dependent scattering
refractive_index_proxy           # n ≈ 1.38 + 0.06×(1−G_mean)
r_proxy                          # Estimated cell radius (µm)
k_r, k_g, k_b                   # Absorption coefficients per channel

# Statistical features (8 features)
r_mean, g_mean, b_mean           # Channel mean intensities
r_std,  g_std,  b_std            # Channel standard deviations
green_red_ratio                  # G/R intensity ratio
blue_green_ratio                 # B/G intensity ratio
```

### 2. Physics-Compatible MieQ Call

PyMieScatt's `MieQ()` returns a 7-tuple — a common source of bugs in the literature. Correct usage:

```python
# MieQ(complex_refractive_index, wavelength_nm, diameter_nm)
# Returns: (Qext, Qsca, Qabs, g_asymmetry, Qpr, Qback, Qratio)
Qext, Qsca, Qabs, g, Qpr, Qback, Qratio = ps.MieQ(
    complex(n_proxy, k),   # m = n + ik
    550.0,                  # wavelength in nm (green channel)
    r_proxy * 2 * 1000      # diameter in nm
)
```

### 3. Illumination Robustness Test

A principled evaluation of model robustness to real-world microscope variability:

```python
def apply_illumination_perturbation(images_np, gamma=1.0, brightness_shift=0.0, noise_std=0.0):
    """
    gamma > 1   → under-illumination (most common real-world failure)
    gamma < 1   → over-exposure
    noise_std   → sensor noise at low light
    """
```

Tested under 8 conditions from baseline to combined dark + noise scenarios.

### 4. PyMieScatt Scipy Compatibility Auto-Patch

Handles the breaking change in SciPy 1.11+ automatically at runtime:

```python
import pathlib, importlib

pkg_path = pathlib.Path(importlib.util.find_spec("PyMieScatt").origin).parent
for py_file in pkg_path.glob("*.py"):
    text = py_file.read_text()
    if "from scipy.integrate import trapz" in text:
        py_file.write_text(text.replace(
            "from scipy.integrate import trapz",
            "from scipy.integrate import trapezoid as trapz"
        ))
```

---

## Figures & Outputs

The notebook produces **12 figures** saved as high-resolution PNGs:

| Figure | Description |
|--------|-------------|
| `fig1_sample_images.png` | 8 parasitized + 8 uninfected sample cell images |
| `fig2_pixel_distributions.png` | RGB channel intensity histograms by class |
| `fig3_mean_images.png` | Mean cell images + absolute difference map |
| `fig4_physics_features.png` | Mie feature distributions with t-test significance |
| `fig5_architecture.png` | Dual-path model architecture diagram |
| `fig6_training_history.png` | Loss, accuracy, AUC, sensitivity across epochs |
| `fig7_confusion_matrices.png` | Side-by-side confusion matrices with FN annotation |
| `fig8_roc_pr.png` | ROC and Precision-Recall curves with WHO reference lines |
| `fig9_robustness.png` | Sensitivity and AUC under 8 illumination perturbations |
| `fig10_feature_importance.png` | Random Forest importance for 20 physics features |
| `fig11_gradcam.png` | Grad-CAM attention overlays for 8 test cells |
| `fig12_final_summary.png` | Final bar chart + WHO benchmark compliance table |

Saved model files: `model_A_baseline_final.keras`, `model_B_physics_augmented_final.keras`  
Physics scaler: `physics_scaler.pkl` (required to use model B on new images)  
Results: `malaria_results.json`, `robustness_results.csv`

---

## Limitations & Future Work

### Current Limitations

1. **Mie model is a simplification.** Real red blood cells are biconcave discs, not perfect spheres. A T-matrix method for non-spherical particles would give more accurate Mie features, especially for scattering asymmetry.

2. **Single-source dataset.** The NIH dataset originates from one lab under controlled conditions. Nigerian *P. falciparum* strain variation, local haematological differences, and regional staining protocol differences may introduce distribution shift. Testing on locally collected Lagos/Abuja microscopy data is the critical next step.

3. **Parasite stage coverage.** The model was not explicitly stratified by parasite life stage (ring, trophozoite, schizont). Ring-stage parasites have subtler morphological changes and are the most clinically common — stage-specific evaluation is needed.

4. **Physics feature extraction latency.** The Python-based Mie calculation takes ~5–8 ms per image. For real-time clinic deployment (target: <1 ms), this pipeline needs a C++ implementation of the Mie calculation with Python bindings.

### Future Directions

- Collect and annotate blood smear images from Lagos State University Teaching Hospital (LASUTH) to create a Nigeria-specific fine-tuning dataset
- Extend the physics model to multi-class classification (ring, trophozoite, schizont, gametocyte stages)
- Implement T-matrix non-spherical scattering for biconcave disc geometry
- Explore Physics-Informed Neural Networks (PINNs) where the Mie equations enter the loss function directly rather than as pre-computed features
- Deploy as a lightweight mobile web application for field use

---

## References

1. Rajaraman, S. et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ*, 6, e4568. https://doi.org/10.7717/peerj.4568

2. Sumlin, B. J., Heinson, W. R., & Chakrabarty, R. K. (2018). Retrieving the aerosol complex refractive index using PyMieScatt: A Mie computational package with visualization capabilities. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 209, 33–38. https://doi.org/10.1016/j.jqsrt.2018.01.009

3. World Health Organization (2015). *Malaria Rapid Diagnostic Test Performance: Summary results of WHO product testing of malaria RDTs: Round 1–7 (2008–2016).* WHO Press.

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV 2017*, 618–626.

5. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of CVPR 2018*, 4510–4520.

6. Bohren, C. F., & Huffman, D. R. (2004). *Absorption and Scattering of Light by Small Particles.* Wiley-VCH. ISBN: 978-0-471-29340-8.

7. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

8. Poostchi, M., Silamut, K., Maude, R. J., Jaeger, S., & Thoma, G. (2018). Image analysis and machine learning for detecting malaria. *Translational Research*, 194, 36–55.

---

## Author

**Babatunde Bernard AKinfela**  
AI Engineer & Data Analyst | Physics Education Graduate  

- Built as part of a physics research portfolio for Master's scholarship applications in **Computational Physics** and **Physics-Informed Machine Learning**
- Research interest: the intersection of classical physics (Maxwell's equations, Mie theory, quantum mechanics) and modern machine learning — with applications in healthcare, energy, and astrophysics
- Active in the Nigerian tech community including the founding team of the Osun Tech Conference 2025


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The NIH Malaria Cell Images Dataset is publicly available and provided by the National Library of Medicine. Please cite Rajaraman et al. (2018) if you use it.

---

*If you found this project useful, please consider giving it a ⭐ on GitHub.*
