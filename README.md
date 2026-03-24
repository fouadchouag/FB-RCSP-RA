# FB-RCSP-RA: Filter-Bank Regularized CSP with Per-Band Riemannian Alignment

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange.svg)]()

> **Official implementation of:**
> *"FB-RCSP-RA: A Filter-Bank Regularized Common Spatial Pattern Framework with
> Per-Band Riemannian Alignment for Cross-Subject Motor Imagery EEG Decoding"*
> [Authors] — [Journal Name], Under Review, 2026.

---

## Overview

FB-RCSP-RA is a novel cross-subject MI-EEG classification framework that simultaneously
addresses three sources of inter-subject performance degradation:

1. **Filter-Bank decomposition** — 9 overlapping sub-bands (8–30 Hz) to capture subject-specific
   mu and beta oscillatory patterns
2. **Per-band Ledoit-Wolf regularization** — analytical shrinkage estimation for well-conditioned
   covariance matrices under limited training data
3. **Per-band Riemannian alignment** — normalizes inter-subject covariance domain shift at the
   manifold level before spatial filter estimation

**Key results on BCI Competition IV Dataset 2a (cross-subject LOSO protocol):**

| Method | LOSO Accuracy | CV5 Accuracy | ITR (bits/min) |
|--------|:------------:|:------------:|:--------------:|
| CSP | 38.46% | 57.84% | 0.951 |
| ACMCSP [1] | 38.93% | 57.26% | 1.015 |
| RCSP | 38.89% | 62.69% | 1.010 |
| Riemannian MDM | 40.70% | 63.96% | 1.279 |
| **FB-RCSP-RA (Proposed)** | **42.98%** | **71.18%** | **1.658** |

> FB-RCSP-RA achieves both the highest cross-subject LOSO accuracy and the highest
> within-subject CV5 accuracy among all evaluated methods — a dual advantage.

---

## Repository Structure

```
FB-RCSP-RA/
├── README.md
├── requirements.txt
├── LICENSE
├── run_all.py                        # Run all experiments (LOSO + CV5)
│
├── data/
|   ├── bbci2a                        # Dataset bci cmpetition 2a must be here
│   ├── loader.py                     # EEG data loader with EA preprocessing
│   └── README_data.md                # Dataset download instructions
│
├── features/
│   ├── csp.py                        # Standard CSP (baseline)
│   ├── acmcsp.py                     # Adaptive Multi-Class CSP (baseline) [1]
│   ├── rcsp.py                       # Regularized CSP — Ledoit-Wolf (baseline)
│   ├── riemann.py                    # Riemannian MDM classifier (baseline)
│   └── fbrcspra.py                   # FB-RCSP-RA (proposed method)
│
├── experiments/
│   ├── run_csp_loso.py
│   ├── run_acmcsp_loso.py
│   ├── run_rcsp_loso.py
│   ├── run_mdm_loso.py
│   ├── run_fbrcspra_loso.py          # Proposed method — LOSO
│   ├── run_csp_all_subjects.py
│   ├── run_acmcsp_all_subjects.py
│   ├── run_rcsp_all.py
│   ├── run_mdm_all.py
│   └── run_fbrcspra_cv5.py           # Proposed method — CV5
│
├── results/
│   ├── *_loso.csv            # LOSO results 
│   └── *_cv5_metrics.csv     # CV5 results 
│
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fouadchouag/FB-RCSP-RA.git
cd FB-RCSP-RA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
mne>=1.6.0
pyriemann>=0.5
scikit-learn>=1.4.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
```

---

## Dataset

The BCI Competition IV Dataset 2a is **not included** in this repository.
See [`data/README_data.md`](data/README_data.md) for complete download instructions.

---

## Reproducing Results

### Run all experiments (LOSO + CV5, all methods)

```bash
python run_all.py
```

### Run only FB-RCSP-RA LOSO (proposed method)

```bash
python experiments/run_fbrcspra_loso.py
```

**Expected output:**
```
Testing on A01 ... Accuracy: 0.6875
Testing on A02 ... Accuracy: 0.2118
Testing on A03 ... Accuracy: 0.6458
...
Mean LOSO: 0.4298   Std: 0.1809
```

## Key Implementation Details

| Parameter | Value |
|-----------|-------|
| EEG channels | 22 (EOG excluded) |
| Sampling frequency | 250 Hz |
| Epoch window | tmin = 0.5 s, tmax = 4.5 s (relative to cue) |
| Filter bank | 9 bands: 8–12, 10–14, 12–16, 14–18, 16–20, 18–22, 20–24, 22–26, 24–30 Hz |
| Covariance estimator | Ledoit-Wolf (analytical shrinkage) |
| Riemannian alignment | Per-band, applied before RCSP estimation |
| RCSP components | n_components = 6 per class per band (OVR) |
| Classifier | SVM (RBF kernel, C = 4, γ = 'scale') |
| Evaluation | Cross-subject LOSO (9 folds) + Within-subject CV5 |
| Preprocessing | Euclidean Alignment (EA) per subject |
| Random seed | 42 (fully deterministic) |

---

## Processing Pipeline

```
Raw EEG (22 ch, 250 Hz)
        ↓
Euclidean Alignment (EA) — per subject, no target data needed
        ↓
Filter Bank — 9 overlapping sub-bands (4th-order Butterworth, zero-phase)
        ↓  for each sub-band b
Ledoit-Wolf Covariance Estimation — analytical shrinkage
        ↓
Per-Band Riemannian Alignment — Σ̃ = M^(-½) Σ M^(-½)
        ↓
RCSP Spatial Filtering — 6 filters/class, OVR strategy
        ↓
Log-Variance Features — f^(b) = log(var(W^T X̃))
        ↓  concatenate 9 bands
StandardScaler → SVM (RBF, C=4)
        ↓
Predicted MI class: {left hand, right hand, feet, tongue}

Online inference: < 40 ms per trial (CPU only, no GPU required)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{[AuthorLastName]2025fbrcspra,
  title     = {FB-RCSP-RA: A Filter-Bank Regularized Common Spatial Pattern
               Framework with Per-Band Riemannian Alignment for Cross-Subject
               Motor Imagery EEG Decoding},
  author    = {[Authors]},
  journal   = {[Journal Name]},
  year      = {2026},
  note      = {Under Review}
}
```

The ACMCSP baseline used in this work follows the adaptive covariance fusion
principle of:

```bibtex
@article{song2015improving,
  title   = {Improving brain-computer interface classification using adaptive
             common spatial patterns},
  author  = {Song, Xugang and Yoon, Seon-Chil},
  journal = {Computers in Biology and Medicine},
  volume  = {61},
  pages   = {150--160},
  year    = {2015},
  doi     = {10.1016/j.compbiomed.2015.03.023}
}
```

---
## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

## Contact

For questions or issues, please open a GitHub Issue or contact:
fouad.chouag@univ-setif.dz
amfouad.chouag@gmail.com
