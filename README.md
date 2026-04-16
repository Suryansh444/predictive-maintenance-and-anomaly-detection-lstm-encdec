# Predictive Maintenance & Anomaly Detection in Industrial IoT
### LSTM Encoder-Decoder (EncDec-AD) on NASA C-MAPSS FD001 Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS%20FD001-lightgrey)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

> A semi-supervised anomaly detection system for turbofan engine degradation using LSTM Encoder-Decoder architectures, extending Malhotra et al. (2016) with Bahdanau attention — achieving **F1 = 0.9536** on held-out test data.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [References](#references)
- [Authors](#authors)

---

## Overview

This project implements and extends the **EncDec-AD** framework (Malhotra et al., 2016) for anomaly detection on the NASA C-MAPSS FD001 turbofan engine degradation dataset. Two models are developed:

| Model | Description |
|-------|-------------|
| **Baseline LSTM EncDec-AD** | Reproduces the original paper architecture — LSTM encoder compresses the sequence into a fixed latent vector, LSTM decoder reconstructs it in reverse |
| **Attention LSTM EncDec-AD** | Augments the baseline with **Bahdanau attention**, eliminating the fixed-length information bottleneck by letting the decoder selectively attend to all encoder states |

Both models are trained **semi-supervised** — exclusively on normal (healthy) engine data — and scored at inference time using **Mahalanobis distance** on reconstruction errors modelled as a multivariate Gaussian.

---

## Results

### Test Set Performance

| Model | Precision | Recall | F1-Score | TPR/FPR |
|-------|-----------|--------|----------|---------|
| Baseline EncDec-AD | 0.8840 | 0.8972 | 0.8905 | 4.84 |
| **Attention EncDec-AD** | **0.9377** | **0.9700** | **0.9536** | **9.57** |
| Improvement | +5.4% | +7.3% | **+6.3%** | +97.7% |

### Comparison with Reference Paper (Malhotra et al., 2016)

| Dataset | Model | Fβ-Score |
|---------|-------|----------|
| Power Demand | EncDec-AD | 0.77 |
| Space Shuttle | EncDec-AD | 0.81 |
| ECG | EncDec-AD | 0.65 |
| **FD001 (Ours)** | **Attention** | **0.9536 ** |

The attention variant **surpasses all results from the reference paper**.

### Anomaly Score Separation

| Metric | Baseline | Attention |
|--------|----------|-----------|
| Mean score (Normal) | 27.65 | 58.92 |
| Mean score (Anomalous) | 609.91 | 4034.08 |
| **Score Ratio** | **22.1×** | **68.5×** |

---

## Project Structure

```
.
├── final_fd001.ipynb           # Main notebook — full pipeline end-to-end
├── DAC_204_Project_Report.pdf  # Detailed project report
├── cmapss_data/                # Dataset directory (auto-created by notebook)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── saved_models/               # Saved model weights (auto-created)
│   ├── baseline_best.keras
│   └── attention_best.keras
├── requirements.txt
└── README.md
```

---

## Dataset

**NASA C-MAPSS FD001** — Commercial Modular Aero-Propulsion System Simulation

| Property | Value |
|----------|-------|
| Training Engines | 100 |
| Test Engines | 100 |
| Operating Condition | Single (Sea Level) |
| Fault Mode | Single (HPC Degradation) |
| Sensors | 21 raw sensors + 3 operational settings |
| Training Cycles | 20,631 rows |

### Download

The dataset is publicly available from NASA's Prognostics Data Repository:

🔗 [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Download **CMAPSS Data** and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in the project root directory (the notebook will move them automatically).

### Labeling Strategy

| Label | RUL Range | Description |
|-------|-----------|-------------|
| Normal | RUL > 100 | Engine far from failure |
| Transition (excluded) | 30 ≤ RUL ≤ 100 | Ambiguous region — excluded from both training and evaluation |
| Anomalous | RUL < 30 | Engine in degraded operating regime |

---

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- GPU recommended (NVIDIA CUDA-compatible) for training

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/predictive-maintenance-lstm-encdec.git
cd predictive-maintenance-lstm-encdec
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the C-MAPSS dataset from the NASA link above and place the following files in the project root:

```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

---

## Usage

### Run the full pipeline

Open and run `final_fd001.ipynb` end-to-end in Jupyter:

```bash
jupyter notebook final_fd001.ipynb
```

The notebook is organized into clearly labeled sections:

| Section | Description |
|---------|-------------|
| 1. Introduction | Background, motivation, EncDec-AD summary |
| 2. Dataset Description | C-MAPSS FD001 properties and degradation behavior |
| 3. Data Loading & Preprocessing | Sensor selection, normalization, sliding window |
| 4. Data Splits | Paper-correct train/val/test splits |
| 5. Model Architecture | Baseline and Attention model definitions |
| 6. Training | Semi-supervised training on normal data only |
| 7. Anomaly Detection | Gaussian error modeling, Mahalanobis scoring |
| 8. Threshold Optimization | F1-maximizing threshold selection |
| 9. Evaluation | Confusion matrices, PR curves, metrics |
| 10. Comparative Analysis | Baseline vs. Attention, vs. reference paper |

---

## Model Architecture

### Baseline LSTM Encoder-Decoder

```
Input (batch, 30, 11)
    └─► Encoder LSTM (64 units) → final hidden state h_L ∈ ℝ⁶⁴
            └─► RepeatVector (30×)
                    └─► Decoder LSTM (64 units)
                            └─► TimeDistributed Dense (11 units)
                                    └─► Reconstructed sequence (batch, 30, 11)

Total Parameters: 53,195
```

### Attention LSTM Encoder-Decoder

```
Input (batch, 30, 11)
    └─► Encoder LSTM (64 units) → all hidden states H ∈ ℝ³⁰ˣ⁶⁴
            └─► Bahdanau Attention (32 units)
                    ↓ dynamic context vector c_t at each decoder step
            └─► Decoder LSTM (64 units) with [h_{t-1}; c_t] input
                    └─► TimeDistributed Dense (11 units)
                            └─► Reconstructed sequence (batch, 30, 11)

Total Parameters: 58,027  (+9% over baseline)
```

**Bahdanau Attention:**

```
e_{t,s}  = vᵀ · tanh(W_a · h^D_{t-1} + U_a · h^E_s)
α_{t,s}  = softmax(e_{t,s})
c_t      = Σ_s α_{t,s} · h^E_s
```

---

## Methodology

### Training (Semi-supervised)

- Models are trained **only on normal sequences** (RUL > 100)
- Loss function: Mean Squared Error (MSE) reconstruction loss
- Optimizer: Adam (lr = 5×10⁻⁴)
- Early stopping with patience = 15 epochs on a held-out normal validation set

### Anomaly Scoring

1. **Reconstruction Error**: Per-timestep absolute difference `|x - x̂| ∈ ℝ¹¹`
2. **Gaussian Fitting**: Fit multivariate Gaussian `N(μ, Σ)` to errors from a normal validation split
3. **Mahalanobis Distance**: Anomaly score per timestep: `a = (e - μ)ᵀ Σ⁻¹ (e - μ)`
4. **Sequence Score**: Maximum Mahalanobis distance across all 30 timesteps
5. **Threshold τ**: Selected to maximize F1 on a mixed validation set (normal + anomalous)

### Sensor Selection

Of 21 raw sensors, 10 were removed due to near-zero variance (uninformative). **11 sensors** were retained: sensors 2, 3, 4, 7, 9, 11, 12, 14, 17, 20, 21.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window Size (L) | 30 cycles |
| Latent Dimension | 64 |
| Attention Units | 32 |
| Batch Size | 64 |
| Max Epochs | 80 |
| Learning Rate | 5×10⁻⁴ |
| Early Stopping Patience | 15 |
| Normal RUL Threshold | > 100 |
| Anomaly RUL Threshold | < 30 |

---

## Key Findings

1. **Semi-supervised learning is effective** — Training on normal data alone yields strong anomaly discrimination without any labelled anomalous examples.
2. **Sensor selection matters** — Removing 10 non-informative sensors reduces dimensionality from 21→11 without degrading performance.
3. **Attention significantly improves performance** — Bahdanau attention increased F1 by +6.3%, reduced false positives by 45%, and added only 4,832 parameters (+9%).
4. **Mahalanobis scoring provides strong separation** — 22× (baseline) and 68× (attention) anomaly-to-normal score ratios.
5. **Attention converges faster** — 21 epochs vs. 80 epochs for the baseline, indicating richer representational capacity.
6. **Single operating condition facilitates detection** — FD001's clean single-condition setup enables higher F1 than the multi-condition FD002–FD004 variants.

---

## References

1. Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.* ICML 2016 Anomaly Detection Workshop.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
3. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
4. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *ICSPHM 2008*.

---

## Authors

| Name | Roll No. |
|------|----------|
| Suryansh Rathore | 24125037 |
| Sahil Kaler | 24125034 |
| Diptanshu Pati | 24125012 |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with TensorFlow 2.10.1 · Python 3 · GPU-accelerated training*
