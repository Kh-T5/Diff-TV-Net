## Diff-TV-Net

A PyTorch implementation of **Differentiable Total Variation Denoising** using `cvxpylayers`.

This project integrates convex optimization solvers directly into deep learning architectures. By utilizing a **CNN backbone** as a weight predictor, the model learns to generate spatially-adaptive regularization maps **($\Lambda$)** that are passed into a differentiable TV-denoising layer. This end-to-end pipeline is trained via implicit differentiation of the KKT conditions implemented in `cvxpylayers`.

---

## Repository Structure

```text
.
├── data/
│   └── sample/           # Sample images for qualitative assessment
├── notebooks/
│   └── convex_optimization_layer.ipynb  # Primary benchmarking and analysis
├── results/
│   ├── models/           # Serialized (.pth) weights for the 4 variants
│   └── plots/            # Quantitative training curves
├── src/
│   ├── data/             # Dataset utilities and noise injection logic
│   ├── models/           # Hybrid CNN-CvxpyLayer architectures
│   └── utils/            # Trainer, evaluation metrics (SSIM, MSE) and solvers
│   └── config.py         # Global hyperparameters and solver configurations         
│── scripts/
│   └── train.py/         # Main training execution script
├── env.yaml              # Conda environment specification

```

---

## Experiments & Benchmarking

The project evaluates the trade-offs between regularization geometry (Isotropic vs. Anisotropic) and solver selection (SCS vs. CLARABEL), it also performs analysis between the standard solver and its counterpart with a trained CNN-backbone.

### Comparative Analysis

A comprehensive performance study is available in the **[Benchmarking Notebook](notebook/convex_optimization_layer.ipynb)**. Key metrics include:

| Model Variant | Solver | Forward Pass (avg) | Backward Pass (avg) |
| --- | --- | --- | --- |
| **Anisotropic-SCS** | SCS | ~0.08s | ~0.5s |
| **Anisotropic-CLARABEL** | Clarabel | ~0.10s | ~0.8s |
| **Isotropic-SCS** | SCS | ~0.13s | ~17.2s |
| **Isotropic-CLARABEL** | Clarabel | ~0.17s | ~20.4s |

### Core Findings

* **Computational Complexity:** Isotropic regularization via Second-Order Cone Programming (SOCP) exhibits significantly higher backward-pass latency compared to Anisotropic Quadratic Programming (QP).
* **Structural Preservation:** The CNN successfully learns to predict lower  values at edges, effectively mitigating the "staircasing" effect inherent in standard TV-denoising by allowing for sharper discontinuities.

---

## Setup and Training

### 1. Environment Configuration

Create the environment using the provided YAML file:

```bash
conda env create -f env.yaml
conda activate cv-opti-nn

```
For training purposes, user should download DIV2K datasets (train, val, test) in data/DIV2K/
### 2. Hardware Acceleration

The pipeline is optimized for **Apple Silicon (MPS)**:

* **CNN Backbone:** Executes on GPU/MPS for high-throughput feature extraction.
* **Optimization Layer:** Executes on CPU due to specific linear algebra requirements in `cvxpylayers`.

### 3. Training Execution

To train a model variant, configure the `reg_type` and `solver` parameters in `config.py` and execute:

```bash
python -m scripts.train--reg [isotropic|anisotropic] --solver [SCS|CLARABEL] --epochs 50 --alpha 0.8 --lr 1e-4 --batch_size 16

```

The hybrid model is trained using a weighted objective function that balances pixel-wise intensity accuracy with structural preservation:

$$\mathcal{L}(y, \hat{y}) = \alpha \cdot \text{MSE}(y, \hat{y}) + (1 - \alpha) \cdot (1 - \text{SSIM}(y, \hat{y}))$$

Where:
* $y$ is the ground truth image.
* $\hat{y}$ is the denoised output from the `CvxpyLayer`.
* $\alpha$ is a hyperparameter set to 0.8 in this case weighting the importance of the Mean Squared Error.
* $\text{SSIM}$ is the Structural Similarity Index, providing a perceptual measure of degradation.

---
