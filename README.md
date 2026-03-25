# DataFusion

> **Multi-Fidelity Data Fusion Algorithms for Aerodynamic Coefficient Prediction**

A Python research framework implementing and benchmarking four multi-fidelity data fusion algorithms. The core idea is to combine abundant low-fidelity (LF) simulation data with scarce high-fidelity (HF) experimental/CFD data to achieve accurate aerodynamic coefficient predictions at low cost.

---

## Table of Contents

- [Background](#background)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmark Results](#benchmark-results)
- [Evaluation Metrics](#evaluation-metrics)
- [API Reference](#api-reference)

---

## Background

In aerospace engineering, high-fidelity CFD simulations and wind tunnel experiments provide accurate aerodynamic data but are extremely expensive. Low-fidelity methods (e.g., engineering estimation codes) are fast and cheap but less accurate. Multi-fidelity data fusion bridges this gap: use LF data to build a global surrogate, then correct it with sparse HF observations.

This project implements four such fusion strategies and evaluates them systematically on four aerodynamic datasets (LTV missile, AGARD-B, HB2, HSCM3 configurations).

---

## Algorithms

All four algorithms are implemented in `DataFusionAlgorithms.py` and share the same interface:

```python
y_pred = algorithm(X_LF, Y_LF, X_HF, Y_HF, X_pred)
```

### 1. MF-IDW — Multi-Fidelity Inverse Distance Weighting

A lightweight, purely geometric approach with no scale transformation.

**Steps:**
1. Fit a global LF surface via IDW interpolation
2. Evaluate the LF model at HF training locations
3. Compute residuals: `r = Y_HF − Ŷ_LF(X_HF)`
4. Interpolate residuals to prediction points via IDW
5. Fuse: `Ŷ_MF = Ŷ_LF + r̂` (additive correction)

**Key parameters:** `p_LF=2` (LF IDW power), `p_R=2` (residual IDW power)

---

### 2. ConvexHull-GP — Convex Hull Distance–Guided Gaussian Process

An adaptive strategy that applies different prediction rules inside vs. outside the convex hull of HF training data.

**Steps:**
1. Normalize inputs to [−1, 1] using MinMaxScaler (fit on LF data)
2. For each test point, compute its distance to the convex hull of HF points via SLSQP optimization
3. Fit GP (RBF kernel) on LF data; fit GP on HF data
4. **Inside hull** (dist ≤ 0.01): predict directly with HF GP
5. **Outside hull** (dist > 0.01): `Ŷ = GP_LF(x) + [GP_HF(x_nearest) − GP_LF(x_nearest)]`

---

### 3. GPy-CoKriging — Co-Kriging (Kennedy & O'Hagan Framework)

A probabilistic Bayesian approach with full uncertainty quantification.

**Steps:**
1. Fit GP model on LF data
2. Predict LF values at HF locations: `Ŷ_l(X_h)`
3. Estimate scale factor ρ via OLS regression: `Y_h ≈ ρ · Ŷ_l`
4. Compute residuals: `δ = Y_h − ρ · Ŷ_l(X_h)`
5. Fit a correction GP on residuals
6. Predict: `Ŷ_h = ρ · GP_LF(x) + GP_δ(x)`, variance: `σ²_h = ρ² σ²_l + σ²_δ`

*Can optionally return predictive variance for uncertainty-aware applications.*

---

### 4. GPy-InvDistMean — GP + Inverse-Distance-Weighted Residual Correction

Combines GP-based LF modeling with IDW residual blending.

**Steps:**
1. Normalize inputs to [−1, 1]
2. Fit GP on LF data
3. Evaluate LF GP at HF locations and prediction points
4. Compute residuals at HF locations
5. For each test point, compute IDW weights to all HF training points
6. Fuse: `Ŷ_new = Ŷ_LF(x) + Σ w_i · r_i`

**Note:** IDW power is fixed at 1 (i.e., `w = 1/d`, not `1/d²`).

---

### Algorithm Comparison

| Dimension | MF-IDW | ConvexHull-GP | GPy-CoKriging | GPy-InvDistMean |
|-----------|--------|---------------|---------------|-----------------|
| LF model | IDW | GP (RBF) | GP (RBF) | GP (RBF) |
| Residual method | IDW interpolation | GP point correction | GP on residuals | IDW weighting |
| Scale factor ρ | None | None | OLS estimate | None |
| Input normalization | None | MinMaxScaler [−1,1] | None | MinMaxScaler [−1,1] |
| Uncertainty output | No | No | Yes (optional) | No |
| Domain awareness | No | Yes (convex hull) | No | No |
| Computational cost | Low | High (SLSQP per point) | Medium | Medium |

---

## Project Structure

```
DataFusion/
│
├── DataFusionAlgorithms.py          # Core algorithm library (4 algorithms)
│
├── benchmark_fusion.py              # LTV CN — 10% HF training
├── benchmark_fusion_20pct.py        # LTV CN — 20% HF training
├── benchmark_fusion_50pct.py        # LTV CN — 50% HF training
├── benchmark_fusion_80pct.py        # LTV CN — 80% HF training
├── benchmark_CM.py                  # LTV CM — 10%/20%/50%/80% HF training
├── benchmark_AGARD_B_CL.py          # AGARD-B CL — 20%/50%/80% HF training
├── benchmark_HB2_CN.py              # HB2 CN — 20%/50%/80% HF training
├── benchmark_HSCM3_CN.py            # HSCM3 CN — 30%/60% HF training
├── benchmark_HSCM3_CN_50pct.py      # HSCM3 CN — 50% HF training
├── benchmark_LTV_CN_region.py       # LTV CN — region-based split
├── benchmark_LTV_CN_region_mid.py   # LTV CN — mid-range region split
├── benchmark_LTV_CN_region_narrow.py# LTV CN — narrow-range region split
│
├── LTV_Low-Fidelity.xlsx            # LTV missile LF data (1000 samples)
├── LTV_High-Fidelity.xlsx           # LTV missile HF data
├── AGARD-B_CL_Low-Fidelity.xlsx     # AGARD-B LF data (23 samples)
├── AGARD-B_CL_High-Fidelity.xlsx    # AGARD-B HF data
├── HB2_CN_Low-Fidelity.xlsx         # HB2 LF data (39 samples)
├── HB2_CN_High-Fidelity.xlsx        # HB2 HF data
├── HSCM3_CN_Low-Fidelity.xlsx       # HSCM3 LF data (28 samples)
├── HSCM3_CN_High-Fidelity.xlsx      # HSCM3 HF data
│
├── check_env.py                     # Environment check utility
├── inspect_data.py                  # LTV data inspection utility
├── _inspect_agard.py                # AGARD-B data inspection
├── _inspect_hb2.py                  # HB2 data inspection
├── _inspect_hscm3.py                # HSCM3 data inspection
│
├── requirements.txt                 # Python dependencies
│
└── [output *.png / *.xlsx]          # Generated benchmark figures and results
```

---

## Datasets

| Dataset | Configuration | Features | Target | LF Samples | Feature Dim |
|---------|---------------|----------|--------|------------|-------------|
| LTV | Missile body | Mach, Alpha | CN, CM | 1000 | 2D |
| AGARD-B | AGARD standard model | Ma, sinA | CL | 23 | 2D |
| HB2 | HB2 standard model | H, Ma, sinA | CN | 39 | 3D |
| HSCM3 | HSCM3 vehicle | H, Ma, sinA | CN | 28 | 3D |

**Feature glossary:**
- `Mach` / `Ma` — Mach number
- `Alpha` — angle of attack (degrees)
- `sinA` — sin(angle of attack)
- `H` — flight altitude
- `CN` — normal force coefficient
- `CL` — lift coefficient
- `CM` — pitching moment coefficient

---

## Installation

### Prerequisites

- Python 3.9+
- A virtual environment (recommended)

### Setup

```bash
# Clone or navigate to the DataFusion directory
cd DataFusion

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python check_env.py
```

Expected output:
```
numpy       : 1.26.4
pandas      : 3.0.1
scipy       : 1.12.0
matplotlib  : 3.10.8
GPy         : 1.13.2
scikit-learn: 1.8.0
openpyxl    : 3.1.5

ALL IMPORTS OK - DataFusion venv is ready!
```

---

## Quick Start

### Run a single benchmark

```bash
# LTV CN, 10% HF training ratio
python benchmark_fusion.py

# LTV CN, 50% HF training ratio
python benchmark_fusion_50pct.py

# AGARD-B CL, multiple ratios (20% / 50% / 80%)
python benchmark_AGARD_B_CL.py
```

Each script produces:
- **PNG figures**: prediction scatter plots, residual plots, and metric bar charts
- **Excel file**: per-point predictions and summary performance table

### Use algorithms in your own code

```python
import numpy as np
from DataFusionAlgorithms import (
    mf_idw_interpolate,
    mf_ConvexHull,
    mf_GPy_CoKriging,
    mf_GPy_inverseDistanceMean,
)

# Example: 2D inputs, column-vector outputs
X_lf = np.random.rand(100, 2)       # 100 LF samples, 2 features
Y_lf = np.random.rand(100, 1)

X_hf = np.random.rand(20, 2)        # 20 HF samples
Y_hf = np.random.rand(20, 1)

X_pred = np.random.rand(50, 2)      # 50 test points

# --- MF-IDW (Y inputs must be 1D) ---
y_pred_idw = mf_idw_interpolate(X_lf, Y_lf.ravel(), X_hf, Y_hf.ravel(), X_pred)

# --- ConvexHull-GP ---
y_pred_ch = mf_ConvexHull(X_lf, Y_lf, X_hf, Y_hf, X_pred)

# --- GPy-CoKriging ---
y_pred_cok = mf_GPy_CoKriging(X_lf, Y_lf, X_hf, Y_hf, X_pred)

# --- GPy-InvDistMean ---
y_pred_idm = mf_GPy_inverseDistanceMean(X_lf, Y_lf, X_hf, Y_hf, X_pred)
```

> **Important:** MF-IDW requires 1D (raveled) Y arrays. All other algorithms require column vectors `(n, 1)`.

---

## Benchmark Results

Each benchmark script outputs a three-row visualization panel:

| Row | Content |
|-----|---------|
| Row 1 | Predicted vs. true value scatter plots (one per algorithm) with R² annotation |
| Row 2 | Residual plots showing prediction errors vs. true values, with ±RMSE bands |
| Row 3 | Bar charts comparing RMSE, MAE, R², and Mean Relative Error across algorithms |

Results are also saved to Excel with two sheets:
- **Performance** — summary metrics table
- **Predictions** — per-point predictions and residuals

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `RMSE` | Root Mean Squared Error |
| `MAE` | Mean Absolute Error |
| `R²` | Coefficient of determination (1 = perfect fit) |
| `MaxError` | Maximum absolute prediction error |
| `MeanRelErr(%)` | Mean relative error (%), computed only where `|y_true| > threshold` to avoid division near zero |
| `Time(s)` | Wall-clock runtime in seconds |

Relative error thresholds by dataset:
- CN (LTV): `|y_true| > 0.05`
- CL (AGARD-B): `|y_true| > 0.005`
- CM (LTV): `|y_true| > 0.5`

---

## API Reference

### `idw_interpolate(X_known, y_known, X_pred, p=2, epsilon=1e-8)`

Standard Inverse Distance Weighting interpolation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_known` | `ndarray (N, D)` | Known point coordinates |
| `y_known` | `ndarray (N,)` | Known values (1D) |
| `X_pred` | `ndarray (M, D)` | Prediction point coordinates |
| `p` | `float` | Power parameter (default: 2) |
| `epsilon` | `float` | Zero-distance guard (default: 1e-8) |

**Returns:** `ndarray (M,)`

---

### `mf_idw_interpolate(X_LF, y_LF, X_HF, y_HF, X_pred, p_LF=2, p_R=2)`

Multi-fidelity IDW fusion.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_LF, y_LF` | `ndarray` | LF coordinates and values **(1D ravel)** |
| `X_HF, y_HF` | `ndarray` | HF coordinates and values **(1D ravel)** |
| `X_pred` | `ndarray` | Prediction coordinates |
| `p_LF` | `float` | LF IDW power (default: 2) |
| `p_R` | `float` | Residual IDW power (default: 2) |

**Returns:** `ndarray (M,)`

---

### `mf_ConvexHull(X_l, Y_l, X_h, Y_h, X_new)`

Convex hull distance–guided GP fusion.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_l, Y_l` | `ndarray (n_l, D)`, `(n_l, 1)` | LF data |
| `X_h, Y_h` | `ndarray (n_h, D)`, `(n_h, 1)` | HF data |
| `X_new` | `ndarray (n_new, D)` | Test points |

**Returns:** `ndarray (n_new,)`

---

### `mf_GPy_CoKriging(X_l, Y_l, X_h, Y_h, X_new)`

Co-Kriging multi-fidelity fusion with optional variance output.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_l, Y_l` | `ndarray (n_l, D)`, `(n_l, 1)` | LF data (no normalization applied) |
| `X_h, Y_h` | `ndarray (n_h, D)`, `(n_h, 1)` | HF data |
| `X_new` | `ndarray (n_new, D)` | Test points |

**Returns:** `ndarray (n_new,)` *(modify source to also return variance)*

---

### `mf_GPy_inverseDistanceMean(X_l, Y_l, X_h, Y_h, X_new)`

GP + inverse-distance-weighted residual fusion.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_l, Y_l` | `ndarray (n_l, D)`, `(n_l, 1)` | LF data |
| `X_h, Y_h` | `ndarray (n_h, D)`, `(n_h, 1)` | HF data |
| `X_new` | `ndarray (n_new, D)` | Test points |

**Returns:** `ndarray (n_new,)`

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `numpy` | 1.26.4 | Numerical computation |
| `scipy` | 1.12.0 | Distance calculation, SLSQP optimization |
| `GPy` | 1.13.2 | Gaussian Process Regression |
| `scikit-learn` | 1.8.0 | Data splitting, metrics, normalization |
| `pandas` | 3.0.1 | Data loading (Excel) |
| `matplotlib` | 3.10.8 | Visualization |
| `openpyxl` | 3.1.5 | Excel I/O |

---

## Notes

- All benchmark scripts fix `np.random.seed(42)` for reproducibility.
- GP hyperparameters (RBF `variance` and `lengthscale`) are initialized to 1.0 and optimized via Maximum Likelihood Estimation (`model.optimize()`). No multi-start or cross-validation is performed.
- ConvexHull-GP can be slow for large test sets due to per-point SLSQP optimization.
