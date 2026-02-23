# pyLocalizer
This repo trains a 2D CNN to predict **(x, y)** location from a short **time sequence** of RF measurements collected from **4 Omnidirectional Antennas**. The points grid is a 5.3x5.3 meters 2D grid where uniformly 15x15 training points are placed. The antennas are located at the corners of the 5x5 meters 2D space. The test locations are placed randomly within the points grid. The results achieve mean localization accuracy of ~5 cm on the test set. The median localization accuracy is also ~5 cm.  

Each time step contains:
- RSSI (dBm)
- phase encoded as **cos(φ)** and **sin(φ)** (to avoid phase wrap discontinuities)

Training can optionally use **Delaunay triangulation augmentation** to synthesize additional labeled samples inside the convex hull of training locations.

## Directory Structure

- `pyLocalizer/`
  - `train.py` — main training entry (creates `results/` run folders)
  - `results/` — auto-created; each run gets its own timestamped folder
  - `src/`
    - `__init__.py`
    - `io_utils.py` — load `.npz`, create run dir, JSON-safe saving
    - `utils_seed.py` — deterministic seeding helpers
    - `dataset.py` — PyTorch Dataset + normalization
    - `model.py` — `CNN2D_TimeAnchor` architecture
    - `losses.py` — Euclid + Huber combined loss
    - `augmentation.py` — Delaunay augmentation (phase-correct mixing)
    - `metrics.py` — localization error + summary stats
    - `train_eval.py` — training loop + checkpointing + plot generation
    - `plotting.py` — training curves + error CDF plots


## Dataset (`.npz`) Format

Training expects a single NumPy archive (e.g., `rssiData.npz) with these keys:

### Required keys

- `X`: `float32` array of shape **(P, T, 3, 4)**
  - `P` = number of spatial points (locations)
  - `T` = number of time snapshots per point
  - channel dimension (size 3):
    - `X[..., 0, :]` = RSSI (dBm) for 4 anchors
    - `X[..., 1, :]` = cos(φ) for 4 anchors
    - `X[..., 2, :]` = sin(φ) for 4 anchors
  - last dimension (size 4) = anchor index

- `Y`: `float32` array of shape **(P, 2)**
  - `Y[:, 0] = x`, `Y[:, 1] = y`

- `train_pts`: `int64` array of indices into `0..P-1` for training locations
- `test_pts`: `int64` array of indices into `0..P-1` for evaluation locations

### Optional keys
- `pts`: `float32 (P,2)` (same as `Y`, included for clarity)
- `meta`: a dict-like object storing dataset parameters (seed, anchors, etc.)

---

## Input to the CNN

The raw stored input for one sample is `X[i]` with shape `(T, 3, 4)`.  
The dataset loader:
1. Normalizes using **train-only mean/std per (channel, anchor)**  
2. Transposes to Conv2D format: **(C, H, W) = (3, T, 4)**

So the network sees:
- **Channels:** {RSSI, cos(φ), sin(φ)}
- **Height:** time axis `T`
- **Width:** anchor axis (4 anchors)

---

## Model: `CNN2D_TimeAnchor`

The model is a compact 2D CNN designed to learn joint time/anchor patterns:

### Architecture summary
- **Stem:** Conv → BN → GELU blocks
  - `3 → 32` (stride 1)
  - `32 → 64` (stride `(2,1)` downsample time only)
- **Mid:** deeper conv stack with progressive time downsampling
  - `64 → 128` (stride `(2,1)`)
  - `128 → 128` (stride 1)
  - `128 → 256` (stride `(2,1)`)
  - `256 → 256` (stride 1)
- **Head:** global pooling + MLP regression head
  - AdaptiveAvgPool2d → Flatten
  - Linear(256→256) + GELU + Dropout(0.15)
  - Linear(256→2) outputs `(x,y)`

Key design choices:
- **Downsample time, not anchors** (anchors remain width=4)
- **GELU + BatchNorm** for stable training
- **Global pooling** so the model can handle different T values if needed

---

## Training Strategy

### Determinism
Training is designed to be reproducible:
- fixed seeds (Python, NumPy, Torch)
- deterministic cuDNN settings
- deterministic Delaunay augmentation via `AUG_SEED`

### Optimization
- Optimizer: **AdamW**
- LR schedule: **cosine decay** each epoch (`lr_for_epoch`)
- Gradient clipping: `clip_grad_norm_ = 1.0`

### Loss: Euclid + Huber (`EuclidHuberLoss`)
We combine:
1. **Smooth L1 (Huber)** on x/y coordinate regression  
2. **Euclidean distance** error term on (x,y)

Form:
- `loss = huber(pred, target) + lam * mean_euclid(pred, target)`

Why this helps:
- Huber stabilizes coordinate regression and reduces sensitivity to outliers
- Euclidean term directly targets the evaluation metric (distance in meters)

---

## Delaunay Augmentation (Optional)

### What it does
Given training locations, we:
1. Build a **Delaunay triangulation** over training `(x,y)` points
2. For each triangle (simplex), sample barycentric weights `w ~ Dirichlet(1,1,1)`
3. Create synthetic samples inside the triangle by mixing three corner points

### Phase-correct mixing
The phase channels are stored as **cos(φ), sin(φ)**.  
Naively mixing cos and sin can break unit magnitude.

Instead we treat phase as a complex phasor:

- `u = cos(φ) + j sin(φ)`

Then:
- `u_syn = w1*uA + w2*uB + w3*uC`
- normalize: `u_syn = u_syn / (|u_syn| + eps)`
- recover:
  - `cos_syn = Re(u_syn)`
  - `sin_syn = Im(u_syn)`

RSSI is mixed linearly:
- `rssi_syn = w1*rssiA + w2*rssiB + w3*rssiC`

Label is mixed linearly:
- `y_syn = w1*yA + w2*yB + w3*yC`

### Parameters
- `AUG_PER_SIMPLEX`: how many synthetic points per triangle
- `MAX_SIMPLICES`: cap the number of triangles used (for speed)
- `AUG_SEED`: makes augmentation deterministic

### Why it helps
It densifies training data in a geometry-consistent way:
- more samples “between” known locations
- improves generalization when test points fall between training grid points

---

## Results Output (Auto-saved)

Each training run creates a timestamped folder:

- results/run_YYYYMMDD_HHMMSS[_runName]
- run_meta.json # dataset meta + train/aug hyperparams
- curves.csv # per-epoch train loss + val stats
- train_loss.png # train loss vs epoch
- train_val_curves.png # val mean/median/p95 vs epoch
- error_cdf.png # CDF of test euclidean error (meters)
- errors_test.npy # raw test errors (meters)
- final_ckpt.pt # last epoch model checkpoint
- best_ckpt.pt # best checkpoint by select_metric (default p95)

### Metrics reported
Evaluation computes Euclidean error per sample (meters) and summarizes:
- mean
- median
- p90
- p95
- max

---

## Running Training

Basic run:

python3 train.py --data rssiData.npz


## How to Run

python3 train.py --data rssiData.npz \
  --epochs 200 --batch 32 --lr 8e-4 \
  --weight_decay 5e-4 --huber_beta 0.02 --lam_euclid 0.9 \
  --aug_per_simplex 12 --max_simplices 1500 \
  --select_metric p95_m \
  --run_name sweep1

## Accuracy

<img width="1400" height="1000" alt="image" src="https://github.com/user-attachments/assets/2e02c26a-9f7c-44a1-8761-deacd1ec63c8" />

