
# MammoCRKAN (PyTorch)

Unofficial, faithful implementation of **"Rethinking Multi-view Mammogram Representation Learning via Counterfactual Reasoning with Kolmogorov–Arnold Theorem"** (MICCAI camera-ready).  
This code follows the paper's *Dual-view Mammogram Causal Graph (DMCG)* and implements both modules:

- **SSM (Spherical Sample Module)** — strengthens the **direct effect** `X → Y` by aligning cross-view tumor geometry in a spherical coordinate space.
- **KAAM (Kolmogorov–Arnold Aggregate Module)** — promotes the positive side of the **mediation effect** `X → M → Y` via:
  - **MCP** (Maximum Coverage Patch selection) over feature maps,
  - a lightweight **KAN-style** B-spline aggregator to decompose joint causality into (approximate) univariate effects.

> Paper cues used throughout (equations, figures, hyper-parameters) are annotated in the code and comments.


## Environment

```bash
conda create -n mammo_crkan python=3.10 -y
conda activate mammo_crkan
pip install -r requirements.txt
```

Tested with: Python 3.10, PyTorch ≥ 2.2, CUDA 12.x (CPU also works for sanity checks).


## Data preparation

The repo expects a single CSV per dataset that provides **paired views** from the **same breast** (ipsilateral): CC & MLO. Required columns:

- `patient_id` (string/int)
- `side` (`L` or `R`) — used to right-orient images (left breasts are horizontally flipped)
- `cc_path` (path to CC image: PNG/JPG/DICOM)  
- `mlo_path` (path to MLO image: PNG/JPG/DICOM)
- `label` (int: 0=Normal, 1=Benign, 2=Malignant; or 0/1 for binary)

See the template generator:

```bash
python scripts/prepare_pairs_template.py --out pairs.csv
```

### File formats
- PNG/JPG are supported out of the box.  
- DICOM (`.dcm`) is supported via `pydicom` if installed; otherwise, convert to PNG first.

### Example directory layout

```
/data
  ├── CBIS_DDSM/
  │    ├── images/...
  │    └── pairs.csv
  ├── INBreast/
  │    ├── images/...
  │    └── pairs.csv
  └── ...
```


## Quick start

```bash
python mammo_crkan/train.py \
  --cfg configs/default.yaml \
  --train_csv /data/CBIS_DDSM/pairs.csv \
  --val_csv /data/CBIS_DDSM/pairs_val.csv \
  --num_classes 3 \
  --outdir outputs/cbis
```

Evaluate a trained checkpoint:
```bash
python mammo_crkan/eval.py \
  --cfg configs/default.yaml \
  --test_csv /data/CBIS_DDSM/pairs_test.csv \
  --ckpt outputs/cbis/best.ckpt
```


## Reproducing paper settings

- Image resize: **2944 × 1920**, right-oriented (left breasts are horizontally flipped).
- Optimizer: **AdamW**, lr=1e-4, weight_decay=1e-4, cosine annealing (min_lr=1e-8).
- Epochs: 80 with **early stopping** on validation AUC.
- Batch size: 8 (adjust per GPU memory).
- KAAM defaults:
  - cover rate `r = 0.3`
  - patch size `512 × 512`
  - number of patches `M = 6`
  - temperature `τ = 0.5`

> Defaults reflect the strongest configurations reported (e.g., Fig. 3 and Table 3).


## Notes on KAN layer

We implement a compact **B-spline basis aggregator** that approximates the KAN behavior (arXiv:2404.19756). It maps each feature channel independently through cubic B-spline bases (uniform knots), then sums univariate responses — consistent with the Kolmogorov–Arnold theorem spirit used by the paper.


## Citation

If you use this implementation in academic work, please cite the original paper.
