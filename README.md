# GNN-based Drug Discovery — README

> **Repo goal**: Train, evaluate, and explain Graph Neural Networks (GCN/GAT/GIN) on MoleculeNet datasets (ESOL, BBBP, Tox21). Produce metrics (CSV), figures (loss curves, ROC-AUC), and molecule visualizations (saliency/attention).

---

## 1) Quickstart

```bash
# 0) Clone & enter (example)
git clone <your-repo-url>.git
cd gnn_midterm_starter_v2

# 1) Conda env (recommended; works on macOS M1/M2 too)
conda create -n gnn311 python=3.10 -y
conda activate gnn311

# 2) Install core deps
# RDKit from conda-forge is easiest/cross‑platform
conda install -c conda-forge rdkit -y

# DeepChem (for MoleculeNet loaders)
pip install deepchem

# PyTorch (choose one)
#   (a) CPU-only (universal)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#   (b) Apple Silicon (MPS) – optional
# pip install torch torchvision torchaudio

# 3) Project python deps
pip install -r requirements.txt
# if your repo is a package with src/, do a dev install so 'src' can be imported:
pip install -e .
```

> **Tip:** If you see `ModuleNotFoundError: No module named 'src'`, always run Python modules with the `-m` flag from the repo root: `python -m src.train ...` or `python -m scripts.evaluate_and_plot ...`.

---

## 2) Project layout (expected)

```
gnn_midterm_starter_v2/
├── configs/
│   ├── exp_esol_gcn.yaml     ├── exp_esol_gat.yaml     ├── exp_esol_gin.yaml
│   ├── exp_bbbp_gcn.yaml     ├── exp_bbbp_gat.yaml     ├── exp_bbbp_gin.yaml
│   ├── exp_tox21_gcn.yaml    ├── exp_tox21_gat.yaml    ├── exp_tox21_gin.yaml
├── runs/
│   ├── esol_gcn/ best.pt …   ├── bbbp_gat/ best.pt …   ├── tox21_gin/ best.pt …
├── results/
│   ├── esol/      metrics_*.csv, figs/
│   ├── bbbp/      metrics_*.csv, figs/ROC-AUC/
│   └── tox21/     metrics_*.csv, figs/ROC-AUC/
├── scripts/
│   ├── evaluate_and_plot.py
│   ├── compare_roc_auc.py
│   ├── visualize_molecules.py
│   └── visualize_molecule_importance.py
└── src/
    ├── train.py
    └── data/, models/, utils/ ...
```

- **`runs/<dataset>_<model>/best.pt`** — best checkpoint saved by `src.train`.
- **`results/<dataset>`** — per‑run CSV metrics and generated figures.
- **`results/<dataset>/figs`** — loss curves, scatter plots, ROC‑AUC (classification only).
- **`results/<dataset>/ROC-AUC`** — comparison plots from `compare_roc_auc.py` (BBBP/Tox21).

---

## 3) Train: 9 canonical runs (ESOL/BBBP/Tox21 × GCN/GAT/GIN)

> Run these from the **repo root**.

```bash
# ESOL (regression: RMSE/MAE/R2)
python -m src.train --config configs/exp_esol_gcn.yaml
python -m src.train --config configs/exp_esol_gat.yaml
python -m src.train --config configs/exp_esol_gin.yaml

# BBBP (classification: ROC-AUC/PR-AUC/ACC)
python -m src.train --config configs/exp_bbbp_gcn.yaml
python -m src.train --config configs/exp_bbbp_gat.yaml
python -m src.train --config configs/exp_bbbp_gin.yaml

# Tox21 (multi-label classification)
python -m src.train --config configs/exp_tox21_gcn.yaml
python -m src.train --config configs/exp_tox21_gat.yaml
python -m src.train --config configs/exp_tox21_gin.yaml
```

- Checkpoints are written to `runs/<dataset>_<model>/best.pt` (and possibly `last.pt`).

---

## 4) Evaluate & plot per‑run metrics

Generates CSVs into `results/<dataset>` and figures into `results/<dataset>/figs/`.

```bash
# ESOL examples
python -m scripts.evaluate_and_plot --config configs/exp_esol_gcn.yaml --ckpt runs/esol_gcn/best.pt --split test
python -m scripts.evaluate_and_plot --config configs/exp_esol_gat.yaml --ckpt runs/esol_gat/best.pt --split test
python -m scripts.evaluate_and_plot --config configs/exp_esol_gin.yaml --ckpt runs/esol_gin/best.pt --split test

# BBBP examples
python -m scripts.evaluate_and_plot --config configs/exp_bbbp_gcn.yaml --ckpt runs/bbbp_gcn/best.pt --split test
python -m scripts.evaluate_and_plot --config configs/exp_bbbp_gat.yaml --ckpt runs/bbbp_gat/best.pt --split test
python -m scripts.evaluate_and_plot --config configs/exp_bbbp_gin.yaml --ckpt runs/bbbp_gin/best.pt --split test

# Tox21 examples
python -m scripts.evaluate_and_plot --config configs/exp_tox21_gcn.yaml --ckpt runs/tox21_gcn/best.pt --split test
python -m scripts.evaluate_and_plot --config configs/exp_tox21_gat.yaml --ckpt runs/tox21_gat/best.pt --split test
python -m scripts.evaluate_and_plot --config configs/exp_tox21_gin.yaml --ckpt runs/tox21_gin/best.pt --split test
```

**What gets created?** (typical)
- `results/<dataset>/metrics_<dataset>_<model>_<split>.csv`
- `results/<dataset>/figs/loss_curve_<dataset>_<model>.png`
- (classification) `results/<dataset>/figs/roc_auc_<dataset>_<model>.png`

> If your **figures folder is empty**, the usual causes are:
> 1) `--split` didn’t match (e.g., your code expects `test` exactly); 2) evaluation crashed before saving; 3) paths to `results/` or `figs/` are wrong. Check your console logs and ensure the script creates folders with `Path(...).mkdir(parents=True, exist_ok=True)`.

---

## 5) Compare ROC‑AUC across models (classification datasets only)

> ESOL is **regression**, so ROC‑AUC comparison does **not** apply there. Use this for **BBBP** and **Tox21**.

```bash
# BBBP
python scripts/compare_roc_auc.py \
  --dir results/bbbp \
  --dataset bbbp \
  --split test \
  --out results/bbbp/ROC-AUC/roc_auc_comparison.png

# Tox21
python scripts/compare_roc_auc.py \
  --dir results/tox21 \
  --dataset tox21 \
  --split test \
  --out results/tox21/ROC-AUC/roc_auc_comparison.png
```

**Notes**
- The script scans CSVs under `--dir` whose filenames contain `metrics_<dataset>_*_<split>.csv`.
- It looks for AUC-like columns (case-insensitive): `auc, roc_auc, ROC-AUC, roc-auc, test_auc, val_auc`.
- If you see warnings like “No AUC-like column found … skipping.” either:
  - You’re running on a regression dataset (ESOL), or
  - Your metrics column names differ — update `AUC_CANDIDATES` in the script, or add a rename step to your evaluation CSVs.

---

## 6) Visualize molecules & importance

Generate curated molecule grids and highlight atom/bond contributions.

```bash
# Grid of molecules (e.g., top-N correctly predicted)
python scripts/visualize_molecules.py \
  --config configs/exp_esol_gin.yaml \
  --ckpt runs/esol_gin/best.pt \
  --split test \
  --n 16

# Atom/bond importance (Grad‑CAM/attention/saliency)
python scripts/visualize_molecule_importance.py \
  --config configs/exp_bbbp_gat.yaml \
  --ckpt runs/bbbp_gat/best.pt \
  --split test \
  --n 16
```

Expected outputs (examples):
- **Tox21 molecule — GCN highlights reactive sites**: `results/tox21/figs/importance_tox21_gcn_top.png`
- **ESOL molecule — GIN focuses on aromatic ring regions**: `results/esol/figs/importance_esol_gin_top.png`
- **ROC‑AUC comparison for BBBP model**: `results/bbbp/ROC-AUC/roc_auc_comparison.png`
- **Training vs Validation Loss (ESOL)**: `results/esol/figs/loss_curve_esol_<model>.png`

> **Tip:** Make sure your visualization scripts save to `results/<dataset>/figs/` and create the directory:  
> `Path("results")/dataset/"figs"`.mkdir(parents=True, exist_ok=True)`.

---

## 7) One‑command runners (optional)

We provide two convenience scripts you can use **as-is** or copy into your repo’s `scripts/`:

```bash
# make executable and run
chmod +x scripts/run_all_train_eval.sh
./scripts/run_all_train_eval.sh

chmod +x scripts/run_all_visuals.sh
./scripts/run_all_visuals.sh
```

- `scripts/run_all_train_eval.sh` — trains all 9 runs and evaluates them (writes CSVs + figs).
- `scripts/run_all_visuals.sh` — generates example visualizations for ESOL/BBBP/Tox21.

> Store these `.sh` files in the repo’s `scripts/` folder (committed to Git) so others can reproduce easily.

---

## 8) Reproducibility knobs (recommended in YAML)

- **seed**: fix random seeds for PyTorch/NumPy.
- **splitter**: scaffold vs random; report which you used.
- **metrics**: ensure regression (RMSE/MAE/R2) vs classification (ROC‑AUC/PR‑AUC/ACC) are configured.
- **output_root**: a central place like `results/<dataset>` so scripts can find CSVs.
- **checkpoint_dir**: `runs/<dataset>_<model>`.

---

## 9) Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**  
  Run from repo root and use module form: `python -m src.train ...` or `python -m scripts.evaluate_and_plot ...`. Also `pip install -e .` after cloning.

- **Figures folder is empty**  
  Usually evaluation didn’t run to completion or save paths weren’t created. Verify the script calls `mkdir(parents=True, exist_ok=True)` and check for exceptions earlier in the logs.

- **AUC script says “No AUC-like column …”**  
  You probably pointed it at ESOL (regression) or your column names differ from the defaults. Use BBBP/Tox21, or add your metric name to `AUC_CANDIDATES` in `compare_roc_auc.py`.

- **RDKit install on Apple Silicon**  
  Prefer `conda install -c conda-forge rdkit`. Avoid `pip install rdkit` on macOS.

---

## 10) What’s next (final deliverables)

- Finish **all 9 runs** and evaluations.
- Add **interpretability**: saliency/attention maps with captions.
- Add **extra datasets** (e.g., SIDER, ClinTox) for robustness.
- Try **Graph Transformer / MPNN** baselines.
- Provide a **reproducible pipeline** with checkpoints + analysis notebook.

---

## 11) Citation

If you use MoleculeNet or DeepChem loaders, please cite their respective papers/projects.
