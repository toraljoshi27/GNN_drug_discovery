# GNN for Molecular Property Prediction — Midterm Starter

This repository trains **GCN/GIN/GAT** on **Tox21 (multi-label)**, **BBBP (binary)**, and **ESOL (regression)** with **scaffold splits**.

## Quickstart
```bash
# 1) Create environment (edit versions as needed)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Download and preprocess data
python src/data/download_moleculenet.py
# (optional) run a quick sanity split test
python tests/test_scaffold_split.py

# 3) Train
python src/train.py --config configs/exp_tox21_gcn.yaml

# 4) Evaluate
python src/eval.py --config configs/exp_tox21_gcn.yaml

# 5) Generate interpretability visuals
python src/explain.py --config configs/exp_tox21_gcn.yaml
```

## Structure
- `configs/` — experiment YAMLs (dataset, model, split, metrics, logging)
- `src/` — data loaders, featurization, models, training, evaluation
- `results/` — CSV metrics by dataset/model
- `figures/` — plots and attributions
- `docs/midterm_report/` — LaTeX midterm template

## Reproducibility
- All configs set `seed: 42` and `split: scaffold`. See `tests/` for deterministic checks.
