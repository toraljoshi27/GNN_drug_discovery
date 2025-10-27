#!/bin/bash
# ============================================================
# Script: generate_all_molecules.sh
# Purpose: Generate 16 molecule images for all 9 GNN runs
# ============================================================

# Activate environment
conda activate gnn311

# Move to project root (in case it’s run from elsewhere)
cd /Users/toral/PycharmProjects/GNNDrugDiscovery/gnn_midterm_starter_v2

# ---- ESOL ----
python scripts/visualize_molecules.py --config configs/exp_esol_gin.yaml --ckpt runs/esol_gin/best.pt --split test --n 16
python scripts/visualize_molecules.py --config configs/exp_esol_gat.yaml --ckpt runs/esol_gat/best.pt --split test --n 16
python scripts/visualize_molecules.py --config configs/exp_esol_gcn.yaml --ckpt runs/esol_gcn/best.pt --split test --n 16

# ---- BBBP ----
python scripts/visualize_molecules.py --config configs/exp_bbbp_gin.yaml --ckpt runs/bbbp_gin/best.pt --split test --n 16
python scripts/visualize_molecules.py --config configs/exp_bbbp_gat.yaml --ckpt runs/bbbp_gat/best.pt --split test --n 16
python scripts/visualize_molecules.py --config configs/exp_bbbp_gcn.yaml --ckpt runs/bbbp_gcn/best.pt --split test --n 16

# ---- TOX21 ----
python scripts/visualize_molecules.py --config configs/exp_tox21_gin.yaml --ckpt runs/tox21_gin/best.pt --split test --n 16
python scripts/visualize_molecules.py --config configs/exp_tox21_gat.yaml --ckpt runs/tox21_gat/best.pt --split test --n 16
python scripts/visualize_molecules.py --config configs/exp_tox21_gcn.yaml --ckpt runs/tox21_gcn/best.pt --split test --n 16

echo "✅ All molecule images generated successfully!"
