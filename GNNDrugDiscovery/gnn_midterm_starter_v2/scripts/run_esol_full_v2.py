# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Complete ESOL experiments with GCN, GIN, and GAT.
#
# This version saves results into:
#     results/esol_full_run/metrics/
#     results/esol_full_run/figs/
#
# What the script does:
#   - Load ESOL dataset from MoleculeNet (PyTorch Geometric)
#   - Create scaffold-based train/val/test split
#   - Train GCN, GIN, GAT regression models
#   - Compute RMSE and MAE
#   - Save metrics + learning curves + parity plots + error histograms
# """
#
# import os
# import json
# import random
# from collections import defaultdict
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch_geometric.datasets import MoleculeNet
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import (
#     GCNConv,
#     GINConv,
#     GATConv,
#     global_mean_pool,
#     BatchNorm,
# )
#
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
#
# # ---------------------------------------------------------
# # 1. Utils: seeds, device, new output directories
# # ---------------------------------------------------------
#
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set_seed(42)
#
# # NEW folder instead of overwriting existing results
# RESULTS_DIR = "results/esol_full_run"
# FIG_DIR = os.path.join(RESULTS_DIR, "figs")
# METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")
#
# os.makedirs(FIG_DIR, exist_ok=True)
# os.makedirs(METRIC_DIR, exist_ok=True)
#
#
# # ---------------------------------------------------------
# # 2. Scaffold split
# # ---------------------------------------------------------
#
# def generate_scaffold(smiles, include_chirality=True):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     try:
#         scaffold = MurckoScaffold.MurckoScaffoldSmiles(
#             mol=mol, includeChirality=include_chirality
#         )
#     except Exception:
#         scaffold = None
#     return scaffold
#
#
# def scaffold_split_indices(dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1):
#     """Create scaffold-based train/val/test split for MoleculeNet datasets."""
#     assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
#
#     scaffolds = defaultdict(list)
#
#     for idx, data in enumerate(dataset):
#         smiles = data.smiles if hasattr(data, "smiles") else None
#         if smiles is None:
#             raise ValueError("Dataset does not contain SMILES information.")
#         scaffold = generate_scaffold(smiles)
#         scaffold = scaffold if scaffold is not None else f"None_{idx}"
#         scaffolds[scaffold].append(idx)
#
#     scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
#
#     n_total = len(dataset)
#     train_cutoff = train_frac * n_total
#     val_cutoff = (train_frac + val_frac) * n_total
#
#     train_idx, val_idx, test_idx = [], [], []
#     n_train = n_val = n_test = 0
#
#     for scaf_indices in scaffold_sets:
#         if n_train + len(scaf_indices) <= train_cutoff:
#             train_idx.extend(scaf_indices)
#             n_train += len(scaf_indices)
#         elif n_val + len(scaf_indices) <= val_cutoff:
#             val_idx.extend(scaf_indices)
#             n_val += len(scaf_indices)
#         else:
#             test_idx.extend(scaf_indices)
#             n_test += len(scaf_indices)
#
#     return train_idx, val_idx, test_idx
#
#
# # ---------------------------------------------------------
# # 3. Model architectures: GCN, GIN, GAT
# # ---------------------------------------------------------
#
# class GCNRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#
#         self.convs.append(GCNConv(in_dim, hidden_dim))
#         self.bns.append(BatchNorm(hidden_dim))
#         for _ in range(num_layers - 1):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
#             self.bns.append(BatchNorm(hidden_dim))
#
#         self.dropout = dropout
#         self.lin1 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x, edge_index, batch):
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_mean_pool(x, batch)
#         x = F.relu(self.lin1(x))
#         return self.lin2(x).view(-1)
#
#
# class GINRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.3):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#
#         nn1 = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.convs.append(GINConv(nn1))
#         self.bns.append(BatchNorm(hidden_dim))
#
#         for _ in range(num_layers - 1):
#             nn_layer = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim)
#             )
#             self.convs.append(GINConv(nn_layer))
#             self.bns.append(BatchNorm(hidden_dim))
#
#         self.dropout = dropout
#         self.lin1 = nn.Linear(hidden_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x, edge_index, batch):
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_mean_pool(x, batch)
#         x = F.relu(self.lin1(x))
#         return self.lin2(x).view(-1)
#
#
# class GATRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim=64, num_layers=3, heads=4, dropout=0.3):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#
#         self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
#         self.bns.append(BatchNorm(hidden_dim * heads))
#
#         for _ in range(num_layers - 1):
#             self.convs.append(
#                 GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
#             )
#             self.bns.append(BatchNorm(hidden_dim * heads))
#
#         self.dropout = dropout
#         self.lin1 = nn.Linear(hidden_dim * heads, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x, edge_index, batch):
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = F.elu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_mean_pool(x, batch)
#         x = F.elu(self.lin1(x))
#         return self.lin2(x).view(-1)
#
#
# def build_model(model_type, in_dim):
#     if model_type == "gcn":
#         return GCNRegressor(in_dim)
#     if model_type == "gin":
#         return GINRegressor(in_dim)
#     if model_type == "gat":
#         return GATRegressor(in_dim)
#     raise ValueError(f"Unknown model type: {model_type}")
#
#
# # ---------------------------------------------------------
# # 4. Training & evaluation
# # ---------------------------------------------------------
#
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss = 0
#     n = 0
#
#     for batch in loader:
#         batch = batch.to(DEVICE)
#         optimizer.zero_grad()
#         pred = model(batch.x, batch.edge_index, batch.batch)
#         y = batch.y.view(-1).float()
#         loss = criterion(pred, y)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item() * batch.num_graphs
#         n += batch.num_graphs
#
#     return total_loss / n
#
#
# @torch.no_grad()
# def eval_model(model, loader):
#     model.eval()
#     all_true, all_pred = [], []
#
#     for batch in loader:
#         batch = batch.to(DEVICE)
#         pred = model(batch.x, batch.edge_index, batch.batch)
#         y = batch.y.view(-1).float()
#         all_pred.append(pred.cpu().numpy())
#         all_true.append(y.cpu().numpy())
#
#     y_true = np.concatenate(all_true)
#     y_pred = np.concatenate(all_pred)
#
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#
#     return rmse, mae, y_true, y_pred
#
#
# # ---------------------------------------------------------
# # 5. Plot helpers
# # ---------------------------------------------------------
#
# def plot_learning_curves(train_losses, val_rmses, model_name):
#     plt.figure()
#     epochs = np.arange(1, len(train_losses) + 1)
#     plt.plot(epochs, train_losses, label="Train MSE")
#     plt.plot(epochs, val_rmses, label="Val RMSE")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss / RMSE")
#     plt.title(f"ESOL Learning Curves - {model_name.upper()}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(FIG_DIR, f"{model_name}_learning_curves.png"))
#     plt.close()
#
#
# def plot_parity(y_true, y_pred, model_name):
#     plt.figure()
#     plt.scatter(y_true, y_pred, alpha=0.6)
#     min_v = min(y_true.min(), y_pred.min())
#     max_v = max(y_true.max(), y_pred.max())
#     plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
#     plt.xlabel("True logS")
#     plt.ylabel("Predicted logS")
#     plt.title(f"ESOL Parity Plot - {model_name.upper()}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(FIG_DIR, f"{model_name}_parity.png"))
#     plt.close()
#
#
# def plot_error_hist(y_true, y_pred, model_name):
#     errors = y_pred - y_true
#     plt.figure()
#     plt.hist(errors, bins=30, alpha=0.8)
#     plt.xlabel("Prediction Error")
#     plt.ylabel("Count")
#     plt.title(f"ESOL Error Histogram - {model_name.upper()}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(FIG_DIR, f"{model_name}_error_hist.png"))
#     plt.close()
#
#
# # ---------------------------------------------------------
# # 6. Main experiment function
# # ---------------------------------------------------------
#
# def run_esol(model_type="gcn", epochs=200, batch_size=64, lr=1e-3, patience=25):
#     print(f"\n=== Training ESOL model: {model_type.upper()} ===")
#
#     dataset = MoleculeNet(root="data/moleculenet", name="ESOL")
#     in_dim = dataset.num_node_features
#
#     train_idx, val_idx, test_idx = scaffold_split_indices(dataset)
#
#     train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)
#
#     model = build_model(model_type, in_dim).to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#
#     train_losses, val_rmses = [], []
#     best_val_rmse = float("inf")
#     best_state = None
#     no_improve = 0
#
#     # Training loop
#     for epoch in range(1, epochs + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
#         val_rmse, val_mae, _, _ = eval_model(model, val_loader)
#
#         train_losses.append(train_loss)
#         val_rmses.append(val_rmse)
#
#         print(f"Epoch {epoch} | Train MSE={train_loss:.4f} | Val RMSE={val_rmse:.4f}")
#
#         if val_rmse < best_val_rmse - 1e-4:
#             best_val_rmse = val_rmse
#             best_state = model.state_dict()
#             no_improve = 0
#         else:
#             no_improve += 1
#
#         if no_improve >= patience:
#             print(f"Early stopping at epoch {epoch}")
#             break
#
#     # Load best model
#     if best_state:
#         model.load_state_dict(best_state)
#
#     # Final evaluation
#     train_rmse, train_mae, _, _ = eval_model(model, train_loader)
#     val_rmse, val_mae, _, _ = eval_model(model, val_loader)
#     test_rmse, test_mae, y_test_true, y_test_pred = eval_model(model, test_loader)
#
#     metrics = {
#         "model": model_type,
#         "train_rmse": float(train_rmse),
#         "train_mae": float(train_mae),
#         "val_rmse": float(val_rmse),
#         "val_mae": float(val_mae),
#         "test_rmse": float(test_rmse),
#         "test_mae": float(test_mae),
#     }
#
#     # Save metrics
#     with open(os.path.join(METRIC_DIR, f"{model_type}_metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=2)
#
#     # Save plots
#     plot_learning_curves(train_losses, val_rmses, model_type)
#     plot_parity(y_test_true, y_test_pred, model_type)
#     plot_error_hist(y_test_true, y_test_pred, model_type)
#
#     print(f"Saved metrics & plots for {model_type}.")
#
#
# # ---------------------------------------------------------
# # 7. Run all models
# # ---------------------------------------------------------
#
# if __name__ == "__main__":
#     for m in ["gcn", "gin", "gat"]:
#         run_esol(model_type=m)


#!/usr/bin/env python

# !/usr/bin/env python

#!/usr/bin/env python

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool

# -------------------------------------------------------------------
# Make project root importable
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Your scaffold split (uses RDKit & SMILES)
from src.data.scaffold_split import scaffold_split


# ==================== UTILITIES ==================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== SIMPLE LOCAL MODELS ==================== #

class SimpleGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        task: str,
    ):
        super().__init__()
        self.task = task

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Manual linear head to avoid any weirdness
        self.head_weight = nn.Parameter(torch.empty(out_dim, hidden_dim))
        self.head_bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.head_weight)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # CRITICAL: GNN expects float features
        x = x.to(torch.float32)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_add_pool(x, batch)  # [num_graphs, hidden_dim]

        out = torch.matmul(x, self.head_weight.t()) + self.head_bias  # [N, out_dim]
        if out.dim() == 2 and out.shape[1] == 1:
            out = out.view(-1)
        return out


class SimpleGIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        task: str,
    ):
        super().__init__()
        self.task = task

        def mlp_block(in_dim, out_dim_):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim_),
            )

        self.convs = nn.ModuleList()
        self.convs.append(GINConv(mlp_block(in_channels, hidden_dim)))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(mlp_block(hidden_dim, hidden_dim)))

        self.dropout = nn.Dropout(dropout)

        self.head_weight = nn.Parameter(torch.empty(out_dim, hidden_dim))
        self.head_bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.head_weight)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x.to(torch.float32)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_add_pool(x, batch)

        out = torch.matmul(x, self.head_weight.t()) + self.head_bias
        if out.dim() == 2 and out.shape[1] == 1:
            out = out.view(-1)
        return out


class SimpleGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        task: str,
        heads: int = 4,
    ):
        super().__init__()
        self.task = task

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))
        in_dim = hidden_dim * heads
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
            in_dim = hidden_dim * heads

        self.dropout = nn.Dropout(dropout)

        self.head_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.head_bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.head_weight)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x.to(torch.float32)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_add_pool(x, batch)

        out = torch.matmul(x, self.head_weight.t()) + self.head_bias
        if out.dim() == 2 and out.shape[1] == 1:
            out = out.view(-1)
        return out


def build_model(
    model_name: str,
    num_node_features: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    model_name = model_name.lower()
    task = "regression"  # ESOL

    if model_name == "gcn":
        return SimpleGCN(
            num_node_features,  # in_channels
            hidden_dim,
            out_dim,
            num_layers,
            dropout,
            task,
        )
    elif model_name == "gin":
        return SimpleGIN(
            num_node_features,
            hidden_dim,
            out_dim,
            num_layers,
            dropout,
            task,
        )
    elif model_name == "gat":
        return SimpleGAT(
            num_node_features,
            hidden_dim,
            out_dim,
            num_layers,
            dropout,
            task,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== DATA LOADING ==================== #

def load_esol_dataloaders(
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load ESOL via MoleculeNet and split using your scaffold_split(smiles_list,...).
    """
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="ESOL")

    # Build SMILES list for your scaffold_split
    smiles_list: List[str] = []
    for data in dataset:
        smi = getattr(data, "smiles", None)
        if smi is None:
            raise ValueError("Data object missing `smiles` attribute.")
        smiles_list.append(smi)

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        seed=seed,
        frac_train=0.8,
        frac_val=0.1,
        frac_test=0.1,
    )

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ==================== TRAIN / EVAL ==================== #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        if out.dim() > 1:
            out = out.view(-1)
        y = batch.y.view(-1).to(torch.float32)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs

    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if out.dim() > 1:
                out = out.view(-1)
            y = batch.y.view(-1).to(torch.float32)
            ys.append(y.cpu())
            preds.append(out.cpu())

    if not ys:
        return {"rmse": float("nan"), "mae": float("nan")}

    y_all = torch.cat(ys)
    pred_all = torch.cat(preds)

    mse = torch.mean((pred_all - y_all) ** 2).item()
    rmse = float(mse ** 0.5)
    mae = float(torch.mean(torch.abs(pred_all - y_all)).item())

    return {"rmse": rmse, "mae": mae}


def run_single_seed(
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader = load_esol_dataloaders(
        batch_size=cfg["batch_size"],
        seed=seed,
    )

    sample_batch = next(iter(train_loader))
    num_node_features = sample_batch.x.size(-1)
    out_dim = 1  # ESOL is scalar regression

    model = build_model(
        model_name=model_name,
        num_node_features=num_node_features,
        out_dim=out_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_val = float("inf")
    best_state = None

    epoch_history: List[Dict[str, float]] = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )
        val_rmse = val_metrics["rmse"]
        val_mae = val_metrics.get("mae", float("nan"))

        print(
            f"[{model_name} | seed {seed} | epoch {epoch}] "
            f"train_loss={train_loss:.4f} val_rmse={val_rmse:.4f}"
        )
        # log this epoch
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
            }
        )


        if val_rmse < best_val:
            best_val = val_rmse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    print(
        f"[{model_name} | seed {seed}] "
        f"TEST: rmse={test_metrics['rmse']:.4f}, mae={test_metrics['mae']:.4f}"
    )

    # keep rmse/mae at top-level so aggregate_results still works
    result: Dict[str, Any] = {
        "rmse": float(test_metrics["rmse"]),
        "mae": float(test_metrics.get("mae", float("nan"))),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "epoch_history": epoch_history,
    }
    return result
    # return test_metrics


def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    all_metrics = {}
    for metrics in results:
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    agg = {}
    for k, vs in all_metrics.items():
        arr = np.asarray(vs, dtype=float)
        agg[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
        }
    return agg


# ==================== MAIN ==================== #

# def load_summary_json(path: Path):
#     if not path.exists():
#         raise FileNotFoundError(f"Summary JSON not found at: {path}")
#     with path.open("r") as f:
#         return json.load(f)
#
#
# def build_agg_dataframe(summary: dict) -> pd.DataFrame:
#     """
#     Build aggregate stats DataFrame:
#     columns: model, rmse_mean, rmse_std, mae_mean, mae_std
#     """
#     rows = []
#     for model_name, res in summary.items():
#         agg = res.get("agg", {})
#         rmse = agg.get("rmse", {})
#         mae = agg.get("mae", {})
#
#         rows.append(
#             {
#                 "model": model_name.upper(),
#                 "rmse_mean": rmse.get("mean", float("nan")),
#                 "rmse_std": rmse.get("std", float("nan")),
#                 "mae_mean": mae.get("mean", float("nan")),
#                 "mae_std": mae.get("std", float("nan")),
#             }
#         )
#
#     df = pd.DataFrame(rows)
#     # Nice ordering
#     df = df[["model", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]]
#     return df
#
#
# def build_per_seed_dataframe(summary: dict) -> pd.DataFrame:
#     """
#     Build per-seed DataFrame:
#     columns: model, seed_index, rmse, mae
#     where seed_index is 0,1,2,... in the order runs were done.
#     """
#     rows = []
#     for model_name, res in summary.items():
#         per_seed = res.get("per_seed", [])
#         for seed_idx, metrics in enumerate(per_seed):
#             rows.append(
#                 {
#                     "model": model_name.upper(),
#                     "seed_index": seed_idx,
#                     "rmse": metrics.get("rmse", float("nan")),
#                     "mae": metrics.get("mae", float("nan")),
#                 }
#             )
#
#     df = pd.DataFrame(rows)
#     df = df[["model", "seed_index", "rmse", "mae"]]
#     return df
#
#
# def export_to_excel(summary: dict, out_path: Path):
#     """
#     Create an Excel file with:
#     - Sheet 'aggregate': mean ± std per model
#     - Sheet 'per_seed': each seed's metrics per model
#     """
#     agg_df = build_agg_dataframe(summary)
#     per_seed_df = build_per_seed_dataframe(summary)
#
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#         agg_df.to_excel(writer, sheet_name="aggregate", index=False)
#         per_seed_df.to_excel(writer, sheet_name="per_seed", index=False)
#
#     print(f"[OK] Wrote detailed Excel results to {out_path}")
#
#
# def main():
#     base_dir = Path("results/esol")
#     summary_json_path = base_dir / "esol_all_models_summary.json"
#     excel_out_path = base_dir / "esol_results_detailed.xlsx"
#
#     summary = load_summary_json(summary_json_path)
#     export_to_excel(summary, excel_out_path)

#_________________________________________________________________________

#
# def main():
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="ESOL experiments with Simple GCN/GIN/GAT."
#     )
#     parser.add_argument(
#         "--models",
#         type=str,
#         default="gcn,gin,gat",
#         help="Comma-separated list: gcn, gin, gat",
#     )
#     parser.add_argument(
#         "--seeds",
#         type=int,
#         default=3,
#         help="Number of random seeds.",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="results/esol",
#         help="Directory to store JSON results.",
#     )
#
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--hidden_dim", type=int, default=128)
#     parser.add_argument("--num_layers", type=int, default=3)
#     parser.add_argument("--dropout", type=float, default=0.2)
#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--weight_decay", type=float, default=1e-5)
#     parser.add_argument("--epochs", type=int, default=50)
#
#     args = parser.parse_args()
#
#     cfg = {
#         "batch_size": args.batch_size,
#         "hidden_dim": args.hidden_dim,
#         "num_layers": args.num_layers,
#         "dropout": args.dropout,
#         "lr": args.lr,
#         "weight_decay": args.weight_decay,
#         "epochs": args.epochs,
#     }
#
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
#     all_results = {}
#
#     for model_name in model_names:
#         print(f"\n===== {model_name.upper()} on ESOL ({args.seeds} seeds) =====")
#         seed_results: List[Dict[str, float]] = []
#
#         for i in range(args.seeds):
#             seed = 42 + i
#             metrics = run_single_seed(cfg, model_name, seed)
#             seed_results.append(metrics)
#
#         agg = aggregate_results(seed_results)
#         all_results[model_name] = {"per_seed": seed_results, "agg": agg}
#
#         model_out = output_dir / f"{model_name}_esol_results.json"
#         with open(model_out, "w") as f:
#             json.dump(all_results[model_name], f, indent=2)
#
#         print(f"\n>>> {model_name.upper()} ESOL: mean ± std <<<")
#         for metric_name, stats in agg.items():
#             print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
#
#     summary_path = output_dir / "esol_all_models_summary.json"
#     with open(summary_path, "w") as f:
#         json.dump(all_results, f, indent=2)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ESOL experiments with Simple GCN/GIN/GAT."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gcn,gin,gat",
        help="Comma-separated list: gcn, gin, gat",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/esol",
        help="Directory to store JSON/Excel results.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    cfg = {
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    all_results: Dict[str, Any] = {}

    # collect per-epoch rows for Excel
    epoch_rows: List[Dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n===== {model_name.upper()} on ESOL ({args.seeds} seeds) =====")
        seed_results: List[Dict[str, float]] = []

        for i in range(args.seeds):
            seed = 42 + i
            res = run_single_seed(cfg, model_name, seed)

            # for aggregate stats
            seed_results.append(
                {
                    "rmse": res["rmse"],
                    "mae": res["mae"],
                }
            )

            # expand epoch_history into flat rows
            for ep in res["epoch_history"]:
                epoch_rows.append(
                    {
                        "model": model_name.upper(),
                        "seed": seed,
                        "epoch": ep["epoch"],
                        "train_loss": ep["train_loss"],
                        "val_rmse": ep["val_rmse"],
                        "val_mae": ep["val_mae"],
                    }
                )

        agg = aggregate_results(seed_results)
        all_results[model_name] = {"per_seed": seed_results, "agg": agg}

        model_out = output_dir / f"{model_name}_esol_results.json"
        with open(model_out, "w") as f:
            json.dump(all_results[model_name], f, indent=2)

        print(f"\n>>> {model_name.upper()} ESOL: mean ± std <<<")
        for metric_name, stats in agg.items():
            print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Write combined JSON summary (unchanged)
    summary_path = output_dir / "esol_all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # === Build DataFrames for Excel ===
    # 1) Epoch-wise DataFrame
    df_epochs = pd.DataFrame(epoch_rows)

    # 2) Summary DataFrame
    summary_rows = []
    for model_name, res in all_results.items():
        agg = res["agg"]
        rmse = agg.get("rmse", {})
        mae = agg.get("mae", {})
        summary_rows.append(
            {
                "model": model_name.upper(),
                "rmse_mean": rmse.get("mean", float("nan")),
                "rmse_std": rmse.get("std", float("nan")),
                "mae_mean": mae.get("mean", float("nan")),
                "mae_std": mae.get("std", float("nan")),
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary[["model", "rmse_mean", "rmse_std", "mae_mean", "mae_std"]]

    # === Export to Excel ===
    excel_path = output_dir / "esol_results_detailed.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_epochs.to_excel(writer, sheet_name="epochs", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"[OK] Wrote detailed Excel results to {excel_path}")


if __name__ == "__main__":
    main()
