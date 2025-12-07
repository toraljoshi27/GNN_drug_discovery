#!/usr/bin/env python

"""
Complete Tox21 experiments with GCN, GIN, and GAT (classification).

This script:
  - Loads Tox21 from MoleculeNet
  - Uses scaffold-based train/val/test split via src.data.scaffold_split.scaffold_split
  - Trains Simple GCN / GIN / GAT classification models
  - Handles multi-task labels with missing entries (-1 / NaN) via masked BCE
  - Computes macro ROC-AUC across tasks
  - Runs multiple seeds per model
  - Saves:
      * Per-model JSON (per_seed metrics + aggregate)
      * Combined JSON summary
      * Excel with:
          - 'epochs' sheet: all epochs for all seeds and models
          - 'summary' sheet: mean ± std AUC/BCE per model
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# Make project root importable
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Your scaffold split (expects list of SMILES)
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

        x = global_add_pool(x, batch)  # [N, hidden_dim]
        out = torch.matmul(x, self.head_weight.t()) + self.head_bias  # [N, out_dim]
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
    task = "classification"  # Tox21

    if model_name == "gcn":
        return SimpleGCN(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
    elif model_name == "gin":
        return SimpleGIN(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
    elif model_name == "gat":
        return SimpleGAT(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== DATA LOADING ==================== #

def load_tox21_dataloaders(
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load Tox21 via MoleculeNet and split using your scaffold_split(smiles_list,...).
    """
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="Tox21")

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


# ==================== CLASSIFICATION LOSS & METRICS ==================== #

def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy with logits, masking out missing labels.
    Treat labels == -1 or NaN as missing.
    logits, labels: [N, T]
    """
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)

    mask = ~torch.isnan(labels) & (labels != -1)
    if mask.sum() == 0:
        # no valid labels in this batch
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits = logits[mask]
    labels = labels[mask]
    return F.binary_cross_entropy_with_logits(logits, labels)


def compute_multitask_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute macro AUC across tasks, ignoring tasks with only one class
    or all-missing labels.
    logits, labels: [N, T]
    """
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)

    y = labels.cpu().numpy()
    y_pred = torch.sigmoid(logits).cpu().numpy()

    num_tasks = y.shape[1]
    aucs = []

    for t in range(num_tasks):
        y_t = y[:, t]
        p_t = y_pred[:, t]

        # mask missing labels
        mask = ~np.isnan(y_t) & (y_t != -1)
        y_t = y_t[mask]
        p_t = p_t[mask]

        # need at least one positive and one negative
        if y_t.size == 0 or len(np.unique(y_t)) < 2:
            continue

        aucs.append(roc_auc_score(y_t, p_t))

    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


# ==================== TRAIN / EVAL ==================== #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)  # [B, T]
        labels = batch.y

        loss = masked_bce_loss(logits, labels)
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
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            labels = batch.y

            if labels.dim() == 1:
                labels = labels.view(-1, 1)
            if logits.dim() == 1:
                logits = logits.view(-1, 1)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    if not all_logits:
        return {"auc": float("nan"), "bce": float("nan")}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    auc = compute_multitask_auc(logits, labels)
    bce = masked_bce_loss(logits, labels).item()

    return {"auc": auc, "bce": bce}


# ==================== AGGREGATION ==================== #

def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, List[float]] = {}
    for metrics in results:
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    agg: Dict[str, Dict[str, float]] = {}
    for k, vs in all_metrics.items():
        arr = np.asarray(vs, dtype=float)
        agg[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
        }
    return agg


# ==================== RUN SINGLE SEED ==================== #

def run_single_seed(
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()

    train_loader, val_loader, test_loader = load_tox21_dataloaders(
        batch_size=cfg["batch_size"],
        seed=seed,
    )

    sample_batch = next(iter(train_loader))
    num_node_features = sample_batch.x.size(-1)
    # Multi-task: use y.shape[1] if 2D, else 1
    out_dim = sample_batch.y.size(-1) if sample_batch.y.dim() > 1 else 1

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

    best_val_auc = -1.0
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
        val_auc = val_metrics["auc"]
        val_bce = val_metrics["bce"]

        print(
            f"[{model_name} | seed {seed} | epoch {epoch}] "
            f"train_loss={train_loss:.4f} val_auc={val_auc:.4f}"
        )

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auc": float(val_auc),
                "val_bce": float(val_bce),
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
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
        f"TEST: auc={test_metrics['auc']:.4f}, bce={test_metrics['bce']:.4f}"
    )

    result: Dict[str, Any] = {
        "auc": float(test_metrics["auc"]),
        "bce": float(test_metrics["bce"]),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "epoch_history": epoch_history,
    }
    return result


# ==================== MAIN ==================== #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tox21 experiments with Simple GCN/GIN/GAT."
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
        default="results/tox21",
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

    epoch_rows: List[Dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n===== {model_name.upper()} on Tox21 ({args.seeds} seeds) =====")
        seed_results: List[Dict[str, float]] = []

        for i in range(args.seeds):
            seed = 42 + i
            res = run_single_seed(cfg, model_name, seed)

            # for aggregate stats
            seed_results.append(
                {
                    "auc": res["auc"],
                    "bce": res["bce"],
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
                        "val_auc": ep["val_auc"],
                        "val_bce": ep["val_bce"],
                    }
                )

        agg = aggregate_results(seed_results)
        all_results[model_name] = {"per_seed": seed_results, "agg": agg}

        model_out = output_dir / f"{model_name}_tox21_results.json"
        with open(model_out, "w") as f:
            json.dump(all_results[model_name], f, indent=2)

        print(f"\n>>> {model_name.upper()} Tox21: mean ± std AUC/BCE <<<")
        for metric_name, stats in agg.items():
            print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Combined JSON summary
    summary_path = output_dir / "tox21_all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # === Build DataFrames for Excel ===
    # 1) Epoch-wise
    df_epochs = pd.DataFrame(epoch_rows)

    # 2) Summary
    summary_rows = []
    for model_name, res in all_results.items():
        agg = res["agg"]
        auc = agg.get("auc", {})
        bce = agg.get("bce", {})
        summary_rows.append(
            {
                "model": model_name.upper(),
                "auc_mean": auc.get("mean", float("nan")),
                "auc_std": auc.get("std", float("nan")),
                "bce_mean": bce.get("mean", float("nan")),
                "bce_std": bce.get("std", float("nan")),
            }
        )
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary[["model", "auc_mean", "auc_std", "bce_mean", "bce_std"]]

    # Export to Excel
    excel_path = output_dir / "tox21_results_detailed.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_epochs.to_excel(writer, sheet_name="epochs", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"Detailed Results for tox21 are exported to {excel_path}")


if __name__ == "__main__":
    main()
