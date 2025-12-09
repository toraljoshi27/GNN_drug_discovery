#!/usr/bin/env python
"""
IMPROVED Tox21 GNN Training Script

Key improvements over original:
1. Class imbalance handling with pos_weight per task
2. Higher capacity models (hidden_dim=256, num_layers=5)
3. Batch normalization between layers
4. Cosine annealing LR scheduler
5. Stronger training (epochs=200, batch_size=32)
6. JumpingKnowledge for better feature aggregation
7. Gradient clipping for stability

Target: Beat RF baseline (AUC ≈ 0.749)
"""

import os
import sys
import json
import random
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, 
    global_add_pool, global_mean_pool, global_max_pool,
    BatchNorm
)
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# Make project root importable
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==================== IMPROVED MODELS WITH BATCH NORM ==================== #

class ImprovedGCN(nn.Module):
    """GCN with BatchNorm, residual connections, and JumpingKnowledge."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # GCN layers with batch norm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # JumpingKnowledge: concatenate all layer outputs
        self.jk_linear = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Collect layer outputs for JumpingKnowledge
        layer_outputs = []
        
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            # Residual connection
            x = x + x_new
            layer_outputs.append(global_add_pool(x, batch))
        
        # JumpingKnowledge aggregation
        x = torch.cat(layer_outputs, dim=-1)
        x = self.jk_linear(x)
        x = F.relu(x)
        
        return self.head(x)


class ImprovedGIN(nn.Module):
    """GIN with BatchNorm and JumpingKnowledge."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # JumpingKnowledge
        self.jk_linear = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        layer_outputs = []
        
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual
            layer_outputs.append(global_add_pool(x, batch))
        
        x = torch.cat(layer_outputs, dim=-1)
        x = self.jk_linear(x)
        x = F.relu(x)
        
        return self.head(x)


class ImprovedGAT(nn.Module):
    """GAT with BatchNorm, residual connections, and multi-head attention."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        heads: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # Initial projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # JumpingKnowledge
        self.jk_linear = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.input_proj(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        layer_outputs = []
        
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual
            layer_outputs.append(global_add_pool(x, batch))
        
        x = torch.cat(layer_outputs, dim=-1)
        x = self.jk_linear(x)
        x = F.elu(x)
        
        return self.head(x)


def build_model(
    model_name: str,
    num_node_features: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    model_name = model_name.lower()
    
    if model_name == "gcn":
        return ImprovedGCN(num_node_features, hidden_dim, out_dim, num_layers, dropout)
    elif model_name == "gin":
        return ImprovedGIN(num_node_features, hidden_dim, out_dim, num_layers, dropout)
    elif model_name == "gat":
        return ImprovedGAT(num_node_features, hidden_dim, out_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== DATA LOADING ==================== #

def load_tox21_dataloaders(
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Load Tox21 and compute class weights for imbalance handling.
    Returns: train_loader, val_loader, test_loader, pos_weight
    """
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="Tox21")

    smiles_list = [getattr(data, "smiles", "") for data in dataset]
    
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

    # Compute pos_weight for class imbalance (on training set only)
    pos_weight = compute_pos_weight(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weight


def compute_pos_weight(dataset) -> torch.Tensor:
    """
    Compute pos_weight = num_negatives / num_positives for each task.
    This handles class imbalance in multi-label classification.
    """
    all_labels = []
    for data in dataset:
        y = data.y.view(-1) if data.y.dim() == 1 else data.y.view(1, -1)
        all_labels.append(y)
    
    labels = torch.cat([l.view(1, -1) if l.dim() == 1 else l for l in all_labels], dim=0)
    num_tasks = labels.size(1)
    
    pos_weight = torch.ones(num_tasks)
    
    for t in range(num_tasks):
        y_t = labels[:, t]
        mask = (~torch.isnan(y_t)) & (y_t != -1)
        y_valid = y_t[mask]
        
        if y_valid.numel() == 0:
            continue
            
        num_pos = (y_valid == 1).sum().float()
        num_neg = (y_valid == 0).sum().float()
        
        if num_pos > 0:
            pos_weight[t] = num_neg / num_pos
        else:
            pos_weight[t] = 1.0
    
    # Use sqrt scaling for very imbalanced tasks (more stable than raw ratio)
    # This gives effective weights in range [1, ~5-6] for most imbalanced cases
    pos_weight = torch.sqrt(pos_weight)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)
    
    print(f"[INFO] Computed pos_weight per task (sqrt-scaled):")
    for t, w in enumerate(pos_weight):
        print(f"  Task {t}: {w:.2f}")
    
    return pos_weight


# ==================== LOSS & METRICS ==================== #

def weighted_masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted BCE with logits, masking out missing labels.
    Computes loss per-task then averages for proper weighting.
    """
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)

    # Clamp logits for numerical stability
    logits = torch.clamp(logits, min=-15, max=15)

    num_tasks = labels.size(1)
    task_losses = []
    
    for t in range(num_tasks):
        logits_t = logits[:, t]
        labels_t = labels[:, t]
        
        mask_t = ~torch.isnan(labels_t) & (labels_t != -1)
        
        if mask_t.sum() == 0:
            continue
        
        valid_logits = logits_t[mask_t]
        valid_labels = labels_t[mask_t]
        pw = pos_weight[t].to(logits.device)
        
        # Compute weighted BCE for this task
        loss_t = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            pos_weight=pw.expand_as(valid_logits),
            reduction='mean'
        )
        task_losses.append(loss_t)
    
    if len(task_losses) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    return torch.stack(task_losses).mean()


def unweighted_masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute UNWEIGHTED BCE for proper calibration comparison.
    This is what should be ~0.3-0.5 for a well-calibrated model.
    """
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)
    
    logits = torch.clamp(logits, min=-15, max=15)
    
    mask = ~torch.isnan(labels) & (labels != -1)
    if mask.sum() == 0:
        return float("nan")
    
    valid_logits = logits[mask]
    valid_labels = labels[mask]
    
    # Unweighted BCE
    bce = F.binary_cross_entropy_with_logits(
        valid_logits, valid_labels, reduction='mean'
    )
    return bce.item()


def compute_multitask_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute macro AUC across tasks."""
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)

    # Clamp logits to avoid numerical issues
    logits = torch.clamp(logits, min=-20, max=20)
    
    y = labels.cpu().numpy()
    y_pred = torch.sigmoid(logits).cpu().numpy()

    num_tasks = y.shape[1]
    aucs = []

    for t in range(num_tasks):
        y_t = y[:, t]
        p_t = y_pred[:, t]

        # Mask missing labels AND NaN predictions
        mask = ~np.isnan(y_t) & (y_t != -1) & ~np.isnan(p_t)
        y_t = y_t[mask]
        p_t = p_t[mask]

        if y_t.size == 0 or len(np.unique(y_t)) < 2:
            continue
        
        # Clip predictions to valid range
        p_t = np.clip(p_t, 1e-7, 1 - 1e-7)

        aucs.append(roc_auc_score(y_t, p_t))

    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


# ==================== TRAIN / EVAL ==================== #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        labels = batch.y

        loss = weighted_masked_bce_loss(logits, labels, pos_weight)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("[WARNING] NaN loss detected, skipping batch")
            continue
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs

    return total_loss / max(n, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    pos_weight: torch.Tensor,
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
        return {"auc": float("nan"), "bce": float("nan"), "bce_unweighted": float("nan")}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    auc = compute_multitask_auc(logits, labels)
    bce_weighted = weighted_masked_bce_loss(logits, labels, pos_weight.cpu()).item()
    bce_unweighted = unweighted_masked_bce(logits, labels)

    return {"auc": auc, "bce": bce_weighted, "bce_unweighted": bce_unweighted}


# ==================== RUN SINGLE SEED ==================== #

def run_single_seed(
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()
    print(f"\n[INFO] Using device: {device}")

    train_loader, val_loader, test_loader, pos_weight = load_tox21_dataloaders(
        batch_size=cfg["batch_size"],
        seed=seed,
    )
    pos_weight = pos_weight.to(device)

    sample_batch = next(iter(train_loader))
    num_node_features = sample_batch.x.size(-1)
    out_dim = sample_batch.y.size(-1) if sample_batch.y.dim() > 1 else 1
    
    print(f"[INFO] Node features: {num_node_features}, Output dim: {out_dim}")

    model = build_model(
        model_name=model_name,
        num_node_features=num_node_features,
        out_dim=out_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg["lr"] * 0.01,
    )

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    epoch_history: List[Dict[str, float]] = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pos_weight=pos_weight,
            device=device,
        )
        
        scheduler.step()

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            pos_weight=pos_weight,
            device=device,
        )
        val_auc = val_metrics["auc"]
        val_bce = val_metrics["bce"]
        
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{model_name} | seed {seed} | epoch {epoch:03d}] "
                f"loss={train_loss:.4f} val_auc={val_auc:.4f} lr={current_lr:.6f}"
            )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(val_auc),
            "val_bce": float(val_bce),
            "lr": current_lr,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg["patience"]:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        pos_weight=pos_weight,
        device=device,
    )

    print(
        f"\n[{model_name} | seed {seed}] "
        f"TEST: auc={test_metrics['auc']:.4f}, "
        f"bce_weighted={test_metrics['bce']:.4f}, "
        f"bce_unweighted={test_metrics['bce_unweighted']:.4f}"
    )

    return {
        "auc": float(test_metrics["auc"]),
        "bce": float(test_metrics["bce"]),
        "bce_unweighted": float(test_metrics["bce_unweighted"]),
        "best_val_auc": float(best_val_auc),
        "test_metrics": test_metrics,
        "epoch_history": epoch_history,
    }


# ==================== AGGREGATION ==================== #

def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, List[float]] = {}
    for metrics in results:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(v)

    agg: Dict[str, Dict[str, float]] = {}
    for k, vs in all_metrics.items():
        arr = np.asarray(vs, dtype=float)
        agg[k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=0)),
        }
    return agg


# ==================== MAIN ==================== #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="IMPROVED Tox21 experiments with GCN/GIN/GAT."
    )
    parser.add_argument("--models", type=str, default="gcn,gin,gat")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/tox21_improved")

    # TUNED hyperparameters for TOX21 (12-task learning)
    parser.add_argument("--batch_size", type=int, default=32)      # Smaller batch
    parser.add_argument("--hidden_dim", type=int, default=256)     # ↑ More capacity for 12 tasks
    parser.add_argument("--num_layers", type=int, default=5)       # Deeper
    parser.add_argument("--dropout", type=float, default=0.2)      # ↓ Reduce underfitting
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)         # Train longer!
    parser.add_argument("--patience", type=int, default=50)        # More patience

    args = parser.parse_args()

    cfg = {
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    
    print("="*60)
    print("IMPROVED TOX21 GNN TRAINING")
    print("="*60)
    print(f"Config: {json.dumps(cfg, indent=2)}")
    print("="*60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    all_results: Dict[str, Any] = {}
    epoch_rows: List[Dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} on Tox21 ({args.seeds} seeds)")
        print(f"{'='*60}")
        
        seed_results: List[Dict[str, float]] = []

        for i in range(args.seeds):
            seed = 42 + i
            res = run_single_seed(cfg, model_name, seed)

            seed_results.append({
                "auc": res["auc"],
                "bce": res["bce"],
                "bce_unweighted": res["bce_unweighted"],
            })

            for ep in res["epoch_history"]:
                epoch_rows.append({
                    "model": model_name.upper(),
                    "seed": seed,
                    **ep,
                })

        agg = aggregate_results(seed_results)
        all_results[model_name] = {"per_seed": seed_results, "agg": agg}

        # Save per-model results
        model_out = output_dir / f"{model_name}_tox21_improved_results.json"
        with open(model_out, "w") as f:
            json.dump(all_results[model_name], f, indent=2)

        print(f"\n>>> {model_name.upper()} TOX21 IMPROVED: mean ± std <<<")
        for metric_name, stats in agg.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Combined summary
    summary_path = output_dir / "tox21_improved_all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Excel export
    df_epochs = pd.DataFrame(epoch_rows)
    
    summary_rows = []
    for model_name, res in all_results.items():
        agg = res["agg"]
        summary_rows.append({
            "model": model_name.upper(),
            "auc_mean": agg.get("auc", {}).get("mean", float("nan")),
            "auc_std": agg.get("auc", {}).get("std", float("nan")),
            "bce_mean": agg.get("bce", {}).get("mean", float("nan")),
            "bce_std": agg.get("bce", {}).get("std", float("nan")),
        })
    df_summary = pd.DataFrame(summary_rows)

    excel_path = output_dir / "tox21_improved_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_epochs.to_excel(writer, sheet_name="epochs", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

