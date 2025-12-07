# #!/usr/bin/env python
#
# """
# Complete BBBP experiments with GCN, GIN, and GAT (binary classification).
#
# This script:
#   - Loads BBBP from MoleculeNet (PyTorch Geometric)
#   - Uses scaffold-based train/val/test splits via src.data.scaffold_split.scaffold_split
#   - Trains Simple GCN / GIN / GAT classification models
#   - Handles missing labels (-1 / NaN) via masked BCE
#   - Computes ROC-AUC for BBBP (single-task classification)
#   - Runs multiple seeds per model
#   - Saves:
#       * Per-model JSON (per_seed metrics + aggregate)
#       * Combined JSON summary
#       * Excel with:
#           - 'epochs' sheet: all epochs for all seeds/models
#           - 'summary' sheet: mean ± std AUC/BCE per model
# """
#
# import os
# import sys
# import json
# import random
# from pathlib import Path
# from typing import Dict, Any, List, Tuple
#
# import numpy as np
# import pandas as pd
#
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import MoleculeNet
# from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool
# from sklearn.metrics import roc_auc_score
#
# # -------------------------------------------------------------------
# # Make project root importable
# # -------------------------------------------------------------------
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
#
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
#
# # Your scaffold split (expects list of SMILES)
# from src.data.scaffold_split import scaffold_split
#
#
# # ==================== UTILITIES ==================== #
#
# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# def get_device() -> torch.device:
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ==================== SIMPLE LOCAL MODELS ==================== #
#
# class SimpleGCN(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_dim: int,
#         out_dim: int,
#         num_layers: int,
#         dropout: float,
#         task: str,
#     ):
#         super().__init__()
#         self.task = task
#
#         self.convs = nn.ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_dim))
#         for _ in range(num_layers - 1):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
#
#         self.dropout = nn.Dropout(dropout)
#         self.head_weight = nn.Parameter(torch.empty(out_dim, hidden_dim))
#         self.head_bias = nn.Parameter(torch.zeros(out_dim))
#         nn.init.xavier_uniform_(self.head_weight)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = x.to(torch.float32)
#
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = torch.relu(x)
#             x = self.dropout(x)
#
#         x = global_add_pool(x, batch)
#         out = torch.matmul(x, self.head_weight.t()) + self.head_bias
#         return out
#
#
# class SimpleGIN(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_dim: int,
#         out_dim: int,
#         num_layers: int,
#         dropout: float,
#         task: str,
#     ):
#         super().__init__()
#         self.task = task
#
#         def mlp_block(in_dim, out_dim_):
#             return nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, out_dim_),
#             )
#
#         self.convs = nn.ModuleList()
#         self.convs.append(GINConv(mlp_block(in_channels, hidden_dim)))
#         for _ in range(num_layers - 1):
#             self.convs.append(GINConv(mlp_block(hidden_dim, hidden_dim)))
#
#         self.dropout = nn.Dropout(dropout)
#         self.head_weight = nn.Parameter(torch.empty(out_dim, hidden_dim))
#         self.head_bias = nn.Parameter(torch.zeros(out_dim))
#         nn.init.xavier_uniform_(self.head_weight)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = x.to(torch.float32)
#
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = torch.relu(x)
#             x = self.dropout(x)
#
#         x = global_add_pool(x, batch)
#         out = torch.matmul(x, self.head_weight.t()) + self.head_bias
#         return out
#
#
# class SimpleGAT(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_dim: int,
#         out_dim: int,
#         num_layers: int,
#         dropout: float,
#         task: str,
#         heads: int = 4,
#     ):
#         super().__init__()
#         self.task = task
#
#         self.convs = nn.ModuleList()
#         self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))
#         in_dim = hidden_dim * heads
#         for _ in range(num_layers - 1):
#             self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
#             in_dim = hidden_dim * heads
#
#         self.dropout = nn.Dropout(dropout)
#         self.head_weight = nn.Parameter(torch.empty(out_dim, in_dim))
#         self.head_bias = nn.Parameter(torch.zeros(out_dim))
#         nn.init.xavier_uniform_(self.head_weight)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = x.to(torch.float32)
#
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = torch.relu(x)
#             x = self.dropout(x)
#
#         x = global_add_pool(x, batch)
#         out = torch.matmul(x, self.head_weight.t()) + self.head_bias
#         return out
#
#
# def build_model(
#     model_name: str,
#     num_node_features: int,
#     out_dim: int,
#     hidden_dim: int,
#     num_layers: int,
#     dropout: float,
# ) -> nn.Module:
#     model_name = model_name.lower()
#     task = "classification"  # BBBP
#
#     if model_name == "gcn":
#         return SimpleGCN(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
#     elif model_name == "gin":
#         return SimpleGIN(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
#     elif model_name == "gat":
#         return SimpleGAT(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
#
#
# # ==================== DATA LOADING ==================== #
#
# def load_bbbp_dataloaders(
#     batch_size: int,
#     seed: int,
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Load BBBP via MoleculeNet and split using scaffold_split(smiles_list,...).
#     """
#     root = Path("data/moleculenet")
#     dataset = MoleculeNet(root=str(root), name="BBBP")
#
#     smiles_list: List[str] = []
#     for data in dataset:
#         smi = getattr(data, "smiles", None)
#         if smi is None:
#             raise ValueError("Data object missing `smiles` attribute.")
#         smiles_list.append(smi)
#
#     train_idx, val_idx, test_idx = scaffold_split(
#         smiles_list,
#         seed=seed,
#         frac_train=0.8,
#         frac_val=0.1,
#         frac_test=0.1,
#     )
#
#     train_dataset = dataset[train_idx]
#     val_dataset = dataset[val_idx]
#     test_dataset = dataset[test_idx]
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, val_loader, test_loader
#
#
# # ==================== LOSS & METRICS ==================== #
#
# def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     """
#     Binary cross-entropy with logits, masking out missing labels.
#     For BBBP, it's single-task, but we still support masking -1 / NaN just in case.
#     """
#     labels = labels.to(torch.float32)
#     if labels.dim() == 1:
#         labels = labels.view(-1, 1)
#     if logits.dim() == 1:
#         logits = logits.view(-1, 1)
#
#     mask = ~torch.isnan(labels) & (labels != -1)
#     if mask.sum() == 0:
#         return torch.tensor(0.0, device=logits.device, requires_grad=True)
#
#     logits = logits[mask]
#     labels = labels[mask]
#     return F.binary_cross_entropy_with_logits(logits, labels)
#
#
# def compute_auc_binary(logits: torch.Tensor, labels: torch.Tensor) -> float:
#     """
#     Compute ROC-AUC for a single-task binary classification.
#     Ignores missing labels (-1 / NaN).
#     """
#     labels = labels.to(torch.float32)
#     if labels.dim() == 1:
#         labels = labels.view(-1, 1)
#     if logits.dim() == 1:
#         logits = logits.view(-1, 1)
#
#     y = labels.cpu().numpy()
#     y_pred = torch.sigmoid(logits).cpu().numpy()
#
#     y_t = y[:, 0]
#     p_t = y_pred[:, 0]
#
#     mask = ~np.isnan(y_t) & (y_t != -1)
#     y_t = y_t[mask]
#     p_t = p_t[mask]
#
#     if y_t.size == 0 or len(np.unique(y_t)) < 2:
#         return float("nan")
#
#     return float(roc_auc_score(y_t, p_t))
#
#
# # ==================== TRAIN / EVAL ==================== #
#
# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
# ) -> float:
#     model.train()
#     total_loss = 0.0
#     n = 0
#
#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#
#         logits = model(batch)
#         labels = batch.y
#
#         loss = masked_bce_loss(logits, labels)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item() * batch.num_graphs
#         n += batch.num_graphs
#
#     return total_loss / max(n, 1)
#
#
# def evaluate(
#     model: nn.Module,
#     loader: DataLoader,
#     device: torch.device,
# ) -> Dict[str, float]:
#     model.eval()
#     all_logits = []
#     all_labels = []
#
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             logits = model(batch)
#             labels = batch.y
#
#             if labels.dim() == 1:
#                 labels = labels.view(-1, 1)
#             if logits.dim() == 1:
#                 logits = logits.view(-1, 1)
#
#             all_logits.append(logits.cpu())
#             all_labels.append(labels.cpu())
#
#     if not all_logits:
#         return {"auc": float("nan"), "bce": float("nan")}
#
#     logits = torch.cat(all_logits, dim=0)
#     labels = torch.cat(all_labels, dim=0)
#
#     auc = compute_auc_binary(logits, labels)
#     bce = masked_bce_loss(logits, labels).item()
#
#     return {"auc": auc, "bce": bce}
#
#
# # ==================== AGGREGATION ==================== #
#
# def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
#     all_metrics: Dict[str, List[float]] = {}
#     for metrics in results:
#         for k, v in metrics.items():
#             all_metrics.setdefault(k, []).append(v)
#
#     agg: Dict[str, Dict[str, float]] = {}
#     for k, vs in all_metrics.items():
#         arr = np.asarray(vs, dtype=float)
#         agg[k] = {
#             "mean": float(arr.mean()),
#             "std": float(arr.std(ddof=0)),
#         }
#     return agg
#
#
# # ==================== RUN SINGLE SEED ==================== #
#
# def run_single_seed(
#     cfg: Dict[str, Any],
#     model_name: str,
#     seed: int,
# ) -> Dict[str, Any]:
#     set_seed(seed)
#     device = get_device()
#
#     train_loader, val_loader, test_loader = load_bbbp_dataloaders(
#         batch_size=cfg["batch_size"],
#         seed=seed,
#     )
#
#     sample_batch = next(iter(train_loader))
#     num_node_features = sample_batch.x.size(-1)
#     out_dim = sample_batch.y.size(-1) if sample_batch.y.dim() > 1 else 1
#
#     model = build_model(
#         model_name=model_name,
#         num_node_features=num_node_features,
#         out_dim=out_dim,
#         hidden_dim=cfg["hidden_dim"],
#         num_layers=cfg["num_layers"],
#         dropout=cfg["dropout"],
#     ).to(device)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=cfg["lr"],
#         weight_decay=cfg["weight_decay"],
#     )
#
#     best_val_auc = -1.0
#     best_state = None
#
#     epoch_history: List[Dict[str, float]] = []
#
#     for epoch in range(1, cfg["epochs"] + 1):
#         train_loss = train_one_epoch(
#             model=model,
#             loader=train_loader,
#             optimizer=optimizer,
#             device=device,
#         )
#
#         val_metrics = evaluate(
#             model=model,
#             loader=val_loader,
#             device=device,
#         )
#         val_auc = val_metrics["auc"]
#         val_bce = val_metrics["bce"]
#
#         print(
#             f"[{model_name} | seed {seed} | epoch {epoch}] "
#             f"train_loss={train_loss:.4f} val_auc={val_auc:.4f}"
#         )
#
#         epoch_history.append(
#             {
#                 "epoch": epoch,
#                 "train_loss": float(train_loss),
#                 "val_auc": float(val_auc),
#                 "val_bce": float(val_bce),
#             }
#         )
#
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             best_state = model.state_dict()
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     test_metrics = evaluate(
#         model=model,
#         loader=test_loader,
#         device=device,
#     )
#
#     print(
#         f"[{model_name} | seed {seed}] "
#         f"TEST: auc={test_metrics['auc']:.4f}, bce={test_metrics['bce']:.4f}"
#     )
#
#     result: Dict[str, Any] = {
#         "auc": float(test_metrics["auc"]),
#         "bce": float(test_metrics["bce"]),
#         "test_metrics": {k: float(v) for k, v in test_metrics.items()},
#         "epoch_history": epoch_history,
#     }
#     return result
#
#
# # ==================== MAIN ==================== #
#
# def main():
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="BBBP experiments with Simple GCN/GIN/GAT."
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
#         default="results/bbbp",
#         help="Directory to store JSON/Excel results.",
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
#     all_results: Dict[str, Any] = {}
#
#     epoch_rows: List[Dict[str, Any]] = []
#
#     for model_name in model_names:
#         print(f"\n===== {model_name.upper()} on BBBP ({args.seeds} seeds) =====")
#         seed_results: List[Dict[str, float]] = []
#
#         for i in range(args.seeds):
#             seed = 42 + i
#             res = run_single_seed(cfg, model_name, seed)
#
#             seed_results.append(
#                 {
#                     "auc": res["auc"],
#                     "bce": res["bce"],
#                 }
#             )
#
#             for ep in res["epoch_history"]:
#                 epoch_rows.append(
#                     {
#                         "model": model_name.upper(),
#                         "seed": seed,
#                         "epoch": ep["epoch"],
#                         "train_loss": ep["train_loss"],
#                         "val_auc": ep["val_auc"],
#                         "val_bce": ep["val_bce"],
#                     }
#                 )
#
#         agg = aggregate_results(seed_results)
#         all_results[model_name] = {"per_seed": seed_results, "agg": agg}
#
#         model_out = output_dir / f"{model_name}_bbbp_results.json"
#         with open(model_out, "w") as f:
#             json.dump(all_results[model_name], f, indent=2)
#
#         print(f"\n>>> {model_name.upper()} BBBP: mean ± std AUC/BCE <<<")
#         for metric_name, stats in agg.items():
#             print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
#
#     # Combined JSON summary
#     summary_path = output_dir / "bbbp_all_models_summary.json"
#     with open(summary_path, "w") as f:
#         json.dump(all_results, f, indent=2)
#
#     # === Build DataFrames for Excel ===
#     df_epochs = pd.DataFrame(epoch_rows)
#
#     summary_rows = []
#     for model_name, res in all_results.items():
#         agg = res["agg"]
#         auc = agg.get("auc", {})
#         bce = agg.get("bce", {})
#         summary_rows.append(
#             {
#                 "model": model_name.upper(),
#                 "auc_mean": auc.get("mean", float("nan")),
#                 "auc_std": auc.get("std", float("nan")),
#                 "bce_mean": bce.get("mean", float("nan")),
#                 "bce_std": bce.get("std", float("nan")),
#             }
#         )
#     df_summary = pd.DataFrame(summary_rows)
#     df_summary = df_summary[["model", "auc_mean", "auc_std", "bce_mean", "bce_std"]]
#
#     excel_path = output_dir / "bbbp_results_detailed.xlsx"
#     with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
#         df_epochs.to_excel(writer, sheet_name="epochs", index=False)
#         df_summary.to_excel(writer, sheet_name="summary", index=False)
#
#     print(f"Detailed Results for bbbp are exported to {excel_path}")
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python

"""
Complete BBBP experiments with GCN, GIN, and GAT (binary classification).

This script:
  - Loads BBBP from MoleculeNet (PyTorch Geometric)
  - Uses scaffold-based train/val/test splits via src.data.scaffold_split.scaffold_split
  - If scaffold split produces single-class val/test, falls back to random
    stratified split (so AUC is well-defined)
  - Trains Simple GCN / GIN / GAT classification models
  - Handles missing labels (-1 / NaN) via masked BCE
  - Computes ROC-AUC for BBBP (single-task classification)
  - Runs multiple seeds per model
  - Saves:
      * Per-model JSON (per_seed metrics + aggregate)
      * Combined JSON summary
      * Excel with:
          - 'epochs' sheet: all epochs for all seeds/models
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
from sklearn.model_selection import train_test_split

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

        x = global_add_pool(x, batch)
        out = torch.matmul(x, self.head_weight.t()) + self.head_bias
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
    task = "classification"  # BBBP

    if model_name == "gcn":
        return SimpleGCN(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
    elif model_name == "gin":
        return SimpleGIN(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
    elif model_name == "gat":
        return SimpleGAT(num_node_features, hidden_dim, out_dim, num_layers, dropout, task)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== SPLIT HELPERS ==================== #

def _get_label_counts(dataset) -> Tuple[int, int]:
    """Return (num_negatives, num_positives) for a PyG dataset."""
    ys = []
    for d in dataset:
        y = d.y.view(-1)
        ys.append(y)
    ys = torch.cat(ys).cpu()
    num_pos = int((ys == 1).sum().item())
    num_neg = int((ys == 0).sum().item())
    return num_neg, num_pos


def _stratified_random_split(dataset, seed, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    """
    Fallback split: random stratified split on labels (0/1) when scaffold split
    produces degenerate label distributions (single-class val/test).
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    ys = torch.cat([d.y.view(-1) for d in dataset]).cpu().numpy()
    idx = np.arange(len(dataset))

    # Train vs (val+test)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        ys,
        test_size=(1.0 - frac_train),
        stratify=ys,
        random_state=seed,
    )

    # Temp -> val vs test
    frac_test_rel = frac_test / (frac_val + frac_test)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=frac_test_rel,
        stratify=y_temp,
        random_state=seed,
    )

    return list(idx_train), list(idx_val), list(idx_test)


# ==================== DATA LOADING ==================== #

def load_bbbp_dataloaders(
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load BBBP via MoleculeNet and split using scaffold_split(smiles_list,...).
    If scaffold split produces val/test sets with only one class, fall back
    to random stratified split so AUC is defined.
    """
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="BBBP")

    # 1) Try scaffold split
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

    train_neg, train_pos = _get_label_counts(train_dataset)
    val_neg, val_pos = _get_label_counts(val_dataset)
    test_neg, test_pos = _get_label_counts(test_dataset)

    print(
        f"[BBBP | seed {seed}] scaffold split -> "
        f"train: neg={train_neg}, pos={train_pos} | "
        f"val: neg={val_neg}, pos={val_pos} | "
        f"test: neg={test_neg}, pos={test_pos}"
    )

    # 2) If val or test has only one class, fall back to random stratified split
    if (val_neg == 0 or val_pos == 0 or test_neg == 0 or test_pos == 0):
        print(
            "[BBBP] WARNING: scaffold split produced single-class val/test. "
            "Falling back to random stratified split."
        )
        train_idx, val_idx, test_idx = _stratified_random_split(
            dataset,
            seed=seed,
            frac_train=0.8,
            frac_val=0.1,
            frac_test=0.1,
        )
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        train_neg, train_pos = _get_label_counts(train_dataset)
        val_neg, val_pos = _get_label_counts(val_dataset)
        test_neg, test_pos = _get_label_counts(test_dataset)
        print(
            f"[BBBP | seed {seed}] random stratified split -> "
            f"train: neg={train_neg}, pos={train_pos} | "
            f"val: neg={val_neg}, pos={val_pos} | "
            f"test: neg={test_neg}, pos={test_pos}"
        )

    # 3) Build loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ==================== LOSS & METRICS ==================== #

def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy with logits, masking out missing labels.
    For BBBP, it's single-task, but we still support masking -1 / NaN.
    """
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)

    mask = ~torch.isnan(labels) & (labels != -1)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits = logits[mask]
    labels = labels[mask]
    return F.binary_cross_entropy_with_logits(logits, labels)


def compute_auc_binary(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute ROC-AUC for a single-task binary classification.
    Ignores missing labels (-1 / NaN).
    """
    labels = labels.to(torch.float32)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if logits.dim() == 1:
        logits = logits.view(-1, 1)

    y = labels.cpu().numpy()
    y_pred = torch.sigmoid(logits).cpu().numpy()

    y_t = y[:, 0]
    p_t = y_pred[:, 0]

    mask = ~np.isnan(y_t) & (y_t != -1)
    y_t = y_t[mask]
    p_t = p_t[mask]

    if y_t.size == 0 or len(np.unique(y_t)) < 2:
        return float("nan")

    return float(roc_auc_score(y_t, p_t))


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

        logits = model(batch)
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

    auc = compute_auc_binary(logits, labels)
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

    train_loader, val_loader, test_loader = load_bbbp_dataloaders(
        batch_size=cfg["batch_size"],
        seed=seed,
    )

    sample_batch = next(iter(train_loader))
    num_node_features = sample_batch.x.size(-1)
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

        if not np.isnan(val_auc) and val_auc > best_val_auc:
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
        description="BBBP experiments with Simple GCN/GIN/GAT."
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
        default="results/bbbp",
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
        print(f"\n===== {model_name.upper()} on BBBP ({args.seeds} seeds) =====")
        seed_results: List[Dict[str, float]] = []

        for i in range(args.seeds):
            seed = 42 + i
            res = run_single_seed(cfg, model_name, seed)

            seed_results.append(
                {
                    "auc": res["auc"],
                    "bce": res["bce"],
                }
            )

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

        model_out = output_dir / f"{model_name}_bbbp_results.json"
        with open(model_out, "w") as f:
            json.dump(all_results[model_name], f, indent=2)

        print(f"\n>>> {model_name.upper()} BBBP: mean ± std AUC/BCE <<<")
        for metric_name, stats in agg.items():
            print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Combined JSON summary
    summary_path = output_dir / "bbbp_all_models_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # === Build DataFrames for Excel ===
    df_epochs = pd.DataFrame(epoch_rows)

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

    excel_path = output_dir / "bbbp_results_detailed.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_epochs.to_excel(writer, sheet_name="epochs", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"[OK] Wrote detailed BBBP Excel results to {excel_path}")


if __name__ == "__main__":
    main()

