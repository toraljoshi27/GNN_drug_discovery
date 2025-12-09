#!/usr/bin/env python

"""
Hyperparameter sweep for a GAT on ESOL (scaffold split) + optional attention visualization.

What this script does:
  1. Loads ESOL with a scaffold-based split (train/val/test).
  2. Sweeps over different GAT hyperparameters:
       - hidden_dim
       - dropout
       - heads
       - multiple random seeds
  3. For each config + seed:
       - trains GAT on train/val
       - evaluates on test set
       - logs RMSE and MAE
  4. Aggregates mean ± std per hyperparameter config.
  5. Prints a summary and picks the best config by mean RMSE.
  6. Exports:
       - per-run + summary metrics to an Excel file
       - a summary RMSE plot to a separate images folder
  7. Retrains the best config and generates attention-based molecule PNGs
     for a few ESOL test molecules under results/esol_attention_gat_diff_hyp/.
"""

import os
import sys
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool

import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

# -------------------------------------------------------------------
# Make project root importable
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.scaffold_split import scaffold_split  # your existing scaffold split


# ==================== Utils ==================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Data loading (scaffold split) ==================== #

def load_esol_scaffold_split(split_seed: int):
    """
    Load ESOL from MoleculeNet and create scaffold-based train/val/test splits.
    Returns:
      full_dataset : full PyG dataset
      train_ds     : train subset
      val_ds       : val subset
      test_ds      : test subset
    """
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="ESOL")

    smiles_list = [d.smiles for d in dataset]

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        seed=split_seed,
        frac_train=0.8,
        frac_val=0.1,
        frac_test=0.1,
    )

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]
    test_ds = dataset[test_idx]

    print(f"[ESOL] scaffold split (seed={split_seed}) "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return dataset, train_ds, val_ds, test_ds


# ==================== GAT model with attention ==================== #

class AttentiveGAT(nn.Module):
    """
    GAT model for regression on ESOL, with attention exposed on the last layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)
        self.heads = heads

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))
        in_dim = hidden_dim * heads

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
            in_dim = hidden_dim * heads

        self.head = nn.Linear(in_dim, out_dim)

        # For attention visualization
        self.last_edge_index = None  # [2, E]
        self.last_att_weights = None  # [E, H]

    def forward(self, data, return_attention: bool = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)

        # All but last GAT layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout_layer(x)

        # Last GAT layer with attention weights
        last_conv: GATConv = self.convs[-1]
        out, (edge_index_out, att_weights) = last_conv(
            x,
            edge_index,
            return_attention_weights=True,
        )
        out = torch.relu(out)
        out = self.dropout_layer(out)

        # Store attention info
        self.last_edge_index = edge_index_out.detach()
        self.last_att_weights = att_weights.detach()  # [E, heads]

        # Pool and regress
        pooled = global_add_pool(out, batch)
        logits = self.head(pooled)  # [num_graphs, 1]

        if return_attention:
            return logits, self.last_edge_index, self.last_att_weights
        return logits


# ==================== Training + Evaluation ==================== #

def train_gat_esol(
    train_ds,
    val_ds,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.2,
    heads: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    epochs: int = 40,
    seed: int = 42,
) -> AttentiveGAT:
    """
    Train GAT on ESOL (regression) and return the best model by val MSE.
    """
    set_seed(seed)
    device = get_device()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    sample = next(iter(train_loader))
    in_channels = sample.x.size(-1)
    out_dim = 1  # ESOL: single continuous target

    model = AttentiveGAT(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch).view(-1)
            labels = batch.y.view(-1).to(torch.float32)
            loss = F.mse_loss(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch).view(-1)
                labels = batch.y.view(-1).to(torch.float32)
                loss = F.mse_loss(preds, labels)
                val_loss += loss.item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)

        print(
            f"[ESOL GAT] epoch {epoch:3d} | "
            f"train_mse={avg_train_loss:.4f} | val_mse={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.to(device)


def eval_esol(
    model: AttentiveGAT,
    test_ds,
) -> Dict[str, float]:
    """
    Evaluate GAT on ESOL test set: return RMSE and MAE.
    """
    device = get_device()
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch).view(-1)
            labels = batch.y.view(-1).to(torch.float32)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if len(all_preds) == 0:
        return {"rmse": float("nan"), "mae": float("nan")}

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
    mae = float(np.abs(preds - labels).mean())
    return {"rmse": rmse, "mae": mae}


# ==================== Attention → atom scores & drawing ==================== #

def edge_attention_to_atom_weights(
    edge_index: torch.Tensor,
    att_weights: torch.Tensor,
    num_nodes: int,
) -> np.ndarray:
    """
    Aggregate edge attention weights into per-atom scores:
      - Average attention across heads per edge
      - Add edge attention to both endpoint atoms
      - Normalize to [0,1]
    """
    edge_index_np = edge_index.cpu().numpy()      # [2, E]
    att_np = att_weights.mean(dim=1).cpu().numpy()  # [E]

    atom_scores = np.zeros(num_nodes, dtype=np.float32)
    for e in range(edge_index_np.shape[1]):
        src = int(edge_index_np[0, e])
        dst = int(edge_index_np[1, e])
        w = float(att_np[e])
        atom_scores[src] += w
        atom_scores[dst] += w

    if atom_scores.max() > 0:
        atom_scores = atom_scores / atom_scores.max()
    return atom_scores


def draw_molecule_with_atom_attention(
    smiles: str,
    atom_weights: np.ndarray,
    out_path: Path,
    size: Tuple[int, int] = (400, 300),
):
    """
    Draw a molecule with atoms colored based on attention weights.
    Higher weight → more red.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[WARN] Could not parse SMILES: {smiles}")
        return

    num_atoms = mol.GetNumAtoms()
    highlight_atoms = list(range(num_atoms))
    highlight_colors = {}

    for idx in highlight_atoms:
        w = float(atom_weights[idx]) if idx < len(atom_weights) else 0.0
        # red channel = 1.0, green/blue fade with weight (more red = higher attention)
        highlight_colors[idx] = (1.0, 1.0 - w, 1.0 - w)

    img = Draw.MolToImage(
        mol,
        size=size,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    img.save(str(out_path))
    print(f"[OK] Saved attention visualization: {out_path}")


# ==================== MAIN ==================== #

def main():
    # ---- Global settings ----
    base_split_seed = 123
    set_seed(base_split_seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Load ESOL once
    full_dataset, train_ds, val_ds, test_ds = load_esol_scaffold_split(base_split_seed)

    # Hyperparameter grid
    hidden_dims = [64, 128, 256]
    dropouts = [0.1, 0.3, 0.5]
    heads_list = [4, 8]
    num_layers = 3
    seeds = [0, 1, 2]

    config_metrics: Dict[Tuple[int, float, int], Dict[str, List[float]]] = {}
    per_run_rows: List[Dict[str, float]] = []      # each run (config + seed)
    summary_rows: List[Dict[str, float]] = []      # per-config summary

    # ---- Hyperparameter sweep ----
    for hidden_dim in hidden_dims:
        for dropout in dropouts:
            for heads in heads_list:
                config_key = (hidden_dim, dropout, heads)
                config_metrics[config_key] = {"rmse": [], "mae": []}

                for seed in seeds:
                    print(
                        f"\n[RUN ESOL] hidden_dim={hidden_dim}, "
                        f"dropout={dropout}, heads={heads}, seed={seed}"
                    )

                    model = train_gat_esol(
                        train_ds=train_ds,
                        val_ds=val_ds,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        heads=heads,
                        seed=seed,
                    )

                    metrics = eval_esol(model, test_ds)
                    rmse = metrics["rmse"]
                    mae = metrics["mae"]

                    print(
                        f"[RESULT ESOL] h={hidden_dim}, d={dropout}, heads={heads}, "
                        f"seed={seed} | RMSE={rmse:.4f}, MAE={mae:.4f}"
                    )

                    config_metrics[config_key]["rmse"].append(rmse)
                    config_metrics[config_key]["mae"].append(mae)

                    per_run_rows.append({
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "heads": heads,
                        "seed": seed,
                        "rmse": rmse,
                        "mae": mae,
                    })

    # ---- Aggregate results per config ----
    print("\n==================== ESOL Hyperparameter Summary ====================")
    best_config = None
    best_rmse_mean = float("inf")

    for (hidden_dim, dropout, heads), metrics in config_metrics.items():
        rmse_arr = np.array(metrics["rmse"], dtype=np.float32)
        mae_arr = np.array(metrics["mae"], dtype=np.float32)

        rmse_mean = float(rmse_arr.mean())
        rmse_std = float(rmse_arr.std())
        mae_mean = float(mae_arr.mean())
        mae_std = float(mae_arr.std())

        print(
            f"GAT(h={hidden_dim}, d={dropout}, heads={heads}) | "
            f"RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}, "
            f"MAE = {mae_mean:.4f} ± {mae_std:.4f}"
        )

        summary_rows.append({
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "heads": heads,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
        })

        if rmse_mean < best_rmse_mean:
            best_rmse_mean = rmse_mean
            best_config = (hidden_dim, dropout, heads)

    print("\n==================== Best ESOL Config ====================")
    if best_config is not None:
        h_best, d_best, heads_best = best_config
        print(
            f"Best GAT config for ESOL (by mean RMSE): "
            f"hidden_dim={h_best}, dropout={d_best}, heads={heads_best}, "
            f"RMSE_mean={best_rmse_mean:.4f}"
        )
    else:
        print("No valid ESOL config found.")

    # ---------------- Save results to Excel ----------------
    results_dir = Path("results/esol_gat")
    results_dir.mkdir(parents=True, exist_ok=True)

    df_runs = pd.DataFrame(per_run_rows)
    df_summary = pd.DataFrame(summary_rows)

    excel_path = results_dir / "esol_gat_hyp_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_runs.to_excel(writer, sheet_name="per_run", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"[INFO] Saved Excel results to: {excel_path}")

    # ---------------- Save summary RMSE plot ----------------
    images_dir = Path("results/esol_gat_images")
    images_dir.mkdir(parents=True, exist_ok=True)

    if len(summary_rows) > 0:
        xs = np.arange(len(summary_rows))
        rmse_means = [row["rmse_mean"] for row in summary_rows]
        labels = [
            f"h={row['hidden_dim']},d={row['dropout']},heads={row['heads']}"
            for row in summary_rows
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(summary_rows) * 0.6), 5))
        ax.bar(xs, rmse_means)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("RMSE (mean)")
        ax.set_title("ESOL GAT Hyperparameter RMSE (mean)")
        fig.tight_layout()

        img_path = images_dir / "esol_gat_rmse_summary.png"
        fig.savefig(img_path, dpi=200)
        plt.close(fig)

        print(f"[INFO] Saved RMSE summary plot to: {img_path}")

    # ---------------- Retrain best config and generate attention images ----------------
    if best_config is not None:
        print("\n[INFO] Retraining best ESOL config for attention visualization...")
        attn_dir = Path("results/esol_attention_gat_diff_hyp")
        attn_dir.mkdir(parents=True, exist_ok=True)

        best_model = train_gat_esol(
            train_ds=train_ds,
            val_ds=val_ds,
            hidden_dim=h_best,
            num_layers=num_layers,
            dropout=d_best,
            heads=heads_best,
            seed=42,
        ).to(device)
        best_model.eval()

        num_to_visualize = 5
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        print(f"[ESOL GAT attn] Visualizing attention for {num_to_visualize} test molecules...")

        count = 0
        for i, batch in enumerate(test_loader):
            if count >= num_to_visualize:
                break

            batch = batch.to(device)

            with torch.no_grad():
                preds, edge_index, att_weights = best_model(batch, return_attention=True)

            num_nodes = batch.x.size(0)
            atom_scores = edge_attention_to_atom_weights(edge_index, att_weights, num_nodes)

            data_obj = test_ds[i]
            smiles = data_obj.smiles

            out_path = attn_dir / f"esol_gat_attn_example_{i}.png"
            draw_molecule_with_atom_attention(smiles, atom_scores, out_path)

            count += 1

        print("[ESOL GAT attn] Done.")


if __name__ == "__main__":
    main()
