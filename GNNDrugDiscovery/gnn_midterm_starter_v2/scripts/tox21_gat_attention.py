#!/usr/bin/env python

"""
Train a GAT on Tox21 (scaffold split) and visualize attention
weights on a few test molecules.

Outputs:
  - PNG files in results/tox21_attention/ with atoms colored by attention.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool

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

def load_tox21_scaffold_split(seed: int):
    """
    Load Tox21 from MoleculeNet and create scaffold-based train/val/test splits.
    Returns:
      dataset  : full PyG dataset (for SMILES access)
      train_ds : train subset
      val_ds   : val subset
      test_ds  : test subset
    """
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="Tox21")

    smiles_list = [d.smiles for d in dataset]

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        seed=seed,
        frac_train=0.8,
        frac_val=0.1,
        frac_test=0.1,
    )

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]
    test_ds = dataset[test_idx]

    print(f"[Tox21] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return dataset, train_ds, val_ds, test_ds


# ==================== GAT model with attention ==================== #

class AttentiveGAT(nn.Module):
    """
    GAT model that exposes attention weights from the last GATConv layer.
    """

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.2, heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True))
        in_dim = hidden_dim * heads

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
            in_dim = hidden_dim * heads

        self.head = nn.Linear(in_dim, out_dim)

        # Attributes to store last-layer attention
        self.last_edge_index = None  # [2, E]
        self.last_att_weights = None  # [E, H]

    def forward(self, data, return_attention: bool = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)

        # All but last GAT layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        # Last GAT layer with attention weights
        last_conv: GATConv = self.convs[-1]
        out, (edge_index_out, att_weights) = last_conv(
            x,
            edge_index,
            return_attention_weights=True,
        )
        out = torch.relu(out)
        out = self.dropout(out)

        # Store attention info
        self.last_edge_index = edge_index_out.detach()
        self.last_att_weights = att_weights.detach()  # [E, heads]

        # Pool and classify
        pooled = global_add_pool(out, batch)
        logits = self.head(pooled)  # [num_graphs, out_dim]

        if return_attention:
            return logits, self.last_edge_index, self.last_att_weights
        return logits


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Multi-task BCE with masking for missing labels (-1 or NaN).
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


# ==================== Training ==================== #

def train_gat_for_attention(
    train_ds,
    val_ds,
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.2,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    epochs: int = 40,
    seed: int = 42,
) -> AttentiveGAT:
    """
    Train AttentiveGAT on Tox21 train/val and return the best model (by val loss).
    """
    set_seed(seed)
    device = get_device()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    sample = next(iter(train_loader))
    in_channels = sample.x.size(-1)
    out_dim = sample.y.size(-1) if sample.y.dim() > 1 else 1

    model = AttentiveGAT(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = masked_bce_loss(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = masked_bce_loss(logits, batch.y)
                val_loss += loss.item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)

        print(f"[GAT attn] epoch {epoch:3d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.to(device)


# ==================== Attention → atom scores ==================== #

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
    # edge_index: [2, E], att_weights: [E, H]
    edge_index_np = edge_index.cpu().numpy()
    att_np = att_weights.mean(dim=1).cpu().numpy()  # average over heads, shape [E]

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
    seed = 42
    set_seed(seed)
    device = get_device()

    # Output directory
    out_dir = Path("results/tox21_attention")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data & split
    full_dataset, train_ds, val_ds, test_ds = load_tox21_scaffold_split(seed)

    # Train GAT with attention
    model = train_gat_for_attention(train_ds, val_ds, seed=seed)
    model.eval()

    # Visualize attention on a few test molecules
    num_to_visualize = 5
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"[Tox21 GAT attn] Visualizing attention for {num_to_visualize} test molecules...")

    count = 0
    for i, batch in enumerate(test_loader):
        if count >= num_to_visualize:
            break

        batch = batch.to(device)

        # Forward pass to get attention
        with torch.no_grad():
            logits, edge_index, att_weights = model(batch, return_attention=True)

        num_nodes = batch.x.size(0)
        atom_scores = edge_attention_to_atom_weights(edge_index, att_weights, num_nodes)

        # Get SMILES for this molecule from test_ds
        data_obj = test_ds[i]
        smiles = data_obj.smiles

        out_path = out_dir / f"tox21_gat_attn_example_{i}.png"
        draw_molecule_with_atom_attention(smiles, atom_scores, out_path)

        count += 1

    print("[Tox21 GAT attn] Done.")


if __name__ == "__main__":
    main()
