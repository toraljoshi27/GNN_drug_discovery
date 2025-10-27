
import os
from typing import Tuple
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from .scaffold_split import scaffold_split

def load_dataset(name: str, split: str = "scaffold", batch_size: int = 128, num_workers: int = 0, seed: int = 42):
    name_map = {"tox21": "Tox21", "bbbp": "BBBP", "esol": "ESOL"}
    assert name.lower() in name_map, f"Unsupported dataset: {name}"
    pyg_name = name_map[name.lower()]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moleculenet"))
    ds = MoleculeNet(root=root, name=pyg_name)
    # Extract SMILES for scaffold split
    smiles = [d.smiles if hasattr(d, "smiles") else "" for d in ds]
    if split == "scaffold":
        tr_idx, va_idx, te_idx = scaffold_split(smiles, seed=seed)
    elif split == "random":
        n = len(ds)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        n_tr, n_va = int(0.8*n), int(0.1*n)
        tr_idx, va_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]
        tr_idx, va_idx, te_idx = sorted(tr_idx), sorted(va_idx), sorted(te_idx)
    else:
        raise ValueError(f"Unknown split: {split}")
    # Index into the dataset
    train_dataset = [ds[i] for i in tr_idx]
    val_dataset   = [ds[i] for i in va_idx]
    test_dataset  = [ds[i] for i in te_idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Determine task information
    if pyg_name == "Tox21":
        task = "multilabel"; out_dim = 12
    elif pyg_name == "BBBP":
        task = "binary"; out_dim = 1
    else:
        task = "regression"; out_dim = 1
    num_node_features = ds.num_node_features
    num_edge_features = getattr(ds[0], "edge_attr", None)
    num_edge_features = num_edge_features.size(-1) if num_edge_features is not None else 0
    return {
        "task": task,
        "out_dim": out_dim,
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx,
    }
