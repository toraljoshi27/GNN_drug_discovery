#!/usr/bin/env python3
"""
Preprocess raw CSVs (Tox21, BBBP, ESOL) into PyG Data objects.

Input (must exist):
  data/raw/tox21.csv
  data/raw/BBBP.csv
  data/raw/delaney-processed.csv

Output:
  data/processed/tox21/dataset.pt
  data/processed/bbbp/dataset.pt
  data/processed/esol/dataset.pt
"""

import os
import sys
import csv
import json
import math
import argparse
import pathlib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ---- RDKit (required to turn SMILES -> graph) --------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except Exception as e:
    print(
        "\n[ERROR] RDKit is not installed. This script needs RDKit to parse SMILES into graphs.\n"
        "Install RDKit with conda (recommended):\n"
        "  conda install -c conda-forge rdkit=2024.03.5 -y\n"
        "Or skip scaffold-based workflows and use random splits until RDKit is available.\n"
    )
    raise

RAW_DIR = pathlib.Path("data/raw").resolve()
OUT_DIR = pathlib.Path("data/processed").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- Atom / Bond Featurization --------------------------

_ATOMIC_NUMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H,C,N,O,F,P,S,Cl,Br,I (common)
_HYB_TYPES = [
    rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2
]

def one_hot(value, choices):
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass
    return vec

def atom_features(atom: rdchem.Atom) -> List[float]:
    z = atom.GetAtomicNum()
    feats = []
    # atomic number (coarse one-hot over common elements + 'other')
    feats += one_hot(z, _ATOMIC_NUMS) + [0]  # placeholder for 'other'
    if z not in _ATOMIC_NUMS:
        feats[-1] = 1

    # degree (0..5+)
    deg = min(atom.GetTotalDegree(), 5)
    feats += one_hot(deg, [0,1,2,3,4,5])

    # formal charge (-2..2 clipped)
    chg = int(np.clip(atom.GetFormalCharge(), -2, 2))
    feats += one_hot(chg, [-2,-1,0,1,2])

    # hybridization
    hyb = atom.GetHybridization()
    feats += one_hot(hyb, _HYB_TYPES)

    # aromatic, in ring, chirality
    feats += [1 if atom.GetIsAromatic() else 0]
    feats += [1 if atom.IsInRing() else 0]
    feats += one_hot(atom.GetChiralTag(), [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
    ])

    # implicit/explicit valence (clipped)
    feats += [min(atom.GetImplicitValence(), 5), min(atom.GetExplicitValence(), 5)]
    return feats

def bond_features(bond: rdchem.Bond) -> List[float]:
    if bond is None:
        # self-loop pseudo-bond
        return [1,0,0,0, 0,0, 0,0,0]  # SINGLE on self-loop, rest zeros

    btype = bond.GetBondType()
    feat = one_hot(btype, [
        rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC
    ])
    feat += [1 if bond.GetIsConjugated() else 0]
    feat += [1 if bond.IsInRing() else 0]
    # stereo (E/Z/none-ish)
    feat += one_hot(bond.GetStereo(), [
        rdchem.BondStereo.STEREONONE,
        rdchem.BondStereo.STEREOZ,
        rdchem.BondStereo.STEREOE
    ])
    return feat

def smiles_to_data(smiles: str, y: Optional[np.ndarray]) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # nodes
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # edges (undirected)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.append([a, b]); edge_attr.append(bf)
        edge_index.append([b, a]); edge_attr.append(bf)

    # add self-loops with simple features (optional but common)
    for i in range(mol.GetNumAtoms()):
        edge_index.append([i, i])
        edge_attr.append(bond_features(None))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float)
    data.smiles = smiles
    return data

# ------------------------------ Dataset Parsers --------------------------------

def load_tox21_csv() -> Tuple[List[str], np.ndarray]:
    """Return (smiles_list, labels_matrix[ n, 12 ]) with NaN where missing."""
    path = RAW_DIR / "tox21.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_datasets.py first.")
    df = pd.read_csv(path)
    # canonical Tox21 columns usually include these 12 tasks:
    # NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD,
    # NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
    # but we’ll detect all non-smiles numeric columns as tasks to be robust.
    smiles_col = None
    for cand in ["smiles", "smile", "SMILES", "Smiles"]:
        if cand in df.columns:
            smiles_col = cand; break
    if smiles_col is None:
        raise RuntimeError("Could not find SMILES column in tox21.csv")

    smiles = df[smiles_col].astype(str).tolist()
    # choose numeric columns (float/int) excluding ID-like strings
    task_cols = [c for c in df.columns if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])]
    Y = df[task_cols].to_numpy(dtype=float)
    # keep NaN as is (masked BCE later)
    return smiles, Y

def load_bbbp_csv() -> Tuple[List[str], np.ndarray]:
    path = RAW_DIR / "BBBP.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_datasets.py first.")
    df = pd.read_csv(path)
    # Common columns: 'smiles' and 'p_np' (P=permeable, N=non-permeable)
    smiles_col = None
    for cand in ["smiles", "SMILES", "Smiles"]:
        if cand in df.columns:
            smiles_col = cand; break
    if smiles_col is None:
        raise RuntimeError("Could not find SMILES column in BBBP.csv")
    # pick the first non-smiles column as label if not obvious
    label_col = None
    for cand in ["p_np", "bbbp", "BBBP", "label", "Label"]:
        if cand in df.columns:
            label_col = cand; break
    if label_col is None:
        label_col = [c for c in df.columns if c != smiles_col][0]

    def map_label(v):
        if pd.isna(v): return np.nan
        if isinstance(v, (int, float)): return 1.0 if float(v) >= 0.5 else 0.0
        s = str(v).strip().lower()
        if s in {"p", "yes", "y", "true", "1"}: return 1.0
        if s in {"n", "no", "false", "0"}: return 0.0
        try:
            return 1.0 if float(s) >= 0.5 else 0.0
        except:
            return np.nan

    smiles = df[smiles_col].astype(str).tolist()
    y = np.array([map_label(v) for v in df[label_col].values], dtype=float).reshape(-1, 1)
    return smiles, y

def load_esol_csv() -> Tuple[List[str], np.ndarray]:
    path = RAW_DIR / "delaney-processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_datasets.py first.")
    df = pd.read_csv(path)
    smiles_col = None
    for cand in ["smiles", "SMILES", "smile", "Smiles"]:
        if cand in df.columns:
            smiles_col = cand; break
    if smiles_col is None:
        raise RuntimeError("Could not find SMILES column in delaney-processed.csv")
    # common label column names in this file:
    label_cands = [
        "measured log solubility in mols per litre",
        "measured logS", "LogS", "logS", "solubility"
    ]
    label_col = None
    for cand in label_cands:
        if cand in df.columns:
            label_col = cand; break
    if label_col is None:
        # fallback: first numeric column that isn't smiles
        numeric_cols = [c for c in df.columns if c != smiles_col and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise RuntimeError("Could not identify ESOL label column.")
        label_col = numeric_cols[0]

    smiles = df[smiles_col].astype(str).tolist()
    y = df[label_col].astype(float).to_numpy().reshape(-1, 1)
    return smiles, y

# ---------------------------- Build and Save Datasets --------------------------

def build_dataset(name: str, smiles: List[str], Y: Optional[np.ndarray]):
    data_list = []
    dropped = 0
    for i, smi in enumerate(smiles):
        y = Y[i] if Y is not None else None
        d = smiles_to_data(smi, y)
        if d is None:
            dropped += 1
            continue
        data_list.append(d)

    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.pt"
    meta = {
        "name": name,
        "num_graphs": len(data_list),
        "dropped": int(dropped)
    }
    torch.save({"data_list": data_list, "meta": meta}, out_path)
    print(f"✅ Saved {len(data_list)} graphs to {out_path} (dropped {dropped} invalid SMILES).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["tox21","bbbp","esol"],
                    help="Subset of {tox21, bbbp, esol}")
    args = ap.parse_args()

    wanted = set([s.lower() for s in args.datasets])
    if "tox21" in wanted:
        smi, Y = load_tox21_csv()
        build_dataset("tox21", smi, Y)
    if "bbbp" in wanted:
        smi, Y = load_bbbp_csv()
        build_dataset("bbbp", smi, Y)
    if "esol" in wanted:
        smi, Y = load_esol_csv()
        build_dataset("esol", smi, Y)

if __name__ == "__main__":
    main()
