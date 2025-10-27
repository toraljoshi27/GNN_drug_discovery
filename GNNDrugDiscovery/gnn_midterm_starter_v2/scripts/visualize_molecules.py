# # scripts/visualize_molecules.py
# import argparse
# from pathlib import Path
# import warnings
# import re
# import yaml
#
# # drawing
# from rdkit import Chem
# from rdkit.Chem import Draw
#
# # deepchem loaders
# try:
#     import deepchem as dc
# except Exception as e:
#     raise RuntimeError(
#         "DeepChem is required for this script. Try:\n"
#         "  conda install -c conda-forge deepchem rdkit\n"
#         "or  pip install deepchem rdkit-pypi"
#     ) from e
#
#
# def load_config(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)
#
#
# def infer_dataset_name(cfg_path: str, cfg: dict | None) -> str:
#     """
#     Try to infer dataset name from config file name or YAML.
#     Supports: esol, bbbp, tox21
#     """
#     # 1) from filename, e.g., exp_esol_gin.yaml
#     fname = Path(cfg_path).name.lower()
#     for key in ("esol", "bbbp", "tox21"):
#         if key in fname:
#             return key
#
#     # 2) from YAML (try common places)
#     if cfg:
#         # e.g. cfg["dataset"]["name"] or cfg["data"]["name"]
#         for k1 in ("dataset", "data"):
#             if k1 in cfg and isinstance(cfg[k1], dict):
#                 for k2 in ("name", "dataset", "id"):
#                     val = cfg[k1].get(k2)
#                     if isinstance(val, str):
#                         val_l = val.lower()
#                         for key in ("esol", "bbbp", "tox21"):
#                             if key in val_l:
#                                 return key
#     raise ValueError(
#         "Could not infer dataset name from config path or YAML. "
#         "Please ensure your config filename contains one of: esol, bbbp, tox21."
#     )
#
#
# def load_molnet_split(name: str):
#     """
#     Return (train_ds, valid_ds, test_ds, task_info)
#     where each ds is a DeepChem dataset exposing .ids as SMILES/IDs.
#     """
#     name = name.lower()
#     if name == "esol":
#         tasks, (train, valid, test), transformers = dc.molnet.load_delaney(featurizer="GraphConv")
#         task_info = {"type": "regression"}
#         return train, valid, test, task_info
#     elif name == "bbbp":
#         tasks, (train, valid, test), transformers = dc.molnet.load_bbbp(featurizer="GraphConv")
#         task_info = {"type": "binary"}
#         return train, valid, test, task_info
#     elif name == "tox21":
#         tasks, (train, valid, test), transformers = dc.molnet.load_tox21(featurizer="GraphConv")
#         task_info = {"type": "multilabel", "num_tasks": len(tasks), "task_names": tasks}
#         return train, valid, test, task_info
#     else:
#         raise ValueError(f"Unsupported dataset: {name}")
#
#
# def pick_split(train, valid, test, split: str):
#     s = split.lower()
#     if s in ("val", "valid"):
#         return valid, "val"
#     elif s == "test":
#         return test, "test"
#     else:
#         raise ValueError(f"Unsupported split: {split} (use 'val' or 'test')")
#
#
# def extract_smiles(dataset, index: int) -> str | None:
#     """
#     DeepChem DiskDataset has `ids` which are often SMILES or identifiers.
#     """
#     # 1) ids list
#     try:
#         smi = dataset.ids[index]
#         if isinstance(smi, str):
#             return smi
#     except Exception:
#         pass
#
#     # 2) as fallback, try to reconstruct from X if it looks like RDKit Mol (rare here)
#     try:
#         item = dataset.X[index]
#         # no standard way to recover SMILES here; just fail gracefully
#     except Exception:
#         pass
#     return None
#
#
# def to_mol(smiles: str):
#     try:
#         m = Chem.MolFromSmiles(smiles)
#         return m
#     except Exception:
#         return None
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--ckpt", required=True)  # unused here but kept for CLI symmetry
#     ap.add_argument("--split", default="test", choices=["val", "valid", "test"])
#     ap.add_argument("--index", type=int, default=0)
#     ap.add_argument("--out", default=None, help="Optional output path; else runs/<exp>/molecules/")
#     args = ap.parse_args()
#
#     cfg = load_config(args.config)
#     dataset_name = infer_dataset_name(args.config, cfg)
#
#     # Load MolNet dataset
#     train, valid, test, task_info = load_molnet_split(dataset_name)
#     ds, split_name = pick_split(train, valid, test, args.split)
#
#     # Resolve output directory near the checkpoint
#     ckpt_path = Path(args.ckpt).resolve()
#     run_dir = ckpt_path.parent
#     out_dir = Path(args.out) if args.out else (run_dir / "molecules")
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # Pull SMILES
#     smiles = extract_smiles(ds, args.index)
#     if smiles is None:
#         warnings.warn(
#             "Could not find SMILES for this item from DeepChem ids. "
#             "Rendering a placeholder ethanol (CCO)."
#         )
#         smiles = "CCO"
#
#     mol = to_mol(smiles)
#     if mol is None:
#         warnings.warn(f"RDKit failed on SMILES={smiles!r}. Using placeholder ethanol.")
#         mol = Chem.MolFromSmiles("CCO")
#
#     # Draw and save
#     img = Draw.MolToImage(mol, size=(360, 360))
#     # Build a readable file name like esol_test_idx3.png
#     base = f"{dataset_name}_{split_name}_idx{args.index}.png"
#     out_path = out_dir / base
#     img.save(out_path)
#
#     print(f"✅ Saved molecule image at: {out_path.resolve()}")
#     print(f"ℹ️ Dataset: {dataset_name} | Split: {split_name} | Index: {args.index} | SMILES: {smiles}")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_molecules.py

Save molecule images to a unified folder structure:

Molecules/<dataset>/<model>/
  - <dataset>_<split>_idx<k>.png  (individual images)
  - <dataset>_<split>_grid.png    (grid if multiple)

Supports datasets: esol, bbbp, tox21 (via DeepChem MolNet).
Infers dataset + model names from --config or --ckpt file names.
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import Draw

# DeepChem for MoleculeNet datasets
import deepchem as dc


# ---------- Helpers ----------
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_dataset_and_model(config_path: str | None, ckpt_path: str | None):
    """
    Infer dataset in {esol, bbbp, tox21} and model in {gin, gat, gcn}
    from either the config filename or the checkpoint path.
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path).name.lower())
    if ckpt_path:
        # also include parent folder name like runs/esol_gin/best.pt
        p = Path(ckpt_path)
        candidates.append(p.name.lower())
        candidates.append(p.parent.name.lower())

    dataset = None
    model = None
    for s in candidates:
        if dataset is None:
            m = re.search(r"(esol|bbbp|tox21)", s)
            if m:
                dataset = m.group(1)
        if model is None:
            m = re.search(r"(gin|gat|gcn)", s)
            if m:
                model = m.group(1)
        if dataset and model:
            break

    if dataset is None:
        raise ValueError("Could not infer dataset (expected esol/bbbp/tox21) from --config/--ckpt names.")
    if model is None:
        # default to 'gin' if not found, but warn
        warnings.warn("Could not infer model (gin/gat/gcn) from names; defaulting to 'gin'.")
        model = "gin"
    return dataset, model


def load_molnet(dataset: str):
    """Return (train, valid, test)."""
    dataset = dataset.lower()
    if dataset == "esol":
        tasks, (train, valid, test), _ = dc.molnet.load_delaney(featurizer="GraphConv")
    elif dataset == "bbbp":
        tasks, (train, valid, test), _ = dc.molnet.load_bbbp(featurizer="GraphConv")
    elif dataset == "tox21":
        tasks, (train, valid, test), _ = dc.molnet.load_tox21(featurizer="GraphConv")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train, valid, test


def pick_split(train, valid, test, split: str):
    s = split.lower()
    if s in ("val", "valid"):
        return valid, "val"
    if s == "test":
        return test, "test"
    raise ValueError("Unsupported --split (use 'val' or 'test').")


def extract_smiles(ds, idx: int) -> str | None:
    # DeepChem DiskDataset usually exposes .ids = SMILES (for MoleculeNet)
    try:
        smi = ds.ids[idx]
        return smi if isinstance(smi, str) else None
    except Exception:
        return None


def safe_mol(smiles: str):
    if not smiles:
        return Chem.MolFromSmiles("CCO")
    m = Chem.MolFromSmiles(smiles)
    return m if m is not None else Chem.MolFromSmiles("CCO")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False, help="Config file (used to infer dataset/model).")
    ap.add_argument("--ckpt", required=False, help="Checkpoint path (used to infer dataset/model).")
    ap.add_argument("--dataset", choices=["esol", "bbbp", "tox21"], help="Override dataset name.")
    ap.add_argument("--model", choices=["gin", "gat", "gcn"], help="Override model name.")
    ap.add_argument("--split", default="test", choices=["val", "valid", "test"])
    # selection:
    ap.add_argument("--index", type=int, default=None, help="Single index.")
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="Multiple explicit indices.")
    ap.add_argument("--n", type=int, default=None, help="Take first N (default 16 if none provided).")
    ap.add_argument("--random", action="store_true", help="Randomly sample N molecules.")
    ap.add_argument("--seed", type=int, default=42)
    # grid/output:
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=300, help="Side (px) for individual PNGs.")
    args = ap.parse_args()

    # Resolve dataset/model
    if args.dataset and args.model:
        dataset, model = args.dataset, args.model
    else:
        dataset, model = infer_dataset_and_model(args.config, args.ckpt)

    # Load dataset split
    train, valid, test = load_molnet(dataset)
    ds, split_name = pick_split(train, valid, test, args.split)
    N = len(ds.y) if hasattr(ds, "y") else len(ds.X)

    # Decide indices
    rng = np.random.default_rng(args.seed)
    if args.indices:
        idxs = [i for i in args.indices if 0 <= i < N]
    elif args.index is not None:
        idxs = [args.index] if 0 <= args.index < N else []
    else:
        k = args.n if args.n is not None else 16
        k = min(k, N)
        if args.random:
            idxs = rng.choice(N, size=k, replace=False).tolist()
        else:
            idxs = list(range(k))
    if not idxs:
        raise ValueError("No valid indices selected. Use --index, --indices, or --n (optionally with --random).")

    # Output directory: Molecules/<dataset>/<model>/
    out_dir = Path("Molecules") / dataset / model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render individual images
    mols, legends = [], []
    for i in idxs:
        smi = extract_smiles(ds, i)
        if smi is None:
            warnings.warn(f"No SMILES at index {i}; using ethanol 'CCO'.")
            smi = "CCO"
        mol = safe_mol(smi)

        legends.append(f"idx={i}")
        mols.append(mol)

        img = Draw.MolToImage(mol, size=(args.img_size, args.img_size))
        fname = f"{dataset}_{split_name}_idx{i}.png"
        img.save(out_dir / fname)

    # Save a grid if multiple
    if len(mols) > 1:
        grid = Draw.MolsToGridImage(
            mols,
            molsPerRow=max(1, args.grid_cols),
            subImgSize=(min(260, args.img_size), min(260, args.img_size)),
            legends=legends,
        )
        grid.save(out_dir / f"{dataset}_{split_name}_grid.png")

    print(f"✅ Saved {len(mols)} image(s) to: {out_dir.resolve()}")
    if len(mols) > 1:
        print(f"🖼️ Grid: {(out_dir / f'{dataset}_{split_name}_grid.png').resolve()}")


if __name__ == "__main__":
    main()
