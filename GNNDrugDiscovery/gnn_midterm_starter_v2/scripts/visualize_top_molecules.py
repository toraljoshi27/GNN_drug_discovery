#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Top-N Molecules by prediction-based criteria with robust CSV column detection.

Outputs:
  results/<dataset>/figs/
    - molecule_<mode>_<dataset>_<model>_<split>_grid.png
    - molecule_<mode>_<dataset>_<model>_<split>_idx<k>.png

Datasets: esol (regression), bbbp (binary), tox21 (multi-label)
Selection modes:
  Regression: top_abs_error, top_residual_pos, top_residual_neg
  Classification: top_correct, top_incorrect, top_confident_correct, top_confident_incorrect, top_prob_pos, top_prob_neg

Robust column handling:
  - Auto-detects columns (case-insensitive; trims spaces; normalizes punctuation)
  - Override with --y_true_col, --y_pred_col, --prob_col, --index_col, --id_col
  - If y_pred is absent but prob exists, can derive y_pred from prob with --derive_pred_if_missing
  - If row indices aren’t present but an ID (e.g., SMILES/PMID) is, map IDs to dataset indices

Examples:
  python scripts/visualize_top_molecules.py \
    --config configs/exp_bbbp_gat.yaml \
    --ckpt runs/bbbp_gat/best.pt \
    --split test \
    --pred_csv results/bbbp/metrics_bbbp_gat_test.csv \
    --mode top_confident_correct \
    --prob_col prob \
    --derive_pred_if_missing \
    --threshold 0.5 \
    --n 16 \
    --debug

  python scripts/visualize_top_molecules.py \
    --config configs/exp_tox21_gcn.yaml \
    --ckpt runs/tox21_gcn/best.pt \
    --split test \
    --pred_csv results/tox21/metrics_tox21_gcn_test.csv \
    --task NR-AR \
    --mode top_prob_pos \
    --prob_col prob_NR-AR \
    --n 16

  python scripts/visualize_top_molecules.py \
    --config configs/exp_esol_gin.yaml \
    --ckpt runs/esol_gin/best.pt \
    --split test \
    --pred_csv results/esol/metrics_esol_gin_test.csv \
    --mode top_abs_error \
    --y_true_col y_true \
    --y_pred_col y_pred \
    --n 16
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Draw

# DeepChem MolNet loaders
import deepchem as dc


# -------------------------- Utils --------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_dataset_and_model(config_path: str | None, ckpt_path: str | None):
    """Infer dataset in {esol, bbbp, tox21} and model in {gin, gat, gcn} from names."""
    cand = []
    if config_path:
        cand.append(Path(config_path).name.lower())
    if ckpt_path:
        p = Path(ckpt_path)
        cand.extend([p.name.lower(), p.parent.name.lower()])

    dataset = None
    model = None
    for s in cand:
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
        raise ValueError("Could not infer dataset (esol/bbbp/tox21) from --config/--ckpt. Use --dataset/--model.")
    if model is None:
        warnings.warn("Could not infer model (gin/gat/gcn); defaulting to 'gin'.")
        model = "gin"
    return dataset, model


def load_molnet(dataset: str):
    d = dataset.lower()
    if d == "esol":
        tasks, (train, valid, test), _ = dc.molnet.load_delaney(featurizer="GraphConv")
        task_info = {"type": "regression"}
    elif d == "bbbp":
        tasks, (train, valid, test), _ = dc.molnet.load_bbbp(featurizer="GraphConv")
        task_info = {"type": "binary"}
    elif d == "tox21":
        tasks, (train, valid, test), _ = dc.molnet.load_tox21(featurizer="GraphConv")
        task_info = {"type": "multilabel", "tasks": tasks}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train, valid, test, task_info


def pick_split(train, valid, test, split: str):
    s = split.lower()
    if s in ("val", "valid"):
        return valid, "val"
    if s == "test":
        return test, "test"
    raise ValueError("Unsupported --split. Use 'val' or 'test'.")


def normalize_name(x: str) -> str:
    """Normalize a column name for robust matching: lower, strip, collapse spaces/underscores/hyphens."""
    x = x.strip().lower()
    x = re.sub(r"[\s\-]+", "_", x)
    return x


def mk_name_map(columns) -> tuple[dict, dict]:
    """
    Returns (norm_to_orig, orig_to_norm) dicts for a DataFrame's columns.
    norm_to_orig: normalized -> original
    orig_to_norm: original -> normalized
    """
    norm_to_orig = {}
    orig_to_norm = {}
    for c in columns:
        nc = normalize_name(str(c))
        orig_to_norm[c] = nc
        # First occurrence wins; keep original capitalization if duplicates normalize same
        if nc not in norm_to_orig:
            norm_to_orig[nc] = c
    return norm_to_orig, orig_to_norm


def autodetect(norm_to_orig: dict, candidates: list[str]) -> str | None:
    """Given normalized->original map and candidate names (already normalized or raw), return original name if found."""
    for cand in candidates:
        nc = normalize_name(cand)
        if nc in norm_to_orig:
            return norm_to_orig[nc]
    return None


def ensure_outdir(dataset: str, outdir: str | None) -> Path:
    out = Path(outdir) if outdir else Path("results") / dataset / "figs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def extract_smiles(ds, idx: int) -> str | None:
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


# -------------------- Column resolution --------------------
def resolve_columns(df: pd.DataFrame,
                    task: str | None,
                    index_col: str | None,
                    id_col: str | None,
                    y_true_col: str | None,
                    y_pred_col: str | None,
                    prob_col: str | None,
                    debug: bool = False):
    """Resolve column names with robust auto-detection and user overrides."""
    norm_to_orig, orig_to_norm = mk_name_map(df.columns)

    if debug:
        print("CSV columns:", list(df.columns))
        print("Normalized:", [orig_to_norm[c] for c in df.columns])

    # Index column — dataset integer row index
    if index_col:
        index_col = autodetect(norm_to_orig, [index_col])
    else:
        index_col = autodetect(norm_to_orig, [
            "index", "idx", "row", "data_index", "dataset_index", "i"
        ])

    # ID column — e.g., SMILES or PMID (used if index is not available)
    if id_col:
        id_col = autodetect(norm_to_orig, [id_col])
    else:
        id_col = autodetect(norm_to_orig, [
            "smiles", "smile", "id", "ids", "pmid", "molid", "molecule_id", "mol_id"
        ])

    # Task suffix for Tox21 if provided
    suffix = f"_{task}" if task else ""

    # y_true
    if y_true_col:
        y_true_col = autodetect(norm_to_orig, [y_true_col])
    else:
        y_true_col = autodetect(norm_to_orig, [
            f"y_true{suffix}", f"label{suffix}", f"labels{suffix}", f"target{suffix}",
            "y_true", "label", "labels", "target", "truth", "gt", "y"
        ])

    # y_pred
    if y_pred_col:
        y_pred_col = autodetect(norm_to_orig, [y_pred_col])
    else:
        y_pred_col = autodetect(norm_to_orig, [
            f"y_pred{suffix}", f"pred{suffix}", f"prediction{suffix}", f"pred_label{suffix}",
            "y_pred", "pred", "prediction", "pred_label", "preds"
        ])

    # prob
    if prob_col:
        prob_col = autodetect(norm_to_orig, [prob_col])
    else:
        prob_col = autodetect(norm_to_orig, [
            f"prob{suffix}", f"prob_pos{suffix}", f"p_pred{suffix}",
            f"y_score{suffix}", f"score{suffix}", f"proba{suffix}",
            "prob", "prob_pos", "p_pred", "y_score", "score", "proba", "probability"
        ])

    if debug:
        print("Resolved columns ->",
              f"index_col={index_col}, id_col={id_col}, y_true_col={y_true_col}, "
              f"y_pred_col={y_pred_col}, prob_col={prob_col}")

    return index_col, id_col, y_true_col, y_pred_col, prob_col


# -------------------- Selection logic --------------------
def select_top_indices(
    df: pd.DataFrame,
    dataset_type: str,
    mode: str,
    n: int,
    index_col: str | None,
    id_col: str | None,
    y_true_col: str | None,
    y_pred_col: str | None,
    prob_col: str | None,
    derive_pred_if_missing: bool,
    threshold: float,
    debug: bool = False,
):
    """Return a Series with two columns: 'ds_index' (int) and optional 'id' (str)."""
    if n <= 0:
        return pd.DataFrame(columns=["ds_index", "id"])

    d = df.copy()

    # If y_pred is missing for classification and we have prob + derive flag, create it
    if dataset_type in ("binary", "multilabel") and y_pred_col is None and prob_col is not None and derive_pred_if_missing:
        y_pred_col = "__y_pred__"
        d[y_pred_col] = (d[prob_col] >= threshold).astype(int)
        if debug:
            print(f"Derived y_pred_col '{y_pred_col}' from prob '{prob_col}' with threshold={threshold}")

    # Build selection
    if dataset_type == "regression":
        if y_true_col is None or y_pred_col is None:
            raise ValueError(f"[Regression] mode '{mode}' requires y_true & y_pred columns.")
        d["abs_error"] = (d[y_true_col] - d[y_pred_col]).abs()
        d["residual"] = d[y_pred_col] - d[y_true_col]

        if mode == "top_abs_error":
            sel = d.sort_values("abs_error", ascending=False).head(n)
        elif mode == "top_residual_pos":
            sel = d.sort_values("residual", ascending=False).head(n)
        elif mode == "top_residual_neg":
            sel = d.sort_values("residual", ascending=True).head(n)
        else:
            raise ValueError("Use one of: top_abs_error, top_residual_pos, top_residual_neg for regression.")

    elif dataset_type in ("binary", "multilabel"):
        if y_true_col is None:
            raise ValueError(f"[Classification] mode '{mode}' requires y_true column.")
        if "prob" in mode and prob_col is None:
            raise ValueError(f"[Classification] mode '{mode}' requires prob column.")

        # y_pred may still be None if using prob-only modes
        if y_pred_col is not None:
            d["correct"] = (d[y_true_col] == d[y_pred_col]).astype(int)

        if mode == "top_correct":
            if y_pred_col is None:
                raise ValueError("top_correct requires y_pred column (or use --derive_pred_if_missing).")
            sel = d[d["correct"] == 1].head(n)

        elif mode == "top_incorrect":
            if y_pred_col is None:
                raise ValueError("top_incorrect requires y_pred column (or use --derive_pred_if_missing).")
            sel = d[d["correct"] == 0].head(n)

        elif mode == "top_confident_correct":
            if y_pred_col is None:
                raise ValueError("top_confident_correct requires y_pred (or use --derive_pred_if_missing).")
            if prob_col is None:
                raise ValueError("top_confident_correct requires prob column.")
            # Confidence of predicted class
            d["conf"] = np.where(d[y_pred_col].astype(int) == 1, d[prob_col], 1.0 - d[prob_col])
            sel = d[d["correct"] == 1].sort_values("conf", ascending=False).head(n)

        elif mode == "top_confident_incorrect":
            if y_pred_col is None:
                raise ValueError("top_confident_incorrect requires y_pred (or use --derive_pred_if_missing).")
            if prob_col is None:
                raise ValueError("top_confident_incorrect requires prob column.")
            d["conf"] = np.where(d[y_pred_col].astype(int) == 1, d[prob_col], 1.0 - d[prob_col])
            sel = d[d["correct"] == 0].sort_values("conf", ascending=False).head(n)

        elif mode == "top_prob_pos":
            sel = d.sort_values(prob_col, ascending=False).head(n)

        elif mode == "top_prob_neg":
            sel = d.sort_values(prob_col, ascending=True).head(n)

        else:
            raise ValueError("Unsupported classification mode.")

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Build return: either ds_index or map from id -> ds_index later
    out = pd.DataFrame(index=sel.index)
    if index_col is not None and index_col in d.columns:
        out["ds_index"] = d.loc[sel.index, index_col].astype(int).values
        out["id"] = d.loc[sel.index, id_col].astype(str).values if id_col and id_col in d.columns else None
    elif id_col is not None and id_col in d.columns:
        out["ds_index"] = -1  # placeholder, resolve from id later
        out["id"] = d.loc[sel.index, id_col].astype(str).values
    else:
        # try use df index if it's integer-like
        if d.index.dtype.kind in ("i", "u"):
            out["ds_index"] = sel.index.astype(int)
            out["id"] = None
        else:
            raise ValueError(
                "Could not resolve dataset indices: no integer index, no index_col, and no id_col."
            )

    if debug:
        print("Selected rows (head):")
        print(sel.head(min(5, len(sel))))
        print("Selection mapping (head):")
        print(out.head(min(5, len(out))))

    return out.reset_index(drop=True)


def map_ids_to_dataset_indices(sel_df: pd.DataFrame, ds, debug: bool = False):
    """Map sel_df['id'] strings to dataset row indices using ds.ids; fill ds_index."""
    if "id" not in sel_df.columns or sel_df["id"].isnull().all():
        return sel_df  # nothing to map

    # Build map from ds.ids to position
    id_to_pos = {}
    try:
        for i, ident in enumerate(ds.ids):
            if isinstance(ident, str):
                id_to_pos[ident] = i
    except Exception:
        pass

    def _lookup(x: str) -> int:
        if x in id_to_pos:
            return id_to_pos[x]
        # Try loose matching for SMILES-like ids with possible spaces
        xs = x.strip()
        return id_to_pos.get(xs, -1)

    sel_df = sel_df.copy()
    mask_need = (sel_df["ds_index"] < 0) & sel_df["id"].notnull()
    if mask_need.any():
        sel_df.loc[mask_need, "ds_index"] = sel_df.loc[mask_need, "id"].map(_lookup).fillna(-1).astype(int)

    if debug:
        unresolved = sel_df[sel_df["ds_index"] < 0]
        if len(unresolved) > 0:
            print("Warning: some IDs could not be mapped to dataset indices:")
            print(unresolved[["id"]])

    # Drop any that could not be mapped
    sel_df = sel_df[sel_df["ds_index"] >= 0].reset_index(drop=True)
    return sel_df


# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Dataset/model inference
    ap.add_argument("--config", help="Config file (used to infer dataset/model).")
    ap.add_argument("--ckpt", help="Checkpoint path (used to infer dataset/model).")
    ap.add_argument("--dataset", choices=["esol", "bbbp", "tox21"], help="Override dataset.")
    ap.add_argument("--model", choices=["gin", "gat", "gcn"], help="Override model.")
    ap.add_argument("--split", default="test", choices=["val", "valid", "test"])

    # Predictions CSV & columns
    ap.add_argument("--pred_csv", required=True, help="CSV with per-molecule predictions.")
    ap.add_argument("--index_col", help="Column with dataset integer row index.")
    ap.add_argument("--id_col", help="Column with molecule ID (e.g., SMILES/PMID) if index not present.")
    ap.add_argument("--y_true_col", help="Ground-truth label column.")
    ap.add_argument("--y_pred_col", help="Predicted label column.")
    ap.add_argument("--prob_col", help="Probability of class 1 (classification).")
    ap.add_argument("--task", help="Tox21 task name (e.g., NR-AR) to pick suffixed columns.")

    # Selection
    ap.add_argument("--mode", required=True,
                    choices=[
                        # regression
                        "top_abs_error", "top_residual_pos", "top_residual_neg",
                        # classification
                        "top_correct", "top_incorrect",
                        "top_confident_correct", "top_confident_incorrect",
                        "top_prob_pos", "top_prob_neg",
                    ])
    ap.add_argument("--n", type=int, default=16)

    # Derive y_pred
    ap.add_argument("--derive_pred_if_missing", action="store_true",
                    help="If set and y_pred missing, derive y_pred = (prob >= threshold).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for deriving y_pred.")

    # Output / rendering
    ap.add_argument("--outdir", default=None, help="Output dir (default: results/<dataset>/figs)")
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=300)

    # Debug
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # Resolve dataset/model
    if args.dataset and args.model:
        dataset, model = args.dataset, args.model
    else:
        dataset, model = infer_dataset_and_model(args.config, args.ckpt)

    # Load dataset split
    train, valid, test, task_info = load_molnet(dataset)
    ds, split_name = pick_split(train, valid, test, args.split)
    dataset_type = task_info["type"]

    # Load CSV
    df = pd.read_csv(args.pred_csv)

    # Resolve columns robustly
    index_col, id_col, y_true_col, y_pred_col, prob_col = resolve_columns(
        df=df,
        task=args.task,
        index_col=args.index_col,
        id_col=args.id_col,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        prob_col=args.prob_col,
        debug=args.debug,
    )

    # Select top rows
    sel = select_top_indices(
        df=df,
        dataset_type=dataset_type,
        mode=args.mode,
        n=args.n,
        index_col=index_col,
        id_col=id_col,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        prob_col=prob_col,
        derive_pred_if_missing=args.derive_pred_if_missing,
        threshold=args.threshold,
        debug=args.debug,
    )

    # Map IDs to dataset indices if needed
    sel = map_ids_to_dataset_indices(sel, ds, debug=args.debug)
    if sel.empty:
        raise RuntimeError("No selectable molecules found. Check your CSV columns and mapping to dataset.")

    # Prepare output dir
    out_dir = ensure_outdir(dataset, args.outdir)

    # Render images
    mols, legends = [], []
    for ds_idx in sel["ds_index"].astype(int).tolist():
        smi = extract_smiles(ds, ds_idx)
        if smi is None:
            warnings.warn(f"No SMILES at dataset index {ds_idx}; using 'CCO'.")
            smi = "CCO"
        mol = safe_mol(smi)
        mols.append(mol)
        legends.append(f"idx={ds_idx}")
        fname = f"molecule_{args.mode}_{dataset}_{model}_{split_name}_idx{ds_idx}.png"
        Draw.MolToImage(mol, size=(args.img_size, args.img_size)).save(out_dir / fname)

    grid_name = f"molecule_{args.mode}_{dataset}_{model}_{split_name}_grid.png"
    grid = Draw.MolsToGridImage(
        mols,
        molsPerRow=max(1, args.grid_cols),
        subImgSize=(min(260, args.img_size), min(260, args.img_size)),
        legends=legends,
    )
    grid.save(out_dir / grid_name)

    print("✅ Done.")
    print(f"Saved {len(mols)} molecules and grid to: {out_dir.resolve()}")
    print(f"Grid file: {(out_dir / grid_name).resolve()}")
    if dataset_type == "multilabel" and not args.task:
        print("ℹ️ Tox21 is multi-label. For per-task selection, pass --task (e.g., --task NR-AR).")


if __name__ == "__main__":
    main()
