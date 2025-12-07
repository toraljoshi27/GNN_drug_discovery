#!/usr/bin/env python

"""
Run classical baselines (RF + MLP on ECFP) for ESOL, Tox21, and BBBP
using the SAME splits as the GNN experiments (scaffold + BBBP fallback),
and append results into the combined summary Excel.

Output:
  results/combined/all_datasets_summary_with_baselines.xlsx
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import torch

# -------------------------------------------------------------------
# Make project root importable for scaffold_split
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.scaffold_split import scaffold_split  # your existing function


# ==================== UTILS ==================== #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def smiles_to_ecfp(smiles_list: List[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """
    Convert a list of SMILES into ECFP fingerprints (Morgan).
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # fallback all-zero vector
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        # convert RDKit ExplicitBitVect to numpy array
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(bv, arr)
        fps.append(arr.astype(np.float32))
    return np.stack(fps, axis=0)


def scaffold_split_indices(smiles_list: List[str], seed: int,
                           frac_train=0.8, frac_val=0.1, frac_test=0.1):
    """
    Wrapper around your scaffold_split that just returns indices.
    Note: your scaffold_split already ignores seed (deterministic),
    but we pass it for interface consistency.
    """
    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        seed=seed,
        frac_train=frac_train,
        frac_val=frac_val,
        frac_test=frac_test,
    )
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def stratified_random_split_labels(labels: np.ndarray, seed: int,
                                   frac_train=0.8, frac_val=0.1, frac_test=0.1):
    """
    For BBBP fallback: stratified random split on binary labels (0/1).
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    idx = np.arange(len(labels))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        labels,
        test_size=(1.0 - frac_train),
        stratify=labels,
        random_state=seed,
    )

    frac_test_rel = frac_test / (frac_val + frac_test)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=frac_test_rel,
        stratify=y_temp,
        random_state=seed,
    )

    return idx_train, idx_val, idx_test


# ==================== DATA LOADERS (SKLEARN-FRIENDLY) ==================== #

def load_esol_splits(seed: int):
    from torch_geometric.datasets import MoleculeNet
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="ESOL")

    smiles = [d.smiles for d in dataset]
    y = np.array([float(d.y.item()) for d in dataset], dtype=np.float32)

    train_idx, val_idx, test_idx = scaffold_split_indices(smiles, seed)

    X = smiles_to_ecfp(smiles)
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def load_tox21_splits(seed: int):
    from torch_geometric.datasets import MoleculeNet
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="Tox21")

    smiles = [d.smiles for d in dataset]
    # y: [N, T], may contain -1 / NaN
    ys = []
    for d in dataset:
        y = d.y.view(-1).numpy().astype(np.float32)
        ys.append(y)
    y = np.stack(ys, axis=0)   # [N, T]

    train_idx, val_idx, test_idx = scaffold_split_indices(smiles, seed)

    X = smiles_to_ecfp(smiles)
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def load_bbbp_splits(seed: int):
    from torch_geometric.datasets import MoleculeNet
    root = Path("data/moleculenet")
    dataset = MoleculeNet(root=str(root), name="BBBP")

    smiles = [d.smiles for d in dataset]
    ys = np.array([float(d.y.item()) for d in dataset], dtype=np.float32)

    # try scaffold split first
    train_idx, val_idx, test_idx = scaffold_split_indices(smiles, seed)

    def class_counts(indices):
        y_sub = ys[indices]
        return np.sum(y_sub == 0), np.sum(y_sub == 1)

    val_neg, val_pos = class_counts(val_idx)
    test_neg, test_pos = class_counts(test_idx)

    print(f"[BBBP baseline | seed {seed}] scaffold split label counts "
          f"val: neg={val_neg}, pos={val_pos} | "
          f"test: neg={test_neg}, pos={test_pos}")

    # fallback to stratified random if val/test single-class
    if (val_neg == 0 or val_pos == 0 or test_neg == 0 or test_pos == 0):
        print("[BBBP baseline] WARNING: scaffold split degenerate; "
              "falling back to stratified random split.")
        train_idx, val_idx, test_idx = stratified_random_split_labels(ys, seed)

    X = smiles_to_ecfp(smiles)
    return (X[train_idx], ys[train_idx],
            X[val_idx], ys[val_idx],
            X[test_idx], ys[test_idx])


# ==================== METRICS HELPERS ==================== #

def esol_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def tox21_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    y_true, y_pred: [N, T]; y_true may contain -1 or NaN.
    Computes macro AUC across tasks and BCE (log_loss) with masking.
    """
    T = y_true.shape[1]
    aucs = []
    bces = []

    for t in range(T):
        y_t = y_true[:, t]
        p_t = y_pred[:, t]
        mask = (~np.isnan(y_t)) & (y_t != -1)
        y_t = y_t[mask]
        p_t = p_t[mask]

        if y_t.size == 0 or len(np.unique(y_t)) < 2:
            continue

        aucs.append(roc_auc_score(y_t, p_t))
        # log_loss expects probs in (0,1), add epsilon
        eps = 1e-7
        p_t_clip = np.clip(p_t, eps, 1 - eps)
        bces.append(log_loss(y_t, p_t_clip))

    if len(aucs) == 0:
        auc = float("nan")
        bce = float("nan")
    else:
        auc = float(np.mean(aucs))
        bce = float(np.mean(bces))

    return {"auc": auc, "bce": bce}


def bbbp_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Binary classification metrics: AUC + BCE.
    """
    mask = ~np.isnan(y_true)
    y_t = y_true[mask]
    p_t = y_pred[mask]
    if y_t.size == 0 or len(np.unique(y_t)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_t, p_t))

    eps = 1e-7
    p_t_clip = np.clip(p_t, eps, 1 - eps)
    bce = float(log_loss(y_t, p_t_clip))
    return {"auc": auc, "bce": bce}


# ==================== BASELINE RUNNERS ==================== #

def run_esol_baselines(seeds: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns: {model_name: {metric_name: [values over seeds]}}
    model_name in ["RF", "MLP"]
    """
    results = {"RF": [], "MLP": []}
    for seed in seeds:
        set_seed(seed)
        X_train, y_train, X_val, y_val, X_test, y_test = load_esol_splits(seed)

        # Combine train+val for final training (since hyperparams are fixed)
        X_tr = np.concatenate([X_train, X_val], axis=0)
        y_tr = np.concatenate([y_train, y_val], axis=0)

        # RF baseline
        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_test)
        results["RF"].append(esol_metrics(y_test, y_pred_rf))

        # MLP baseline
        mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=200,
            random_state=seed,
        )
        mlp.fit(X_tr, y_tr)
        y_pred_mlp = mlp.predict(X_test)
        results["MLP"].append(esol_metrics(y_test, y_pred_mlp))

    # aggregate
    agg_out: Dict[str, Dict[str, float]] = {}
    for model_name, metrics_list in results.items():
        agg = {}
        for m in metrics_list:
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        agg_out[model_name] = {
            f"{k}_mean": float(np.mean(vs)) for k, vs in agg.items()
        }
        agg_out[model_name].update(
            {f"{k}_std": float(np.std(vs, ddof=0)) for k, vs in agg.items()}
        )
    return agg_out


def run_tox21_baselines(seeds: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = {"RF": [], "MLP": []}
    for seed in seeds:
        set_seed(seed)
        X_train, y_train, X_val, y_val, X_test, y_test = load_tox21_splits(seed)

        # Combine train+val
        X_tr = np.concatenate([X_train, X_val], axis=0)
        y_tr = np.concatenate([y_train, y_val], axis=0)

        # RF baseline (multi-output)
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
        # Need to handle missing labels: train only where labels are 0/1
        mask_train = (y_tr != -1) & (~np.isnan(y_tr))
        # For simplicity, flatten tasks and then reshape predictions
        # But that's messy; instead, train per-task loop to get probs.
        # We'll do per-task training.

        def train_and_predict_rf():
            probs = np.zeros_like(y_test, dtype=np.float32)
            T = y_tr.shape[1]
            for t in range(T):
                y_t = y_tr[:, t]
                m_t = (y_t != -1) & (~np.isnan(y_t))
                if m_t.sum() < 10 or len(np.unique(y_t[m_t])) < 2:
                    probs[:, t] = 0.5  # neutral
                    continue
                clf = RandomForestClassifier(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=-1,
                )
                clf.fit(X_tr[m_t], y_t[m_t])
                probs[:, t] = clf.predict_proba(X_test)[:, 1]
            return probs

        def train_and_predict_mlp():
            probs = np.zeros_like(y_test, dtype=np.float32)
            T = y_tr.shape[1]
            for t in range(T):
                y_t = y_tr[:, t]
                m_t = (y_t != -1) & (~np.isnan(y_t))
                if m_t.sum() < 10 or len(np.unique(y_t[m_t])) < 2:
                    probs[:, t] = 0.5
                    continue
                clf = MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    max_iter=200,
                    random_state=seed,
                )
                clf.fit(X_tr[m_t], y_t[m_t])
                probs[:, t] = clf.predict_proba(X_test)[:, 1]
            return probs

        y_pred_rf = train_and_predict_rf()
        results["RF"].append(tox21_metrics(y_test, y_pred_rf))

        y_pred_mlp = train_and_predict_mlp()
        results["MLP"].append(tox21_metrics(y_test, y_pred_mlp))

    agg_out: Dict[str, Dict[str, float]] = {}
    for model_name, metrics_list in results.items():
        agg = {}
        for m in metrics_list:
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        agg_out[model_name] = {
            f"{k}_mean": float(np.mean(vs)) for k, vs in agg.items()
        }
        agg_out[model_name].update(
            {f"{k}_std": float(np.std(vs, ddof=0)) for k, vs in agg.items()}
        )
    return agg_out


def run_bbbp_baselines(seeds: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = {"RF": [], "MLP": []}
    for seed in seeds:
        set_seed(seed)
        X_train, y_train, X_val, y_val, X_test, y_test = load_bbbp_splits(seed)

        # Combine train+val
        X_tr = np.concatenate([X_train, X_val], axis=0)
        y_tr = np.concatenate([y_train, y_val], axis=0)

        # RF
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_test)[:, 1]
        results["RF"].append(bbbp_metrics(y_test, p_rf))

        # MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=200,
            random_state=seed,
        )
        mlp.fit(X_tr, y_tr)
        p_mlp = mlp.predict_proba(X_test)[:, 1]
        results["MLP"].append(bbbp_metrics(y_test, p_mlp))

    agg_out: Dict[str, Dict[str, float]] = {}
    for model_name, metrics_list in results.items():
        agg = {}
        for m in metrics_list:
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        agg_out[model_name] = {
            f"{k}_mean": float(np.mean(vs)) for k, vs in agg.items()
        }
        agg_out[model_name].update(
            {f"{k}_std": float(np.std(vs, ddof=0)) for k, vs in agg.items()}
        )
    return agg_out


# ==================== MAIN: RUN BASELINES & MERGE INTO SUMMARY ==================== #

def main():
    seeds = [42, 43, 44]

    print("Running ESOL baselines (RF, MLP)...")
    esol_baselines = run_esol_baselines(seeds)

    print("Running Tox21 baselines (RF, MLP)...")
    tox21_baselines = run_tox21_baselines(seeds)

    print("Running BBBP baselines (RF, MLP)...")
    bbbp_baselines = run_bbbp_baselines(seeds)

    # Load existing combined GNN summary
    combined_dir = Path("results/combined")
    base_summary_path = combined_dir / "all_datasets_summary.xlsx"
    if not base_summary_path.exists():
        raise FileNotFoundError(
            f"Expected combined summary at {base_summary_path}. "
            "Run merge_summaries.py first."
        )

    df_all = pd.read_excel(base_summary_path, sheet_name="summary_all")

    # Build baseline rows
    new_rows = []

    # ESOL
    for model_name, metrics in esol_baselines.items():
        row = {
            "dataset": "ESOL",
            "model": model_name,
            "rmse_mean": metrics.get("rmse_mean"),
            "rmse_std": metrics.get("rmse_std"),
            "mae_mean": metrics.get("mae_mean"),
            "mae_std": metrics.get("mae_std"),
        }
        new_rows.append(row)

    # Tox21
    for model_name, metrics in tox21_baselines.items():
        row = {
            "dataset": "TOX21",
            "model": model_name,
            "auc_mean": metrics.get("auc_mean"),
            "auc_std": metrics.get("auc_std"),
            "bce_mean": metrics.get("bce_mean"),
            "bce_std": metrics.get("bce_std"),
        }
        new_rows.append(row)

    # BBBP
    for model_name, metrics in bbbp_baselines.items():
        row = {
            "dataset": "BBBP",
            "model": model_name,
            "auc_mean": metrics.get("auc_mean"),
            "auc_std": metrics.get("auc_std"),
            "bce_mean": metrics.get("bce_mean"),
            "bce_std": metrics.get("bce_std"),
        }
        new_rows.append(row)

    df_baselines = pd.DataFrame(new_rows)

    # Merge: append baselines to existing GNN summary
    df_all_with_baselines = pd.concat(
        [df_all, df_baselines],
        ignore_index=True,
        sort=False,
    )

    out_path = combined_dir / "all_datasets_summary_with_baselines.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_all_with_baselines.to_excel(writer, sheet_name="summary_all", index=False)
        # also keep original GNN-only summary for reference
        df_all.to_excel(writer, sheet_name="gnn_only", index=False)
        df_baselines.to_excel(writer, sheet_name="baselines_only", index=False)

    print(f"[OK] Wrote combined GNN + baseline summary to {out_path}")


if __name__ == "__main__":
    main()
