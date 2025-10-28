#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export per-sample predictions to CSV so visualization scripts can select top molecules.

Output:
  results/<dataset>/metrics_<dataset>_<model>_<split>_per_sample.csv
Columns (binary):
  index,id,y_true,y_pred,prob
Columns (regression):
  index,id,y_true,y_pred
Columns (tox21 multilabel):
  index,id,task,y_true,y_pred,prob
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import torch
import torch.nn.functional as F
import pandas as pd

# --- Bootstrap repo root onto sys.path so 'src' is importable even if -e not installed ---
REPO_ROOT = Path(__file__).resolve().parents[1]  # .../gnn_midterm_starter_v2
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Try to import your project modules now ---
try:
    from src.data.datasets import load_dataset             # <-- adapt to your project API if needed
    from src.models import build_model                     # <-- adapt to your project API if needed
except Exception as e:
    raise RuntimeError(
        "Failed to import project modules. "
        "Make sure you run from repo root and that 'src' exists.\n"
        f"sys.path head: {sys.path[:3]}\nOriginal error: {e}"
    ) from e


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_classification(model, loader, device, multilabel: bool = False, task_names=None):
    rows = []
    model.eval()
    for batch in loader:
        # Adjust this unpacking to your DataLoader structure if needed.
        if isinstance(batch, dict):
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch"].to(device)
            y = batch["y"].to(device)
            ids = batch.get("ids", None)
            idx_list = batch.get("idx_list", None)
        else:
            data = batch
            x, edge_index, batch_idx = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            y = data.y.to(device)
            ids = getattr(data, "ids", None)
            idx_list = getattr(data, "idx_list", None)

        logits = model(x, edge_index, batch_idx)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        if multilabel:
            prob = torch.sigmoid(logits)  # [B, T]
            y_true = y.float()
            y_pred = (prob >= 0.5).float()
            B, T = prob.shape
            for i in range(B):
                smi = ids[i] if ids is not None else None
                idxv = int(idx_list[i]) if idx_list is not None else None
                for t in range(T):
                    rows.append({
                        "index": idxv,
                        "id": smi,
                        "task": task_names[t] if task_names else f"task_{t}",
                        "y_true": float(y_true[i, t].item()),
                        "y_pred": float(y_pred[i, t].item()),
                        "prob": float(prob[i, t].item()),
                    })
        else:
            if logits.size(1) == 1:
                prob_pos = torch.sigmoid(logits.squeeze(1))
                y_pred = (prob_pos >= 0.5).long()
                y_true = y.long().view(-1)
                for i in range(prob_pos.size(0)):
                    smi = ids[i] if ids is not None else None
                    idxv = int(idx_list[i]) if idx_list is not None else None
                    rows.append({
                        "index": idxv,
                        "id": smi,
                        "y_true": int(y_true[i].item()),
                        "y_pred": int(y_pred[i].item()),
                        "prob": float(prob_pos[i].item()),
                    })
            else:
                probs = torch.softmax(logits, dim=1)
                y_pred = probs.argmax(dim=1)
                y_true = y.long().view(-1)
                for i in range(probs.size(0)):
                    smi = ids[i] if ids is not None else None
                    idxv = int(idx_list[i]) if idx_list is not None else None
                    rows.append({
                        "index": idxv,
                        "id": smi,
                        "y_true": int(y_true[i].item()),
                        "y_pred": int(y_pred[i].item()),
                        "prob": float(probs[i, y_pred[i]].item()),
                    })
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_regression(model, loader, device):
    rows = []
    model.eval()
    for batch in loader:
        if isinstance(batch, dict):
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch"].to(device)
            y = batch["y"].to(device)
            ids = batch.get("ids", None)
            idx_list = batch.get("idx_list", None)
        else:
            data = batch
            x, edge_index, batch_idx = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            y = data.y.to(device)
            ids = getattr(data, "ids", None)
            idx_list = getattr(data, "idx_list", None)

        pred = model(x, edge_index, batch_idx).squeeze(-1)
        y_true = y.view(-1).float()
        for i in range(pred.size(0)):
            smi = ids[i] if ids is not None else None
            idxv = int(idx_list[i]) if idx_list is not None else None
            rows.append({
                "index": idxv,
                "id": smi,
                "y_true": float(y_true[i].item()),
                "y_pred": float(pred[i].item()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "valid", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build loaders (adapt to your load_dataset return type)
    loaders = load_dataset(cfg)
    split_key = "valid" if args.split in ("val", "valid") else "test"
    if isinstance(loaders, dict):
        loader = loaders[split_key]
        dataset_obj = loaders.get(f"{split_key}_dataset", None)
    elif isinstance(loaders, (list, tuple)) and len(loaders) >= 3:
        train_loader, val_loader, test_loader = loaders[:3]
        loader = val_loader if split_key == "valid" else test_loader
        dataset_obj = None
    else:
        raise RuntimeError("Unexpected load_dataset(cfg) return. Adapt this script to your project.")

    # Build model + load weights
    model = build_model(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state
    model.load_state_dict(sd, strict=False)
    model.to(device)

    # Decide task type (heuristics if missing in cfg)
    task_type = str(cfg.get("task_type", cfg.get("dataset", {}).get("task_type", ""))).lower()
    if not task_type:
        name = str(cfg.get("dataset", {}).get("name", "")).lower()
        task_type = "regression" if "esol" in name else ("multilabel" if "tox21" in name else "binary")

    # Predict
    if task_type == "regression":
        df = predict_regression(model, loader, device)
        dataset_name = "esol"
    elif task_type == "multilabel":
        task_names = cfg.get("dataset", {}).get("tasks", None)
        df = predict_classification(model, loader, device, multilabel=True, task_names=task_names)
        dataset_name = "tox21"
    else:
        df = predict_classification(model, loader, device, multilabel=False)
        dataset_name = "bbbp"

    # Infer model name from config filename
    cfgn = Path(args.config).name.lower()
    model_name = "gin"
    for key in ("gin", "gat", "gcn"):
        if key in cfgn:
            model_name = key

    out_dir = Path("results") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"metrics_{dataset_name}_{model_name}_{split_key}_per_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved per-sample predictions to: {out_path.resolve()}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
