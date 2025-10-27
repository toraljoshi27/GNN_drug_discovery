# src/train.py
import os
import argparse
import yaml
import math
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, mean_absolute_error
from src.data.datasets import load_dataset
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT


# ---------------------------
# Utilities
# ---------------------------
def _to_float(x, default=None):
    if x is None and default is not None:
        return float(default)
    # allows strings like "1e-4" or numbers
    return float(x)


def build_model(name, in_dim, cfg, out_dim, task):
    m = cfg["model"]
    if name == "gcn":
        return GCN(
            in_dim,
            hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            out_dim=out_dim,
            task=task,
        )
    if name == "gin":
        return GIN(
            in_dim,
            hidden_dim=m["hidden_dim"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            out_dim=out_dim,
            task=task,
        )
    if name == "gat":
        return GAT(
            in_dim,
            hidden_dim=m["hidden_dim"],
            heads=m.get("heads", 4),
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            out_dim=out_dim,
            task=task,
        )
    raise ValueError(f"Unknown model {name}")


def bce_logits_with_mask(outputs, targets):
    # targets may contain NaNs; mask them out
    mask = ~torch.isnan(targets)
    return nn.functional.binary_cross_entropy_with_logits(outputs[mask], targets[mask])

@torch.no_grad()
def evaluate(model, loader, task, device, metrics=("rmse", "mae")):
    """
    Safe eval:
      - regression: rmse, mae
      - binary: auc (NaN-proof) + acc
      - multilabel: mean-auc over labels with >=2 classes; else skip label
    """
    import numpy as np
    import warnings
    from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score
    from sklearn.exceptions import UndefinedMetricWarning

    # Silence sklearn warning just in case (we avoid the call anyway)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    model.eval()
    ys, ps = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch,
            getattr(batch, "edge_attr", None),
        )
        if task == "regression":
            ys.append(batch.y.view(-1, 1).detach().cpu())
            ps.append(logits.detach().cpu())
        else:
            ys.append(batch.y.detach().cpu())
            ps.append(torch.sigmoid(logits).detach().cpu())

    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(ps, dim=0).numpy()
    results = {}

    if task == "regression":
        mset = tuple(m.lower() for m in metrics)
        if "rmse" in mset:
            rmse = float(np.sqrt(np.mean((y - p) ** 2)))
            results["rmse"] = rmse
        if "mae" in mset:
            mae = float(mean_absolute_error(y, p))
            results["mae"] = mae

    elif task == "binary":
        y_flat = y.reshape(-1)
        p_flat = p.reshape(-1)
        # Class balance info (useful for debugging)
        classes, counts = np.unique(y_flat, return_counts=True)
        # Safe AUC
        if classes.size < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_flat, p_flat))
        # Accuracy for visibility
        preds = (p_flat >= 0.5).astype(np.float32)
        acc = float(accuracy_score(y_flat, preds))
        results["auc"] = auc
        results["acc"] = acc
        # (Optional) expose counts for your console log:
        results["_val_counts_0"] = int(counts[classes.tolist().index(0)]) if 0 in classes else 0
        results["_val_counts_1"] = int(counts[classes.tolist().index(1)]) if 1 in classes else 0

    elif task == "multilabel":
        aucs = []
        for t in range(y.shape[1]):
            y_t = y[:, t]
            p_t = p[:, t]
            mask = ~np.isnan(y_t)
            if mask.sum() == 0:
                continue
            y_m = y_t[mask]
            p_m = p_t[mask]
            classes = np.unique(y_m)
            if classes.size < 2:
                continue  # skip labels with single class
            aucs.append(roc_auc_score(y_m, p_m))
        results["mean-auc"] = float(np.mean(aucs)) if len(aucs) else 0.5

    else:
        raise ValueError(f"Unknown task type: {task}")

    return results








###Early Stopping and Evaluation-Evaluate -Done###
# @torch.no_grad()
# def evaluate(model, loader, task, device, metrics=("rmse", "mae")):
#     """
#     Evaluate the model on a DataLoader.
#
#     Args:
#         model: PyTorch model
#         loader: DataLoader with batches that have x, edge_index, batch, (optional) edge_attr, and y
#         task: one of {"regression", "binary", "multilabel"}
#         device: torch.device
#         metrics: tuple of metric names to compute (only used for regression)
#
#     Returns:
#         dict metric_name -> float
#         - regression: {"rmse": ..., "mae": ...} (subset depending on `metrics`)
#         - binary:     {"auc": ...}
#         - multilabel: {"mean-auc": ...}
#     """
#     import numpy as np
#     from sklearn.metrics import roc_auc_score, mean_absolute_error
#
#     def _safe_binary_auc(y_true_flat, y_prob_flat):
#         # If only one class present, AUC is undefined: return 0.5 without calling sklearn.
#         classes = np.unique(y_true_flat)
#         if classes.size < 2:
#             return 0.5
#         try:
#             return float(roc_auc_score(y_true_flat, y_prob_flat))
#         except Exception:
#             return 0.5
#
#     model.eval()
#     ys, ps = [], []
#
#     for batch in loader:
#         batch = batch.to(device)
#         # Ensure floating dtype for node features to avoid GCNConv dtype errors
#         logits = model(
#             batch.x.float(),
#             batch.edge_index,
#             batch.batch,
#             getattr(batch, "edge_attr", None),
#         )
#
#         if task == "regression":
#             # y: (N, 1), p: (N, 1)
#             ys.append(batch.y.view(-1, 1).detach().cpu())
#             ps.append(logits.detach().cpu())
#         else:
#             # For classification, we will compute metrics on probabilities
#             ys.append(batch.y.detach().cpu())
#             ps.append(torch.sigmoid(logits).detach().cpu())
#
#     y = torch.cat(ys, dim=0).numpy()
#     p = torch.cat(ps, dim=0).numpy()
#
#     results = {}
#
#     if task == "regression":
#         # Compute requested metrics; RMSE computed manually (no sklearn 'squared' flag dependency)
#         metrics_lower = tuple(m.lower() for m in metrics)
#         if "rmse" in metrics_lower:
#             rmse = float(np.sqrt(np.mean((y - p) ** 2)))
#             results["rmse"] = rmse
#         if "mae" in metrics_lower:
#             mae = float(mean_absolute_error(y, p))
#             results["mae"] = mae
#
#     elif task == "binary":
#         # Expect y and p shaped (N, 1) or (N,)
#         y_flat = y.reshape(-1)
#         p_flat = p.reshape(-1)
#         try:
#             auc = float(roc_auc_score(y_flat, p_flat))
#         except Exception:
#             # e.g., only one class present -> undefined AUC
#             auc = 0.5
#         results["auc"] = auc
#
#     elif task == "multilabel":
#         # Average AUC across labels, skipping labels with all-NaN or single-class targets
#         aucs = []
#         for t in range(y.shape[1]):
#             mask = ~np.isnan(y[:, t])
#             if mask.sum() == 0:
#                 continue
#             y_t = y[mask, t]
#             p_t = p[mask, t]
#             # Need both positive and negative examples
#             if np.unique(y_t).size < 2:
#                 continue
#             try:
#                 aucs.append(roc_auc_score(y_t, p_t))
#             except Exception:
#                 pass
#         results["mean-auc"] = float(np.mean(aucs)) if len(aucs) else 0.5
#
#     else:
#         raise ValueError(f"Unknown task type: {task}")
#
#     return results
##################################################
# @torch.no_grad()
# def evaluate(model, loader, task, device, metrics=("rmse",)):
#     """
#     Returns a dict of {metric_name: value} computed on the given loader.
#     Supported metrics for regression: rmse, mae
#     For binary: auc
#     For multilabel: mean-auc (averaged over valid labels)
#     """
#     model.eval()
#
#     ys, ps = [], []
#     for batch in loader:
#         batch = batch.to(device)
#         logits = model(
#             batch.x,
#             batch.edge_index,
#             batch.batch,
#             getattr(batch, "edge_attr", None),
#         )
#         if task == "regression":
#             ys.append(batch.y.view(-1, 1).cpu())
#             ps.append(logits.cpu())
#         else:
#             ys.append(batch.y.cpu())
#             ps.append(torch.sigmoid(logits).cpu())
#
#     y = torch.cat(ys, dim=0).numpy()
#     p = torch.cat(ps, dim=0).numpy()
#
#     results = {}
#
#     if task == "regression":
#         # RMSE without relying on sklearn's 'squared' flag
#         if "rmse" in metrics:
#             rmse = float(np.sqrt(np.mean((y - p) ** 2)))
#             results["rmse"] = rmse
#         if "mae" in metrics:
#             # sklearn's MAE is stable across versions
#             mae = float(mean_absolute_error(y, p))
#             results["mae"] = mae
#
#     elif task == "binary":
#         # Single-column targets/probs expected
#         try:
#             auc = float(roc_auc_score(y, p))
#         except Exception:
#             # If only one class present in y, roc_auc_score fails; fall back to 0.5
#             auc = 0.5
#         results["auc"] = auc
#
#     elif task == "multilabel":
#         # Average AUC over labels (ignoring labels with all-NaN or single class)
#         aucs = []
#         for t in range(y.shape[1]):
#             mask = ~np.isnan(y[:, t])
#             if mask.sum() == 0:
#                 continue
#             try:
#                 aucs.append(roc_auc_score(y[mask, t], p[mask, t]))
#             except Exception:
#                 # skip labels where AUC is undefined
#                 pass
#         results["mean-auc"] = float(np.mean(aucs)) if len(aucs) else 0.5
#
#     else:
#         raise ValueError(f"Unknown task type: {task}")
#
#     return results


def pick_monitor_value(metrics_dict, task, save_best_metric):
    """
    Given a dict of computed metrics and config choice `save_best_metric`,
    return (value, mode) where mode is 'min' or 'max'.
    """
    # Default per task
    if task == "regression":
        # Allowed: rmse, mae
        metric = (save_best_metric or "rmse").lower()
        val = metrics_dict.get(metric)
        if val is None:
            # if not computed, fall back to any available regression metric
            for m in ("rmse", "mae"):
                if m in metrics_dict:
                    metric, val = m, metrics_dict[m]
                    break
        mode = "min"
    else:
        # binary -> auc, multilabel -> mean-auc
        metric = (save_best_metric or ("auc" if task == "binary" else "mean-auc")).lower()
        val = metrics_dict.get(metric)
        if val is None:
            # fall back to the first key
            metric, val = next(iter(metrics_dict.items()))
        mode = "max"
    return metric, val, mode


def train_one(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    dcfg = cfg["dataset"]
    loaders = load_dataset(
        dcfg["name"],
        split=dcfg.get("split", "scaffold"),
        batch_size=int(dcfg.get("batch_size", 128)),
        num_workers=int(dcfg.get("num_workers", 0)),
        seed=seed,
    )
    task = loaders["task"]
    out_dim = loaders["out_dim"]
    in_dim = loaders["num_node_features"]

    # Model
    model = build_model(cfg["model"]["name"], in_dim, cfg, out_dim, task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer (cast numeric fields for robustness)
    tcfg = cfg["train"]
    lr = _to_float(tcfg.get("lr", 1e-3))
    weight_decay = _to_float(tcfg.get("weight_decay", 0.0))
    optimizer_name = str(tcfg.get("optimizer", "adam")).lower()

    if optimizer_name == "adam":
        opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        from torch.optim import AdamW
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Training controls
    max_epochs = int(tcfg.get("max_epochs", 100))
    patience = int(tcfg.get("early_stopping_patience", 10))

    # Eval controls
    ecfg = cfg.get("eval", {})
    eval_interval = int(ecfg.get("eval_interval", 1))
    requested_metrics = tuple(m.lower() for m in ecfg.get("metrics", ["rmse", "mae"]))
    # Logging
    lcfg = cfg.get("logging", {})
    out_dir = lcfg.get("out_dir", "runs/default")
    os.makedirs(out_dir, exist_ok=True)
    save_best_metric = lcfg.get("save_best_metric", None)

    # Best tracking
    # Initialize based on minimize/maximize later
    best_val = None
    best_metric_name = None
    best_mode = None
    wait = 0

    # Training loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in loaders["train_loader"]:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "edge_attr", None),
            )

            if task == "multilabel":
                y = batch.y.to(torch.float32)
                loss = bce_logits_with_mask(logits, y)
            elif task == "binary":
                y = batch.y.to(torch.float32).view(-1, 1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            else:
                y = batch.y.to(torch.float32).view(-1, 1)
                loss = nn.functional.mse_loss(logits, y)

            loss.backward()
            opt.step()

            bs = getattr(batch, "num_graphs", y.shape[0])
            total_loss += loss.item() * bs
            total_graphs += bs

        avg_loss = total_loss / max(1, total_graphs)

        # Validation/Eval
        if epoch % eval_interval == 0:
            val_metrics = evaluate(
                model, loaders["val_loader"], task, device, metrics=requested_metrics
            )
            # Decide what to monitor
            metric_name, metric_val, mode = pick_monitor_value(
                val_metrics, task, save_best_metric
            )

            if best_mode is None:
                best_mode = mode
                best_metric_name = metric_name
                best_val = metric_val
                improved = True
            else:
                if best_mode == "min":
                    improved = metric_val < best_val
                else:
                    improved = metric_val > best_val

            if improved:
                best_val = metric_val
                wait = 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            else:
                wait += 1

            # Console log
            metrics_str = " ".join(f"{k}={v:.5f}" for k, v in val_metrics.items())
            print(
                f"[Epoch {epoch:03d}] train_loss={avg_loss:.5f} | {metrics_str} "
                f"| monitor={best_metric_name}:{metric_val:.5f} (best={best_val:.5f}) wait={wait}"
            )

            if wait >= patience:
                print("Early stopping.")
                break
        else:
            # Still nice to see progress
            print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.5f}")

    return os.path.join(out_dir, "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    ckpt = train_one(args.config)
    print(f"Saved best checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
