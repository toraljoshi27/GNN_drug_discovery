# scripts/evaluate_and_plot.py
import os, json, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---- Project imports ----
from src.data.datasets import load_dataset
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT


# -------------------- utilities --------------------
def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def save_fig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def sanitize(s):
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s)).strip("_")


# -------------------- model builder ----------------
def build_model_from_cfg(cfg, in_dim, out_dim, task):
    m = cfg["model"]; name = m["name"].lower()
    if name == "gcn":
        return GCN(in_dim, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"],
                   dropout=m["dropout"], out_dim=out_dim, task=task)
    if name == "gin":
        return GIN(in_dim, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"],
                   dropout=m["dropout"], out_dim=out_dim, task=task)
    if name == "gat":
        return GAT(in_dim, hidden_dim=m["hidden_dim"], heads=m.get("heads", 4),
                   num_layers=m["num_layers"], dropout=m["dropout"],
                   out_dim=out_dim, task=task)
    raise ValueError(f"Unknown model {name}")


# -------------------- metric helpers ----------------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def safe_binary_auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    y_true, y_prob = np.asarray(y_true).ravel(), np.asarray(y_prob).ravel()
    if np.unique(y_true).size < 2:
        return 0.5
    try: return float(roc_auc_score(y_true, y_prob))
    except Exception: return 0.5

def compute_regression_metrics(y_true, preds):
    y_true, preds = np.asarray(y_true).ravel(), np.asarray(preds).ravel()
    rmse = float(np.sqrt(np.mean((y_true - preds)**2)))
    mae  = float(np.mean(np.abs(y_true - preds)))
    return {"rmse": rmse, "mae": mae}

def compute_binary_metrics(y_true, logits):
    from sklearn.metrics import accuracy_score, average_precision_score
    probs = sigmoid(logits)
    auc = safe_binary_auc(y_true, probs)
    ap  = float(average_precision_score(y_true, probs)) if len(np.unique(y_true))>1 else 0.5
    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_true.ravel(), preds.ravel()))
    return {"auc": auc, "pr_auc": ap, "acc": acc}

def compute_multilabel_metrics(Y_true, L_logits):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    probs = sigmoid(L_logits)
    T = Y_true.shape[1]; aucs, aps, f1s = [], [], []
    for t in range(T):
        y, p = Y_true[:,t], probs[:,t]
        mask = ~np.isnan(y)
        if mask.sum()==0: continue
        y, p = y[mask], p[mask]
        if np.unique(y).size<2: continue
        aucs.append(roc_auc_score(y,p))
        aps.append(average_precision_score(y,p))
        preds = (p>=0.5).astype(int)
        f1s.append(f1_score(y,preds))
    return {
        "mean_auc": float(np.mean(aucs)) if aucs else 0.5,
        "mean_pr_auc": float(np.mean(aps)) if aps else 0.5,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0
    }


# -------------------- plotting ----------------------
def plot_roc_pr_binary(y_true, logits, out_dir, tag):
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    y_true = np.asarray(y_true).ravel()
    probs = sigmoid(np.asarray(logits).ravel())

    classes, counts = np.unique(y_true, return_counts=True)
    has_both = (classes.size == 2)

    # Always create the figures directory
    Path(os.path.join(out_dir, "figures")).mkdir(parents=True, exist_ok=True)

    if has_both:
        # --- ROC ---
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {tag}")
        plt.legend()
        save_fig(os.path.join(out_dir, "figures", f"roc_{tag}.png"))

        # --- PR ---
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {tag}")
        plt.legend()
        save_fig(os.path.join(out_dir, "figures", f"pr_{tag}.png"))

    else:
        # Fallback 1: Class count bar chart
        plt.figure()
        xs = [int(c) for c in classes.tolist()]
        plt.bar([str(x) for x in xs], counts, color="skyblue")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(f"Class Counts (single-class) - {tag}")
        save_fig(os.path.join(out_dir, "figures", f"class_counts_{tag}.png"))

        # Fallback 2: Probability histogram
        plt.figure()
        plt.hist(probs, bins=20, color="salmon", edgecolor="black")
        plt.xlabel("Predicted Probability (Positive)")
        plt.ylabel("Frequency")
        plt.title(f"Score Distribution - {tag}")
        save_fig(os.path.join(out_dir, "figures", f"prob_hist_{tag}.png"))


# def plot_roc_pr_binary(y_true, logits, out_dir, tag):
#     from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
#     probs = sigmoid(np.asarray(logits)); y_true = np.asarray(y_true).ravel()
#     if np.unique(y_true).size<2: return
#     fpr,tpr,_ = roc_curve(y_true,probs); roc_auc=auc(fpr,tpr)
#     plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
#     plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
#     plt.title(f"ROC Curve - {tag}"); plt.legend()
#     save_fig(os.path.join(out_dir,"figures",f"roc_{tag}.png"))
#     prec,rec,_ = precision_recall_curve(y_true,probs)
#     ap=average_precision_score(y_true,probs)
#     plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.3f}")
#     plt.xlabel("Recall"); plt.ylabel("Precision")
#     plt.title(f"PR Curve - {tag}"); plt.legend()
#     save_fig(os.path.join(out_dir,"figures",f"pr_{tag}.png"))

def plot_regression_parity(y_true, preds, out_dir, tag):
    y_true,preds = np.asarray(y_true).ravel(), np.asarray(preds).ravel()
    plt.figure(); plt.scatter(y_true,preds,s=14)
    mn,mx=min(y_true.min(),preds.min()),max(y_true.max(),preds.max())
    plt.plot([mn,mx],[mn,mx],"--"); plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(f"Parity - {tag}"); save_fig(os.path.join(out_dir,"figures",f"parity_{tag}.png"))
    res=preds-y_true
    plt.figure(); plt.scatter(y_true,res,s=14); plt.axhline(0,ls="--")
    plt.xlabel("True"); plt.ylabel("Residual"); plt.title(f"Residual - {tag}")
    save_fig(os.path.join(out_dir,"figures",f"residual_{tag}.png"))

def plot_multilabel_micro(Y_true, L_logits, out_dir, tag):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    probs = sigmoid(L_logits); mask = ~np.isnan(Y_true)
    if mask.sum()==0: return
    y,p = Y_true[mask].astype(int), probs[mask]
    if np.unique(y).size<2: return
    fpr,tpr,_ = roc_curve(y,p); roc_auc=auc(fpr,tpr)
    plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"Micro ROC - {tag}"); plt.legend()
    save_fig(os.path.join(out_dir,"figures",f"roc_micro_{tag}.png"))
    prec,rec,_=precision_recall_curve(y,p)
    ap=average_precision_score(y,p)
    plt.figure(); plt.plot(rec,prec,label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Micro PR - {tag}"); plt.legend()
    save_fig(os.path.join(out_dir,"figures",f"pr_micro_{tag}.png"))


# -------------------- main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val","valid","test"])
    ap.add_argument("--tag", default=None, help="used in filenames to avoid overwrites; defaults to experiment_name or model name")
    ap.add_argument("--save_dir", default=None, help="override base results dir (default: results/<dataset>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_name = sanitize(cfg["dataset"]["name"])
    default_tag = sanitize(cfg.get("experiment_name", cfg["model"]["name"]))
    tag = sanitize(args.tag) if args.tag is not None else default_tag
    split = "val" if args.split == "valid" else args.split

    # Base output dir -> results/<dataset>/  (or custom)
    base_dir = Path(args.save_dir) if args.save_dir else Path("results") / dataset_name
    ensure_dir(base_dir)
    ensure_dir(base_dir / "figures")

    # Data
    seed = int(cfg.get("seed",42))
    dcfg = cfg["dataset"]
    loaders = load_dataset(
        dcfg["name"], split=dcfg.get("split","scaffold"),
        batch_size=int(dcfg.get("batch_size",128)),
        num_workers=int(dcfg.get("num_workers",0)), seed=seed)
    task, out_dim, in_dim = loaders["task"], loaders["out_dim"], loaders["num_node_features"]
    loader = loaders["val_loader"] if split == "val" else loaders["test_loader"]

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg,in_dim,out_dim,task).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location=device)); model.eval()

    # Collect preds
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch,
                           getattr(batch,"edge_attr",None))
            all_logits.append(logits.cpu().numpy()); all_y.append(batch.y.cpu().numpy())
    logits = np.concatenate(all_logits,0); y_true = np.concatenate(all_y,0)

    # ---- compute metrics, plots, and CSVs ----
    if task=="regression":
        preds=logits.squeeze(); metrics=compute_regression_metrics(y_true,preds)
        plot_regression_parity(y_true,preds,base_dir,f"{tag}_{split}")
        df=pd.DataFrame({"y_true":y_true.ravel(),"y_pred":preds.ravel()})

    elif task=="binary":
        yb,lb=y_true.ravel(),logits.ravel()
        metrics=compute_binary_metrics(yb,lb)
        plot_roc_pr_binary(yb,lb,base_dir,f"{tag}_{split}")
        probs=sigmoid(lb); preds=(probs>=0.5).astype(int)
        df=pd.DataFrame({"y_true":yb,"logit":lb,"prob":probs,"pred":preds})

    elif task=="multilabel":
        metrics=compute_multilabel_metrics(y_true,logits)
        plot_multilabel_micro(y_true,logits,base_dir,f"{tag}_{split}")
        probs=sigmoid(logits); preds=(probs>=0.5).astype(int)
        cols={}
        for i in range(logits.shape[1]):
            cols[f"y_true_t{i}"]=y_true[:,i]; cols[f"logit_t{i}"]=logits[:,i]
            cols[f"prob_t{i}"]=probs[:,i];    cols[f"pred_t{i}"]=preds[:,i]
        df=pd.DataFrame(cols)
    else:
        raise ValueError(f"Unknown task {task}")

    # ---- save CSVs in results/<dataset>/ ----
    pred_csv   = base_dir / f"results_{tag}_{split}.csv"
    metrics_csv= base_dir / f"metrics_{tag}_{split}.csv"
    metrics_json = base_dir / f"metrics_{tag}_{split}.json"

    df.to_csv(pred_csv, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    json.dump(metrics, open(metrics_json,"w"), indent=2)

    print(f"\n✅  Saved outputs under {base_dir}")
    print(f" - Predictions CSV : {pred_csv.name}")
    print(f" - Metrics CSV     : {metrics_csv.name}")
    print(f" - Figures         : {base_dir/'figures'}\n")

if __name__=="__main__":
    main()
