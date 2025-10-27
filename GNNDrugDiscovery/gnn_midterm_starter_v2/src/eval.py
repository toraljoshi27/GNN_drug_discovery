
import os, argparse, yaml, csv
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from src.data.datasets import load_dataset
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT

@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "edge_attr", None))
        if task == "regression":
            ys.append(batch.y.view(-1,1).cpu())
            ps.append(logits.cpu())
        else:
            ys.append(batch.y.cpu())
            ps.append(torch.sigmoid(logits).cpu())
    import numpy as np
    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(ps, dim=0).numpy()
    if task == "binary":
        return {"roc_auc": roc_auc_score(y, p)}
    if task == "multilabel":
        aucs = []
        for t in range(y.shape[1]):
            mask = ~np.isnan(y[:,t])
            if mask.sum() == 0: continue
            aucs.append(roc_auc_score(y[mask, t], p[mask, t]))
        return {"roc_auc_macro": float(sum(aucs)/len(aucs))}
    rmse = mean_squared_error(y, p, squared=False)
    mae  = mean_absolute_error(y, p)
    return {"rmse": rmse, "mae": mae}

def build_model(name, in_dim, cfg, out_dim, task):
    m = cfg["model"]
    if name == "gcn":
        return GCN(in_dim, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], out_dim=out_dim, task=task)
    if name == "gin":
        return GIN(in_dim, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], out_dim=out_dim, task=task)
    if name == "gat":
        return GAT(in_dim, hidden_dim=m["hidden_dim"], heads=m.get("heads", 4), num_layers=m["num_layers"], dropout=m["dropout"], out_dim=out_dim, task=task)
    raise ValueError(f"Unknown model {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)
    dcfg = cfg["dataset"]
    loaders = load_dataset(dcfg["name"], split=dcfg.get("split","scaffold"),
                           batch_size=dcfg.get("batch_size",128),
                           num_workers=dcfg.get("num_workers",0),
                           seed=seed)
    task = loaders["task"]; out_dim = loaders["out_dim"]; in_dim = loaders["num_node_features"]
    from src.models.gcn import GCN
    model = build_model(cfg["model"]["name"], in_dim, cfg, out_dim, task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ckpt = os.path.join(cfg["logging"]["out_dir"], "best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    metrics = evaluate(model, loaders["test_loader"], task, device)
    # Write CSV
    subdir = os.path.join("results", dcfg["name"])
    os.makedirs(subdir, exist_ok=True)
    out_csv = os.path.join(subdir, f"metrics_{cfg['model']['name']}.csv")
    with open(out_csv, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow(list(metrics.values()))
    print(f"Wrote metrics to {out_csv}: {metrics}")

if __name__ == "__main__":
    main()
