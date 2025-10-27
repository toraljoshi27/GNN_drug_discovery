
import os, argparse, yaml, math, csv
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from torch.optim import Adam
from src.data.datasets import load_dataset
from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT

def build_model(name, in_dim, cfg, out_dim, task):
    m = cfg["model"]
    if name == "gcn":
        return GCN(in_dim, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], out_dim=out_dim, task=task)
    if name == "gin":
        return GIN(in_dim, hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], out_dim=out_dim, task=task)
    if name == "gat":
        return GAT(in_dim, hidden_dim=m["hidden_dim"], heads=m.get("heads", 4), num_layers=m["num_layers"], dropout=m["dropout"], out_dim=out_dim, task=task)
    raise ValueError(f"Unknown model {name}")

def bce_logits_with_mask(outputs, targets):
    mask = ~torch.isnan(targets)
    return nn.functional.binary_cross_entropy_with_logits(outputs[mask], targets[mask])

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
    y = torch.cat(ys, dim=0).numpy()
    p = torch.cat(ps, dim=0).numpy()
    if task == "binary":
        return roc_auc_score(y, p)
    if task == "multilabel":
        import numpy as np
        aucs = []
        for t in range(y.shape[1]):
            mask = ~np.isnan(y[:,t])
            if mask.sum() == 0: 
                continue
            aucs.append(roc_auc_score(y[mask, t], p[mask, t]))
        return float(sum(aucs)/len(aucs))
    return mean_squared_error(y, p, squared=False)

def train_one(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    dcfg = cfg["dataset"]
    loaders = load_dataset(dcfg["name"], split=dcfg.get("split","scaffold"),
                           batch_size=dcfg.get("batch_size",128),
                           num_workers=dcfg.get("num_workers",0),
                           seed=seed)
    task = loaders["task"]; out_dim = loaders["out_dim"]
    in_dim = loaders["num_node_features"]
    model = build_model(cfg["model"]["name"], in_dim, cfg, out_dim, task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    best_metric = -1e9 if task != "regression" else 1e9
    patience = cfg["train"]["early_stopping_patience"]
    wait = 0
    out_dir = cfg["logging"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, cfg["train"]["max_epochs"]+1):
        model.train(); total_loss = 0.0
        for batch in loaders["train_loader"]:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "edge_attr", None))
            if task == "multilabel":
                y = batch.y.to(torch.float32)
                loss = bce_logits_with_mask(logits, y)
            elif task == "binary":
                y = batch.y.to(torch.float32).view(-1,1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            else:
                y = batch.y.to(torch.float32).view(-1,1)
                loss = nn.functional.mse_loss(logits, y)
            loss.backward(); opt.step()
            total_loss += loss.item() * batch.num_graphs
        # Validation
        if epoch % cfg["eval"]["eval_interval"] == 0:
            val_metric = evaluate(model, loaders["val_loader"], task, device)
            improved = (val_metric > best_metric) if task != "regression" else (val_metric < best_metric)
            if improved:
                best_metric = val_metric; wait = 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            else:
                wait += 1
                if wait >= patience:
                    break
    return os.path.join(out_dir, "best.pt")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    ckpt = train_one(args.config)
    print(f"Saved best checkpoint to {ckpt}")

if __name__ == "__main__":
    main()
