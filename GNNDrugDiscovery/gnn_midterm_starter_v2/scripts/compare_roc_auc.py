# scripts/compare_roc_auc.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

AUC_KEYS = ["auc", "roc_auc", "ROC-AUC", "roc-auc", "test_auc", "val_auc"]
RMSE_KEYS = ["rmse", "test_rmse", "val_rmse"]
MAE_KEYS = ["mae", "test_mae", "val_mae"]

def find_column(df: pd.DataFrame, keys: list[str]) -> str | None:
    lowmap = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in lowmap:
            return lowmap[k.lower()]
    # fallback: contains substring
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None

def find_best_metric_column(df: pd.DataFrame):
    """
    Returns (metric_name_for_label, column_name_in_df, direction)
    direction: 'up' if higher is better, 'down' if lower is better
    """
    col = find_column(df, AUC_KEYS)
    if col:
        return ("ROC-AUC", col, "up")
    col = find_column(df, RMSE_KEYS)
    if col:
        return ("RMSE", col, "down")
    col = find_column(df, MAE_KEYS)
    if col:
        return ("MAE", col, "down")
    return (None, None, None)

def extract_model_name(stem: str) -> str:
    tokens = stem.split("_")
    # prefer known names if present
    for t in reversed(tokens):
        if t.lower() in {"gcn", "gin", "gat", "mlp", "rf"}:
            return t.upper()
    return tokens[-1].upper() if tokens else stem.upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bbbp", help="Folder with metrics CSVs")
    ap.add_argument("--dataset", default="bbbp", help="Dataset key in file names (e.g., bbbp, tox21, esol)")
    ap.add_argument("--split", default="test", help="Split suffix in file names (e.g., test, val)")
    ap.add_argument(
        "--pattern",
        default="metrics_{dataset}_*_{split}.csv",
        help="Glob pattern for files. Use {dataset} and {split} tokens."
    )
    ap.add_argument("--out", default="results/bbbp/figures/metric_comparison.png",
                    help="Output PNG path for the comparison plot (CSV saved alongside).")
    args = ap.parse_args()

    base = Path(args.dir)
    if not base.exists():
        sys.exit(f"Directory not found: {base}")

    pattern = args.pattern.format(dataset=args.dataset, split=args.split)
    paths = sorted(glob.glob(str(base / pattern)))
    if not paths:
        sys.exit(f"No metrics CSVs found matching: {base / pattern}")

    rows = []
    chosen_metric_name = None
    chosen_direction = None

    for p in paths:
        p = Path(p)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue

        metric_name, col, direction = find_best_metric_column(df)
        if not col:
            print(f"[WARN] No AUC/RMSE/MAE column found in {p.name}; skipping.")
            continue

        # lock in the first detected metric type so all bars are comparable
        if chosen_metric_name is None:
            chosen_metric_name, chosen_direction = metric_name, direction
        # if this file has a different metric type, try to find the locked one
        if metric_name != chosen_metric_name:
            # try to find the locked metric column in this df
            if chosen_metric_name == "ROC-AUC":
                col_alt = find_column(df, AUC_KEYS)
            elif chosen_metric_name == "RMSE":
                col_alt = find_column(df, RMSE_KEYS)
            else:  # MAE
                col_alt = find_column(df, MAE_KEYS)
            if col_alt:
                col = col_alt
            else:
                print(f"[WARN] {p.name} lacks {chosen_metric_name}; skipping to keep the plot consistent.")
                continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            print(f"[WARN] {col} column empty in {p.name}; skipping.")
            continue

        value = float(series.iloc[-1])  # last row = final/best
        model = extract_model_name(p.stem.replace("metrics_", "").replace(f"_{args.split}", ""))
        rows.append({"model": model, "value": value, "file": p.name})

    if not rows:
        sys.exit("No valid metric values found in matched CSVs.")

    plot_df = pd.DataFrame(rows).dropna().sort_values("model")

    # Value clipping only for ROC-AUC (0..1)
    if chosen_metric_name == "ROC-AUC":
        clipped = plot_df["value"].clip(0.0, 1.0)
        if not (clipped == plot_df["value"]).all():
            print("[WARN] Some AUC values were outside [0,1] and were clipped for plotting.")
        plot_df["value"] = clipped

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save summary CSV next to figure
    summary_csv = out_path.with_suffix(".csv")
    plot_df_out = plot_df.rename(columns={"value": chosen_metric_name.lower()})
    plot_df_out.to_csv(summary_csv, index=False)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(plot_df["model"], plot_df["value"])
    ylabel = chosen_metric_name
    plt.ylabel(ylabel)
    plt.title(f"{args.dataset.upper()} {chosen_metric_name} Comparison ({args.split})")
    # annotate bars
    for i, v in enumerate(plot_df["value"]):
        # place label sensibly given scale
        offset = 0.02 if chosen_metric_name == "ROC-AUC" else (0.03 * v if v > 0 else 0.02)
        plt.text(i, v + offset, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    # y-limits for ROC-AUC only
    if chosen_metric_name == "ROC-AUC":
        plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure ({chosen_metric_name}): {out_path}")
    print(f"Saved summary CSV: {summary_csv}")

if __name__ == "__main__":
    main()
