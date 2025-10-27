# scripts/compare_roc_auc.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bbbp", help="folder with metrics CSVs")
    ap.add_argument("--out", default="results/bbbp/figures/roc_auc_comparison.png")
    args = ap.parse_args()

    base = Path(args.dir)
    files = [
        base / "metrics_bbbp_gcn_test.csv",
        base / "metrics_bbbp_gin_test.csv",
        base / "metrics_bbbp_gat_test.csv",
    ]
    rows = []
    for f in files:
        if f.exists():
            df = pd.read_csv(f)
            auc = df.get("auc", pd.Series([None])).iloc[0]
            rows.append({"model": f.stem.replace("metrics_","").replace("_test",""), "auc": auc})
    if not rows:
        raise SystemExit("No metrics CSVs found in results/bbbp")

    outdir = Path(args.out).parent; outdir.mkdir(parents=True, exist_ok=True)
    plot_df = pd.DataFrame(rows).dropna()
    plt.figure()
    plt.bar(plot_df["model"], plot_df["auc"])
    plt.ylabel("ROC-AUC"); plt.title("BBBP ROC-AUC Comparison (test)")
    for i, v in enumerate(plot_df["auc"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout(); plt.savefig(args.out, dpi=200); plt.close()
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
