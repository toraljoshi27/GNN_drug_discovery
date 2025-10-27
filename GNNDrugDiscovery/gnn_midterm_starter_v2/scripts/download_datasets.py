#!/usr/bin/env python3
import hashlib, os, sys, gzip, shutil, urllib.request, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    # Tox21 (12-task classification): gzipped CSV
    # Source URL and MD5 seen in multiple toolkits (DeepChem / TorchDrug docs).
    "tox21": {
        "url": "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "file": "tox21.csv.gz",
        "md5":  "2882d69e70bba0fec14995f26787cc25",   # verify .gz
        "post": "ungzip_to_csv"
    },
    # BBBP (binary classification): plain CSV
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "file": "BBBP.csv",
        "md5":  None,  # MD5 not published in the same docs we’re using
        "post": None
    },
    # ESOL / Delaney (regression): plain CSV
    "esol": {
        "url": "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "file": "delaney-processed.csv",
        "md5":  "0c90a51668d446b9e3ab77e67662bd1c",   # verify .csv
        "post": None
    },
}

def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url, out_path):
    print(f"↘ downloading: {url}\n→ to: {out_path}")
    urllib.request.urlretrieve(url, out_path)

def ungzip_to_csv(gz_path, csv_path):
    print(f"🗜 extracting: {gz_path} → {csv_path}")
    with gzip.open(gz_path, "rb") as fin, open(csv_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)

def fetch_one(name: str):
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")

    spec = DATASETS[name]
    url  = spec["url"]
    file = spec["file"]
    path = ROOT / file

    # 1) download (skip if exists and matches md5)
    if path.exists():
        print(f"✔ found existing: {path}")
    else:
        download(url, str(path))

    # 2) md5 (if available)
    if spec["md5"] is not None:
        calc = md5sum(path)
        if calc != spec["md5"]:
            raise SystemExit(
                f"MD5 mismatch for {path}\n expected: {spec['md5']}\n got:      {calc}\n"
                "Delete the file and re-run."
            )
        else:
            print(f"🔐 md5 ok: {calc}")

    # 3) postprocess (tox21 ungzip)
    if spec["post"] == "ungzip_to_csv":
        csv_path = ROOT / "tox21.csv"
        if not csv_path.exists():
            ungzip_to_csv(path, csv_path)
        else:
            print(f"✔ extracted csv exists: {csv_path}")

    print(f"✅ {name} ready under {ROOT}")

def main(args):
    if len(args) >= 2:
        for name in args[1:]:
            fetch_one(name.lower())
    else:
        # Download all if no args
        for name in DATASETS:
            fetch_one(name)

if __name__ == "__main__":
    main(sys.argv)
