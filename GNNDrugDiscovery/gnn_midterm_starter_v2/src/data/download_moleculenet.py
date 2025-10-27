
import os
from torch_geometric.datasets import MoleculeNet

DATASETS = ["Tox21", "BBBP", "ESOL"]

def main():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "moleculenet")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    for name in DATASETS:
        print(f"Downloading {name} to {root} ...")
        _ = MoleculeNet(root=root, name=name)  # triggers download/process
        print(f"✓ {name} ready.")
    print("All datasets downloaded.")

if __name__ == "__main__":
    main()
