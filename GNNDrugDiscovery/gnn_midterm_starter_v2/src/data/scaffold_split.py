
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def _scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def scaffold_split(smiles_list, seed=42, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    """
    Deterministic Bemis-Murcko scaffold split.
    - Groups molecules by scaffold.
    - Sorts groups by descending size, then lexicographically by scaffold string.
    - Fills train/val/test buckets in that order.
    Returns (train_idx, val_idx, test_idx) as lists of indices.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    # Build scaffold -> list of indices map
    scaffold2idx = {}
    for i, smi in enumerate(smiles_list):
        scaf = _scaffold(smi)
        scaffold2idx.setdefault(scaf, []).append(i)
    # Sort scaffold groups by size (desc), then by scaffold key for determinism
    groups = sorted(scaffold2idx.values(), key=lambda g: (-len(g), tuple(sorted(g))))
    n = len(smiles_list)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)
    # If rounding left some indices unassigned, place them in test
    leftover = set(range(n)) - set(train_idx) - set(val_idx) - set(test_idx)
    test_idx.extend(sorted(list(leftover)))
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)
