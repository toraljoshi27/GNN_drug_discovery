# Select between random/scaffold split based on config
def get_splitter(name: str):
    if name == 'scaffold':
        from .data.scaffold_split import scaffold_split
        return scaffold_split
    elif name == 'random':
        return None  # implement random split
    else:
        raise ValueError(f"Unknown split: {name}")
