from __future__ import annotations

from pathlib import Path


def resolve_context_path(dataset_root: str | Path, context: str) -> Path:
    """Resolve a WTQ context relative path to the table TSV path.

    The table storage is TSV, so whatever suffix is stored in the context,
    the resulting path always points to a file with the .tsv suffix.
    """
    root = Path(dataset_root)
    table = root / Path(context).with_suffix(".tsv")
    if not table.exists():
        raise FileNotFoundError(f"Table referenced by context not found: {context}; expected {table}")
    return table
