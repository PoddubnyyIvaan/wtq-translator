from __future__ import annotations

from pathlib import Path


def resolve_context_path(dataset_root: str | Path, context: str) -> Path:
    """Resolve WTQ context relative path without assuming CSV semantics.

    If the exact path is absent and context ends with .csv, also try .tsv.
    This keeps compatibility with older WTQ metadata while the table storage is TSV.
    """
    root = Path(dataset_root)
    rel = Path(context)
    candidates = [root / rel]
    if rel.suffix.lower() == ".csv":
        candidates.append(root / rel.with_suffix(".tsv"))
    elif rel.suffix.lower() != ".tsv":
        candidates.append(root / rel.with_suffix(".tsv"))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Table referenced by context not found: {context}; tried {candidates}")
