"""Explicit provenance resolution for formal frozen-model execution."""
from __future__ import annotations
import subprocess
from pathlib import Path

def resolve_code_sha(explicit: str | None = None, start: str | Path | None = None, require: bool = False) -> str:
    if explicit:return str(explicit).strip()
    root=Path(start or Path.cwd()).resolve()
    for directory in (root,*root.parents):
        if (directory/'.git').exists():
            try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,text=True).strip()
            except Exception:pass
        source=directory/'SOURCE_CODE_COMMIT.txt'
        if source.exists() and source.read_text().strip():return source.read_text().strip()
    if require:raise RuntimeError('SOURCE_CODE_SHA_UNKNOWN')
    return 'UNKNOWN'
