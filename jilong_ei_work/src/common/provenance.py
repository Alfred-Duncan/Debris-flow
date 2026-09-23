"""Explicit provenance resolution for formal frozen-model execution."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming content hash; checkpoint files are never read at once."""
    target=Path(path)
    if not target.is_file():raise RuntimeError(f'PROVENANCE_FILE_MISSING:{target}')
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda:stream.read(chunk_size),b''):digest.update(chunk)
    return digest.hexdigest()

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
