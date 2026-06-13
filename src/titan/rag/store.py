import hashlib
from pathlib import Path

import chromadb

CHROMA_DIR = Path.home() / ".titan" / "chromadb"


def client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def codebase_collection_name(project_dir: Path) -> str:
    h = hashlib.sha1(str(project_dir.resolve()).encode()).hexdigest()[:12]
    return f"codebase__{h}"
