import hashlib
from pathlib import Path

import chromadb

CHROMA_DIR = Path.home() / ".titan" / "chromadb"


def client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def codebase_collection_name(project_dir: Path) -> str:
    h = hashlib.sha1(str(project_dir.resolve()).encode()).hexdigest()[:12]
    return f"codebase__{h}"


def find_projects(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / ".git").exists():
        return [root]
    repos = sorted(p.parent for p in root.glob("*/.git"))
    return repos or [root]
