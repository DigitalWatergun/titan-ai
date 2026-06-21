import hashlib
from pathlib import Path

import chromadb

CHROMA_DIR = Path.home() / ".titan" / "chromadb"
CODEBASE_PREFIX = "codebase__"
VAULT_COLLECTION = "vault"
CHATS_COLLECTION = "chats"


def client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def codebase_collection_name(project_dir: Path) -> str:
    h = hashlib.sha1(str(project_dir.resolve()).encode()).hexdigest()[:12]
    return f"{CODEBASE_PREFIX}{h}"


def codebase_collections():
    return [
        col
        for col in client().list_collections()
        if col.name.startswith(CODEBASE_PREFIX)
    ]


def chats_collection():
    return client().get_or_create_collection(
        name=CHATS_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def find_projects(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / ".git").exists():
        return [root]
    repos = sorted(p.parent for p in root.glob("*/.git"))
    return repos or [root]
