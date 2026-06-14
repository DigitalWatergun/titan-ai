import contextlib
import io
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
import tqdm.std
from chromadb.api.types import Embedding
from sentence_transformers import SentenceTransformer

from titan.rag.chunker import chunk_file, collect_files
from titan.rag.store import client, codebase_collection_name, find_projects


def _patch_tqdm():
    """Prevent tqdm from creating multiprocessing locks.
    tqdm's __new__ always calls create_mp_lock() even when disabled,
    which spawns a resource tracker subprocess. Inside Textual's TUI,
    the file descriptors are in a state that causes this spawn to fail
    with 'bad value(s) in fds_to_keep' on macOS.
    """

    lock_class = getattr(tqdm.std, "TqdmDefaultWriteLock", None)
    if lock_class is not None:
        lock_class.create_mp_lock = classmethod(
            lambda cls: setattr(cls, "mp_lock", None)
        )


_EMBEDDING_MODELS: dict[str, SentenceTransformer] = {}


def _embedding_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_embeddings(device: str | None = None):
    resolved = device or _embedding_device()
    cached = _EMBEDDING_MODELS.get(resolved)
    if cached is not None:
        return cached

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    _patch_tqdm()

    with (
        warnings.catch_warnings(),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        _EMBEDDING_MODELS[resolved] = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True,
            device=resolved,
        )
    return _EMBEDDING_MODELS[resolved]


def _empty_device_cache(device_type: str) -> None:
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def _index_directory(
    directory_path: str,
    collection_name: str,
    on_progress: Callable[[str], None] | None = None,
    suffixes: set[str] | None = None,
) -> int:
    """Index a directory into a named ChromaDB collection. Returns number of chunks indexed."""
    directory = Path(directory_path).resolve()
    vdb_client = client()

    # Delete existing collection if re-indexing
    try:
        vdb_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = vdb_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "project_path": str(directory)},
    )

    if on_progress:
        on_progress("Loading embedding model...")

    embeddings = create_embeddings()
    files = collect_files(directory, suffixes)
    total_files = len(files)

    if on_progress:
        on_progress(f"Found {total_files} files to index")

    all_chunks = []
    for i, file_path in enumerate(files):
        if on_progress and (i % 10 == 0 or i == total_files - 1):
            on_progress(f"Chunking files... ({i + 1}/{total_files})")
        rel_path = file_path.relative_to(directory)
        chunks = chunk_file(file_path)
        for chunk in chunks:
            chunk.file_path = str(rel_path)  # Store relative path
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # Embed and store in batches (ChromaDB has batch limits)
    batch_size = 50
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(all_chunks), batch_size)):
        if on_progress:
            on_progress(f"Embedding... ({batch_idx + 1}/{total_batches} batches)")
        batch = all_chunks[i : i + batch_size]
        texts = [c.content for c in batch]
        vectors = embeddings.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=cast(Embedding, vectors),
            documents=texts,
            metadatas=[
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "language": c.language,
                }
                for c in batch
            ],
        )
        _empty_device_cache(embeddings.device.type)

    return len(all_chunks)


def index_projects(root: str, on_progress=None) -> dict[str, int]:
    results = {}
    vault_dir = os.environ.get("OBSIDIAN_VAULT_DIR")
    vault_resolved = Path(vault_dir).expanduser().resolve() if vault_dir else None
    for project in find_projects(Path(root)):
        if vault_resolved and project.resolve() == vault_resolved:
            if on_progress:
                on_progress(f"Skipping vault (use /index vault): {project.name}")
            continue
        if on_progress:
            on_progress(f"Indexing {project.name}...")
        results[str(project)] = _index_directory(
            str(project),
            collection_name=codebase_collection_name(project),
            on_progress=on_progress,
        )
    return results


def index_vault(vault_dir, on_progress=None) -> dict[str, int]:
    results = {}
    results["vault"] = _index_directory(
        str(vault_dir),
        collection_name="vault",
        on_progress=on_progress,
        suffixes={".md"},
    )
    return results
