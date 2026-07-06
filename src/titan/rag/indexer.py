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
from titan.rag.store import (
    VAULT_COLLECTION,
    client,
    codebase_collection_name,
    find_projects,
)


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
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True,
            device=resolved,
        )
    model.max_seq_length = 1024
    _EMBEDDING_MODELS[resolved] = model
    return model


def _empty_device_cache(device_type: str) -> None:
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()


def _chunk_dir(
    directory: Path,
    suffixes: set[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list:
    files = collect_files(directory, suffixes)
    chunks = []
    for i, file in enumerate(files):
        if on_progress and (i % 10 == 0 or i == len(files) - 1):
            on_progress(f"Chunking {directory.name}... ({i + 1}/{len(files)})")
        rel = file.relative_to(directory)
        for chunk in chunk_file(file):
            chunk.file_path = str(rel)
            chunks.append(chunk)
    return chunks


def collect_and_chunk(
    directory: Path,
    collection_type: str,
    on_progress: Callable[[str], None] | None = None,
) -> list[tuple[str, str, list]]:
    if collection_type == "vault":
        vault = directory.expanduser().resolve()
        chunks = _chunk_dir(vault, {".md"}, on_progress)
        return [(VAULT_COLLECTION, str(vault), chunks)]

    vault_dir = os.environ.get("OBSIDIAN_VAULT_DIR")
    vault_resolved = Path(vault_dir).expanduser().resolve() if vault_dir else None
    entries = []
    for project in find_projects(directory):
        proj = project.resolve()
        if vault_resolved and proj == vault_resolved:
            if on_progress:
                on_progress(f"Skipping vault (use /index vault): {project.name}")
            continue
        collection_name = codebase_collection_name(project)
        chunks = _chunk_dir(proj, None, on_progress)
        entries.append((collection_name, str(proj), chunks))
    return entries


def _embed_entry(
    collection_name: str,
    project_path: str,
    chunks: list,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    if not chunks:
        return 0

    vdb_client = client()
    try:
        vdb_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = vdb_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "project_path": project_path},
    )

    embeddings = create_embeddings()
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
        if on_progress:
            on_progress(
                f"Embedding {Path(project_path).name}... ({batch_idx + 1}/{total_batches} batches)"
            )
        batch = chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]
        vectors = embeddings.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=cast(Embedding, vectors),
            documents=texts,
            metadatas=[
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                }
                for chunk in batch
            ],
        )
        _empty_device_cache(embeddings.device.type)
    return len(chunks)


def embed_entries(
    entries, on_progress: Callable[[str], None] | None = None
) -> dict[str, int]:
    embedded_entries = {}
    for name, path, chunks in entries:
        embed_entry = _embed_entry(name, path, chunks, on_progress)
        embedded_entries[path] = embed_entry
    return embedded_entries
