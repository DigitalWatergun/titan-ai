import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Embedding
from langchain_huggingface import HuggingFaceEmbeddings

from titan.rag.chunker import chunk_file, collect_files


def create_embeddings():
    """Create the embedding function."""
    # Suppress at file descriptor level to catch all output,
    # including HuggingFace trust_remote_code warnings that
    # bypass Python's sys.stderr
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    old_stdout_fd = os.dup(1)
    os.dup2(devnull_fd, 2)
    os.dup2(devnull_fd, 1)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True, "device": "cpu"},
        )
    finally:
        os.dup2(old_stderr_fd, 2)
        os.dup2(old_stdout_fd, 1)
        os.close(devnull_fd)
        os.close(old_stderr_fd)
        os.close(old_stdout_fd)
    return embeddings


def index_codebase(
    directory_path: str,
    collection_name: str = "codebase",
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Index a directory into ChromaDB. Returns number of chunks indexed."""
    directory = Path(directory_path).resolve()

    # ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(directory / ".titan" / "chromadb"))

    # Delete existing collection if re-indexing
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    if on_progress:
        on_progress("Loading embedding model...")

    embeddings = create_embeddings()
    files = collect_files(directory)
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
        vectors = embeddings.embed_documents(texts)

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

    return len(all_chunks)
