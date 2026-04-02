from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Embedding
from langchain_ollama import OllamaEmbeddings
from rich.progress import track

from titan.rag.chunker import chunk_file, collect_files


def create_embeddings():
    """Create the embedding function using Ollama."""
    return OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")


def index_codebase(directory_path: str, collection_name: str = "codebase") -> int:
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

    embeddings = create_embeddings()
    files = collect_files(directory)

    all_chunks = []
    for file_path in track(files, description="Chunking files..."):
        rel_path = file_path.relative_to(directory)
        chunks = chunk_file(file_path)
        for chunk in chunks:
            chunk.file_path = str(rel_path)  # Store relative path
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # Embed and store in batches (ChromeDB has batch limits)
    batch_size = 50
    for i in track(range(0, len(all_chunks), batch_size), description="Embedding..."):
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
