from pathlib import Path

from pydantic import BaseModel, Field

from titan.rag.indexer import create_embeddings
from titan.rag.store import client, codebase_collection_name


class SearchCodebaseInput(BaseModel):
    """Search the indexed codebase for code relevant to the query.
    Use this to find where things are implemented, understand patterns, etc."""

    query: str = Field(description="Query string to find matching code chunks")


def search_codebase(args: SearchCodebaseInput) -> str:
    try:
        vdb_client = client()
        collection = vdb_client.get_collection(codebase_collection_name(Path.cwd()))
    except Exception:
        return "Error: this project isn't indexed yet. Run /index first."

    n_results = 5

    embeddings = create_embeddings()
    query_vector = embeddings.encode(args.query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["metadatas"] or not results["distances"]:
        return "No relevant code found."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist  # cosine distance to similarity
        output.append(
            f"--- {meta['file_path']} (lines {meta['start_line']}-{meta['end_line']}) "
            f"[similarity: {similarity:.2f}] ---\n{doc}\n"
        )

    return "\n".join(output) if output else "No relevant code found."
