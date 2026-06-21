from pathlib import Path

from pydantic import BaseModel, Field

from titan.rag.indexer import create_embeddings
from titan.rag.store import (
    VAULT_COLLECTION,
    client,
    codebase_collections,
)


class SearchCodebaseInput(BaseModel):
    """Search the indexed codebase for code relevant to the query.
    Use this to find where things are implemented, understand patterns, etc."""

    query: str = Field(description="Query string to find matching code chunks")
    scope: str = Field(
        "current",
        description=(
            "'current' = this working directory's project (default). "
            "'all' = every indexed project. "
            "Or a project name (e.g. 'proj2') to target a specific one. "
            "Widen beyond 'current' when the user asks about a different project."
        ),
    )


class SearchVaultInput(BaseModel):
    """Search the Obsidian vault notes for information relevant to the query.
    Use this to find context, information, documentation, etc.
    """

    query: str = Field(description="Query string to find matching vault chunks")


def _resolve_collections(scope: str, cwd: Path) -> list | str:
    collections = codebase_collections()

    if scope == "all":
        return collections

    if scope and scope != "current":
        matches = [
            col
            for col in collections
            if scope in str((col.metadata or {}).get("project_path", ""))
        ]

        if len(matches) > 1:
            names = ", ".join(
                Path(str((col.metadata or {}).get("project_path", ""))).name
                for col in matches
            )
            return f"'{scope}' matches multiple projects: {names}. Be more specific"
        return matches

    here = str(cwd.resolve())
    exact = [
        col
        for col in collections
        if str((col.metadata or {}).get("project_path", "")) == here
    ]
    if exact:
        return exact

    return [
        col
        for col in collections
        if str((col.metadata or {}).get("project_path", "")).startswith(here + "/")
    ]


def _project_label(project_path: str, cwd: Path) -> str:
    p = Path(project_path)
    if p == cwd:
        return p.name
    try:
        return str(p.relative_to(cwd))
    except ValueError:
        return str(p)


def search_codebase(args: SearchCodebaseInput) -> str:
    cwd = Path.cwd()
    resolved = _resolve_collections(args.scope, cwd)
    if isinstance(resolved, str):
        return resolved
    collections = resolved
    if not collections:
        return "Error: nothing indexed for this scope. Run /index first."

    try:
        embeddings = create_embeddings()
        query_vector = embeddings.encode(args.query).tolist()

        hits = []
        for collection in collections:
            project_path = str((collection.metadata or {}).get("project_path", ""))
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append((dist, project_path, meta, doc))
    except Exception as e:
        return f"Error: search failed: {e}"

    if not hits:
        return "No relevant code found."

    hits.sort(key=lambda h: h[0])
    multi = len(collections) > 1

    output = []
    for dist, project_path, meta, doc in hits[:8]:
        similarity = 1 - dist  # cosine distance to similarity
        prefix = f"[{_project_label(project_path, cwd)}] " if multi else ""
        output.append(
            f"--- {prefix}{meta['file_path']}"
            f"(lines {meta['start_line']}-{meta['end_line']}) "
            f"[similarity: {similarity:.2f}] ---\n{doc}\n"
        )

    return "\n".join(output)


def search_vault(args: SearchVaultInput) -> str:
    try:
        vdb_client = client()
        collection = vdb_client.get_collection(VAULT_COLLECTION)
    except Exception:
        return "Error: Obsidian vault hasn't been indexed yet. Run /index vault first."

    vault_root = Path(str((collection.metadata or {}).get("project_path", "")))

    embeddings = create_embeddings()
    query_vector = embeddings.encode(args.query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["metadatas"] or not results["distances"]:
        return "No relevant notes found."

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist  # cosine distance to similarity
        source = vault_root / str(meta["file_path"])
        output.append(
            f"--- {source} (lines {meta['start_line']}-{meta['end_line']}) "
            f"[similarity: {similarity:.2f}] ---\n{doc}\n"
        )

    return "\n".join(output) if output else "No relevant notes found."
