import json
from collections.abc import Callable
from pathlib import Path

from titan.rag.indexer import create_embeddings
from titan.rag.store import chats_collection

MAX_CHARS = 3500  # ~1024-token embed budget (≈3.5 chars/token)
_current_chat_id: str | None = None


def _segment_turns(messages: list[dict]) -> list[dict]:
    turns, user = [], None
    for message in messages:
        role, content = message.get("role"), (message.get("content") or "").strip()
        if role == "user" and content:
            user = content
        elif role == "assistant" and not message.get("tool_calls") and content:
            turns.append({"user": user or "", "assistant": content})
            user = None
    return turns


def _split_if_oversized(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, buff = [], ""
    for paragraph in text.split("\n\n"):
        if buff and len(buff) + len(paragraph) > MAX_CHARS:
            parts.append(buff.strip())
            buff = ""
        buff += paragraph + "\n\n"
    if buff.strip():
        parts.append(buff.strip())
    return parts


def set_current_chat_id(chat_id: str | None) -> None:
    global _current_chat_id
    _current_chat_id = chat_id


def get_current_chat_id() -> str | None:
    return _current_chat_id


def index_chat(
    chat_id: str,
    title: str,
    cwd: str,
    messages: list[dict],
    updated_at: str = "",
) -> int:
    ids, docs, metas = [], [], []
    for i_turn, turn in enumerate(_segment_turns(messages)):
        text = f"user: {turn['user']}\nassistant: {turn['assistant']}"
        for i_piece, piece in enumerate(_split_if_oversized(text)):
            ids.append(f"{chat_id}__{i_turn}__{i_piece}")
            docs.append(piece)
            metas.append(
                {
                    "chat_id": chat_id,
                    "title": title,
                    "cwd": cwd,
                    "turn_index": i_turn,
                    "sub_index": i_piece,
                    "updated_at": updated_at,
                }
            )

    if not docs:
        return 0

    vectors = create_embeddings().encode(docs).tolist()
    chats_collection().upsert(
        ids=ids, embeddings=vectors, documents=docs, metadatas=metas
    )
    return len(docs)


def index_all_chats(
    base_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    paths = [p for p in base_dir.glob("*.json") if p.name != "index.json"]
    total = 0
    for i, path in enumerate(paths):
        if on_progress:
            on_progress(f"Indexing chats... ({i + 1}/{len(paths)})")
        chat = json.loads(path.read_text())
        total += index_chat(
            chat["id"],
            chat["title"],
            chat.get("cwd", ""),
            chat["messages"],
            chat.get("updated_at", ""),
        )
    return total
