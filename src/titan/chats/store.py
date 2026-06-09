import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChatMeta:
    id: str
    title: str
    updated_at: str
    cwd: str = ""


class ChatStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._id = ""
        self._title = ""
        self._cwd = ""
        self._created_at = ""
        self.messages: list[dict] = []
        self.new()

    @property
    def _manifest_path(self) -> Path:
        return self.base_dir / "index.json"

    @property
    def title(self) -> str:
        return self._title

    def new(self) -> None:
        self._id = str(uuid.uuid4())
        self._title = ""
        self._cwd = os.getcwd()  # the project this chat belongs to
        self._created_at = self._now()
        self.messages = []

    def add(self, message: dict) -> None:
        self.messages.append(message)

    def save(self) -> None:
        if not self.messages:
            return
        updated = self._now()
        self._atomic_write(
            self.base_dir / f"{self._id}.json",
            {
                "id": self._id,
                "title": self._title,
                "cwd": self._cwd,
                "created_at": self._created_at,
                "updated_at": updated,
                "messages": self.messages,
            },
        )
        self._upsert_manifest(updated)

    def load(self, chat_id: str) -> None:
        data = json.loads((self.base_dir / f"{chat_id}.json").read_text())
        self._id = data["id"]
        self._title = data["title"]
        self._cwd = data.get("cwd", "")
        self._created_at = data["created_at"]
        self.messages = data["messages"]

    def list_chats(self, cwd: str | None = None) -> list[ChatMeta]:
        manifest = self._read_manifest() or self._rebuild_manifest()
        metas = [
            ChatMeta(cid, m["title"], m["updated_at"], m.get("cwd", ""))
            for cid, m in manifest.items()
            if cwd is None or m.get("cwd") == cwd
        ]
        return sorted(metas, key=lambda m: m.updated_at, reverse=True)

    def set_title(self, title: str) -> None:
        self._title = title

    def _rebuild_manifest(self) -> dict:
        manifest = {}
        for path in self.base_dir.glob("*.json"):
            if path.name == "index.json":
                continue
            try:
                d = json.loads(path.read_text())
                manifest[d["id"]] = {
                    "title": d["title"],
                    "cwd": d.get("cwd", ""),
                    "created_at": d["created_at"],
                    "updated_at": d["updated_at"],
                }
            except (json.JSONDecodeError, KeyError):
                continue  # skip corrupt/partial files
        self._atomic_write(self._manifest_path, manifest)
        return manifest

    def _read_manifest(self) -> dict:
        try:
            return json.loads(self._manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _upsert_manifest(self, updated_at: str) -> None:
        manifest = self._read_manifest()
        manifest[self._id] = {
            "title": self._title,
            "cwd": self._cwd,
            "created_at": self._created_at,
            "updated_at": updated_at,
        }
        self._atomic_write(self._manifest_path, manifest)

    def _atomic_write(self, path: Path, data) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)

    def _now(self) -> str:
        # ISO-8601 UTC. (Persistence legitimately needs real time; the agent loop doesn't.)
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
