import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CodeChunk(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str


# File extensiosn to index
LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".sh": "bash",
}

# Files/directories to skip
IGNORE_PATTERMS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
}


def chunk_file(file_path: Path, max_chunk_lines: int = 60) -> list[CodeChunk]:
    """Split a file into chunks, trying to break on function/class boundaries"""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.exception(f"Failed to chunk file {file_path}: {e}")
        return []

    suffix = file_path.suffix
    language = LANG_MAP.get(suffix, "text")
    lines = content.split("\n")

    if len(lines) <= max_chunk_lines:
        return [
            CodeChunk(
                content=content,
                file_path=str(file_path),
                start_line=1,
                end_line=len(lines),
                language=language,
            )
        ]

    # Split on top-level definitions (def, class, function, etc.)
    chunks = []
    current_chunk_start = 0
    boundary_keywords = ["def ", "class ", "async def ", "function ", "export "]

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        is_boundary = any(stripped.startswith(kw) for kw in boundary_keywords)
        chunk_too_long = (i - current_chunk_start) >= max_chunk_lines

        if is_boundary and i > current_chunk_start and chunk_too_long:
            chunk_content = "\n".join(lines[current_chunk_start:i])
            chunks.append(
                CodeChunk(
                    content=content,
                    file_path=str(file_path),
                    start_line=current_chunk_start + 1,
                    end_line=i,
                    language=language,
                )
            )
            current_chunk_start = i

    # Don't forget the last chunk
    if current_chunk_start < len(lines):
        chunk_content = "\n".join(lines[current_chunk_start:])
        chunks.append(
            CodeChunk(
                content=chunk_content,
                file_path=str(file_path),
                start_line=current_chunk_start + 1,
                end_line=len(lines),
                language=language,
            )
        )

    return chunks


def collect_files(directory: Path) -> list[Path]:
    """Recursively collect all indexable files."""
    try:
        files = []
        for path in directory.rglob("*"):
            if any(part in IGNORE_PATTERMS for part in path.parts):
                continue
            if path.is_file() and path.suffix in LANG_MAP:
                files.append(path)
        return files
    except Exception as e:
        logger.exception(f"Error finding files in directory {directory}: {e}")
        return []
