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
    ".obsidian",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
}

MAX_FILE_BYTES = 1_000_000

SKIP_FILENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
}


def chunk_file(file_path: Path, max_chunk_lines: int = 60) -> list[CodeChunk]:
    """Split a file into chunks. Markdown splits on headings; code on def/class boundaries."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.exception(f"Failed to chunk file {file_path}: {e}")
        return []

    language = LANG_MAP.get(file_path.suffix, "text")
    if language == "markdown":
        return _chunk_markdown(content, file_path, language, max_chunk_lines)
    return _chunk_code(content, file_path, language, max_chunk_lines)


def _chunk_code(
    content: str, file_path: Path, language: str, max_chunk_lines: int
) -> list[CodeChunk]:
    """Split code on top-level definition boundaries (def, class, function, ...)."""
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
                    content=chunk_content,
                    file_path=str(file_path),
                    start_line=current_chunk_start + 1,
                    end_line=i,
                    language=language,
                )
            )
            current_chunk_start = i

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


def _heading_level(line: str) -> int:
    """ATX heading level (1-6), or 0 if the line isn't a heading."""
    s = line.lstrip()
    hashes = len(s) - len(s.lstrip("#"))
    if 1 <= hashes <= 6 and s[hashes : hashes + 1] == " ":
        return hashes
    return 0


def _chunk_markdown(
    content: str,
    file_path: Path,
    language: str,
    max_chunk_lines: int,
    min_chunk_lines: int = 8,
) -> list[CodeChunk]:
    """Split markdown on heading boundaries, prepending the filename + heading trail."""
    lines = content.split("\n")
    chunks: list[CodeChunk] = []
    start = 0

    def context_prefix(idx: int) -> str:
        stack: list[tuple[int, str]] = []
        for j in range(idx):
            level = _heading_level(lines[j])
            if level == 0:
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, lines[j].lstrip("# ").strip()))

        own_level = _heading_level(lines[idx]) if idx < len(lines) else 0
        while own_level and stack and stack[-1][0] >= own_level:
            stack.pop()

        parts = [file_path.stem]
        for _, text in stack:
            parts.append(text)
        return " > ".join(parts)

    def flush(end: int) -> None:
        body = "\n".join(lines[start:end]).strip()
        if not body:
            return
        text = f"{context_prefix(start)}\n\n{body}"
        chunks.append(
            CodeChunk(
                content=text,
                file_path=str(file_path),
                start_line=start + 1,
                end_line=end,
                language=language,
            )
        )

    for i in range(1, len(lines)):
        section_len = i - start
        if (_heading_level(lines[i]) and section_len >= min_chunk_lines) or (
            section_len >= max_chunk_lines
        ):
            flush(i)
            start = i
    flush(len(lines))

    return chunks


def collect_files(directory: Path, suffixes: set[str] | None = None) -> list[Path]:
    """Recursively collect all indexable files."""
    allowed = suffixes or set(LANG_MAP)
    try:
        files = []
        for path in directory.rglob("*"):
            if any(part in IGNORE_PATTERMS for part in path.parts):
                continue
            if not path.is_file() or path.suffix not in allowed:
                continue
            if path.name in SKIP_FILENAMES:
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                logger.info(f"Skipping large file: {path}")
                continue
            files.append(path)
        return files
    except Exception as e:
        logger.exception(f"Error finding files in directory {directory}: {e}")
        return []
