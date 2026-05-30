import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReadFileInput(BaseModel):
    """Read the contents of a file and return it as a string.
    Use this when you need to see what's in a file."""

    file_path: str = Field(description="Path to the file to read")


class WriteFileInput(BaseModel):
    """Write content to a file. Creates the file if it doesn't exist.
    Use this when you need to create or modify a file."""

    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class ListDirectoryInput(BaseModel):
    """List all files and subdirectories in a directory.
    Use this to explore the structure of a project."""

    directory_path: str = Field(description="Path to the directory to list")


def read_file(args: ReadFileInput) -> str:
    try:
        path = Path(args.file_path)
        if not path.exists():
            return f"Error: file {args.file_path} not found"
        if not path.is_file():
            return f"Error: {args.file_path} is not a file"
        return path.read_text()
    except Exception as e:
        logger.exception(f"Failed to read {args.file_path}")
        return f"Error reading {args.file_path}: {e}"


def write_file(args: WriteFileInput) -> str:
    try:
        path = Path(args.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content)
        return f"Successfully wrote to {args.file_path}"
    except PermissionError:
        logger.exception(f"Permisison denied writing to {args.file_path}")
        return f"Error: Permission denied writing to {args.file_path}"
    except Exception as e:
        logger.exception(f"Failed to write to {args.file_path}")
        return f"Error writing to {args.file_path}: {e}"


def list_directory(args: ListDirectoryInput) -> str:
    try:
        path = Path(args.directory_path)
        if not path.exists():
            return f"Error: Directory {args.directory_path} not found"
        entries = sorted(path.iterdir())
        result = []
        for entry in entries:
            prefix = "[DIR]" if entry.is_dir() else "[FILE]"
            result.append(f"{prefix} {entry.name}")
        return "\n".join(result) if result else "Empty directory"
    except Exception as e:
        logger.exception(f"Failed to list {args.directory_path}")
        return f"Error listing {args.directory_path}: {e}"
