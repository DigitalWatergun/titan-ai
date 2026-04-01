import logging
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReadFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to read")


class WriteFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class ListDirectoryInput(BaseModel):
    directory_path: str = Field(description="Path to the directory to list")


@tool(args_schema=ReadFileInput)
def read_file(file_path: str) -> str:
    """Read the contents of a file and return it as as tring.
    Use this when you need to see what's in a file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: file {file_path} not found"
        if not path.is_file():
            return f"Error: {file_path} is not a file"
        return path.read_text()
    except Exception as e:
        logger.exception(f"Failed to read {file_path}")
        return f"Error reading {file_path}: {e}"


@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist.
    Use this when you need to create or modify a file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Successfully wrote to {file_path}"
    except PermissionError:
        logger.exception(f"Permisison denied writing to {file_path}")
        return f"Error: Permission denied writing to {file_path}"
    except Exception as e:
        logger.exception(f"Failed to write to {file_path}")
        return f"Error writing to {file_path}: {e}"


@tool(args_schema=ListDirectoryInput)
def list_directory(directory_path: str) -> str:
    """List all files and subdirectories in a directoory.
    Use this to explore the structure of a project."""
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory {directory_path} not found"
        entries = sorted(path.iterdir())
        result = []
        for entry in entries:
            prefix = "[DIR]" if entry.is_dir() else "[FILE]"
            result.append(f"{prefix} {entry.name}")
        return "\n".join(result) if result else "Empty directory"
    except Exception as e:
        logger.exception(f"Failed to list {directory_path}")
        return f"Error listing {directory_path}: {e}"
