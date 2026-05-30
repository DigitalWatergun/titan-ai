from pydantic import BaseModel

from titan.tools.filesystem import (
    ListDirectoryInput,
    ReadFileInput,
    WriteFileInput,
    list_directory,
    read_file,
    write_file,
)
from titan.tools.rag import SearchCodebaseInput, search_codebase
from titan.tools.search import WebSearchInput, web_search
from titan.tools.shell import RunCommandInput, run_command


def pydantic_to_tool_schema(name: str, model: type[BaseModel]) -> dict:
    """Convert a Pydantic input model to an OpenAI tool-use schema."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (model.__doc__ or "").strip(),
            "parameters": schema,
        },
    }


TOOL_FUNCS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "search_codebase": search_codebase,
    "web_search": web_search,
    "run_command": run_command,
}

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "read_file": ReadFileInput,
    "write_file": WriteFileInput,
    "list_directory": ListDirectoryInput,
    "search_codebase": SearchCodebaseInput,
    "web_search": WebSearchInput,
    "run_command": RunCommandInput,
}

TOOL_SCHEMAS = [
    pydantic_to_tool_schema(name, TOOL_INPUT_MODELS[name]) for name in TOOL_FUNCS
]
