from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from titan.tools.registry import TOOL_FUNCS, TOOL_INPUT_MODELS, TOOL_SCHEMAS


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    tool_funcs: dict[str, Callable[[BaseModel], str]]
    tool_input_models: dict[str, type[BaseModel]]
    tool_schemas: list[dict]
    model_name: str = "titan"


MAIN_SYSTEM_PROMPT = """You are Titan, an expert coding assistant with access to the local filesystem and the web.

Tools available:
- read_file: Read file contents
- write_file: Write or create files
- list_directory: List directory contents (always start with ".")
- run_command: Run shell commands
- search_codebase: Search the indexed codebase by meaning
- web_search: Search the web for current information

Rules:
- ALWAYS use tools instead of writing code snippets to read/write files
- Always start by listing "." — never guess paths
- Read existing code before modifying it
- When researching, use web_search for current info and search_codebase for project-specific questions
- Be concise and direct
"""

MAIN_AGENT = Agent(
    name="main",
    system_prompt=MAIN_SYSTEM_PROMPT,
    tool_funcs=TOOL_FUNCS,
    tool_input_models=TOOL_INPUT_MODELS,
    tool_schemas=TOOL_SCHEMAS,
)
