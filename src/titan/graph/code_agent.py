from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from titan.tools.filesystem import list_directory, read_file, write_file
from titan.tools.rag import search_codebase
from titan.tools.shell import run_command

code_llm = ChatOpenAI(
    base_url="http://localhost:8002/v1",  # llama.cpp on GPU 1
    api_key=SecretStr("not-needed"),
    model="code",
    temperature=0,
)

code_tools = [read_file, write_file, list_directory, run_command, search_codebase]

CODE_PROMPT = """You are Cody, an expert coding agent. You write clean, correct code.
You have these tools available — ALWAYS use them instead of writing code snippets:
- read_file: Read file contents
- write_file: Write content to a file
- list_directory: List files in a directory
- run_command: Run shell commands
- search_codebase: Search the indexed codebase by meaning

Never write Python code to read files or list directories. Use your tools.
Always read existing code before modifying it. Explain what you changed and why.
Be concise and direct in your responses."""

# create_react_agent is a LangGraph helper that builds the tool-calling loop
code_agent = create_agent(
    code_llm,
    code_tools,
    system_prompt=CODE_PROMPT,
)
