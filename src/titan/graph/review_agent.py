from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from titan.tools.filesystem import list_directory, read_file

review_llm = ChatOpenAI(
    base_url="http://localhost:8004/v1",  # llama.cpp on GPU 3
    api_key=SecretStr("not-needed"),
    model="review",
    temperature=0.1,
)

review_tools = [read_file, list_directory]

REVIEW_PROMPT = """You are Mark, an expert code review agent. You analyze code for bugs, security issues, and improvements.
  You have these tools available — ALWAYS use them instead of writing code snippets:
  - read_file: Read file contents
  - list_directory: List files in a directory

  Never write Python code to read files or list directories. Use your tools.
  When reviewing code:
  - Read the relevant files before giving feedback
  - Point out specific lines and explain what's wrong
  - Suggest concrete fixes, not vague advice
  - Check for security issues, error handling, and edge cases
  Be concise and direct in your responses."""

review_agent = create_agent(
    review_llm,
    review_tools,
    system_prompt=REVIEW_PROMPT,
)
