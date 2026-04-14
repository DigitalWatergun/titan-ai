from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from titan.tools.filesystem import list_directory, read_file
from titan.tools.rag import search_codebase
from titan.tools.search import web_search

research_llm = ChatOpenAI(
    base_url="http://localhost:8003/v1",  # llama.cpp on GPU 2
    api_key=SecretStr("not-needed"),
    model="research",
    temperature=0.1,
)

research_tools = [read_file, list_directory, search_codebase, web_search]

RESEARCH_PROMPT = """You are Paige, a research agent. You find information and answer questions.
You can answer general knowledge questions directly from your own knowledge.
You have these tools available:
- search_codebase: Search the indexed codebase by meaning
- web_search: Search the web for current information or documentation
- read_file: Read file contents for more detail on search results
- list_directory: List files in a directory

Use search_codebase for project-specific questions. Use web_search for current
information, documentation, or anything not in the codebase. Answer from your own
knowledge when you're confident and the question doesn't require searching.
Provide thorough, well-sourced answers. Be concise and direct."""

research_agent = create_agent(
    research_llm,
    research_tools,
    system_prompt=RESEARCH_PROMPT,
)
