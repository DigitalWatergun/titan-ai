from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from titan.llm import ChatOpenAIWithReasoning as ChatOpenAI
from titan.tools.filesystem import list_directory, read_file, write_file
from titan.tools.rag import search_codebase
from titan.tools.shell import run_command


class CodeAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


code_tools = [read_file, write_file, list_directory, run_command, search_codebase]

code_llm = ChatOpenAI(
    base_url="http://localhost:8002/v1",
    api_key=SecretStr("not-needed"),
    model="code",
    temperature=0,
).bind_tools(code_tools)

CODE_PROMPT = """You are Cody, an expert coding agent. You write clean, correct code.
You have these tools available — ALWAYS use them instead of writing code snippets:
- read_file: Read file contents
- write_file: Write content to a file
- list_directory: List files in a directory
- run_command: Run shell commands
- search_codebase: Search the indexed codebase by meaning

Never write Python code to read files or list directories. Use your tools.
Always start by listing the current directory with "." — never guess paths.
Always read existing code before modifying it. Explain what you changed and why.
Be concise and direct in your responses."""


def call_code_model(state: CodeAgentState) -> dict:
    messages = [SystemMessage(content=CODE_PROMPT)] + state["messages"]
    response = code_llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: CodeAgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_code_agent():
    graph = StateGraph(CodeAgentState)
    graph.add_node("model", call_code_model)
    graph.add_node("tools", ToolNode(code_tools))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


code_agent = build_code_agent()
