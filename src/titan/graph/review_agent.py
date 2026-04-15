from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from titan.llm import ChatOpenAIWithReasoning as ChatOpenAI
from titan.tools.filesystem import list_directory, read_file


class ReviewAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


review_tools = [read_file, list_directory]

review_llm = ChatOpenAI(
    base_url="http://localhost:8004/v1",
    api_key=SecretStr("not-needed"),
    model="review",
    temperature=0,
).bind_tools(review_tools)

REVIEW_PROMPT = """You are Mark, an expert code review agent. You analyze code for bugs, security issues, and improvements.
  You have these tools available — ALWAYS use them instead of writing code snippets:
  - read_file: Read file contents
  - list_directory: List files in a directory

  Never write Python code to read files or list directories. Use your tools.
  Always start by listing the current directory with "." — never guess paths.
  When reviewing code:
  - Read the relevant files before giving feedback
  - Point out specific lines and explain what's wrong
  - Suggest concrete fixes, not vague advice
  - Check for security issues, error handling, and edge cases
  Be concise and direct in your responses."""


def call_review_model(state: ReviewAgentState) -> dict:
    messages = [SystemMessage(content=REVIEW_PROMPT)] + state["messages"]
    response = review_llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: ReviewAgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_review_agent():
    graph = StateGraph(ReviewAgentState)
    graph.add_node("model", call_review_model)
    graph.add_node("tools", ToolNode(review_tools))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


review_agent = build_review_agent()
