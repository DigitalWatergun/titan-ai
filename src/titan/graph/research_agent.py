from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from titan.llm import ChatOpenAIWithReasoning as ChatOpenAI
from titan.tools.filesystem import list_directory, read_file
from titan.tools.rag import search_codebase
from titan.tools.search import web_search


class ResearchAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


research_tools = [read_file, list_directory, search_codebase, web_search]

research_llm = ChatOpenAI(
    base_url="http://localhost:8003/v1",
    api_key=SecretStr("not-needed"),
    model="research",
    temperature=0,
).bind_tools(research_tools)

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


def call_research_model(state: ResearchAgentState) -> dict:
    messages = [SystemMessage(content=RESEARCH_PROMPT)] + state["messages"]
    response = research_llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: ResearchAgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_research_agent():
    graph = StateGraph(ResearchAgentState)
    graph.add_node("model", call_research_model)
    graph.add_node("tools", ToolNode(research_tools))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


research_agent = build_research_agent()
