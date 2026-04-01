from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from titan.tools.filesystem import list_directory, read_file, write_file
from titan.tools.shell import run_command


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


tools = [read_file, write_file, list_directory, run_command]

llm = ChatOllama(
    model="qwen2.5:7b", base_url="http://localhost:11434", temperature=0
).bind_tools(tools)

SYSTEM_PROMPT = """You are a coding assistant with access to the local filesystem. You can read files, write files, list directories, and run shell commands. When the user asks about code, use your tools to explore the codebase first. Be concise and direct in your responses."""


def call_model(state: AgentState) -> dict:
    """The LLM node - decides whether to use a tool or respond."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Check if the LLM wants to call a tool or is done."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_agent():
    """Construct the LangGraph agent.

    This wires call_model and should_continue into a graph:

    User message → call_model (LLM decides action)
               ↓
             should_continue (has tool calls?)
               ↓            ↓
             "tools"        END → response to user
               ↓
          ToolNode (executes the tool)
               ↓
          call_model (LLM sees tool result, decides next action)
               ... (loops until LLM responds without tool calls)
    """
    graph = StateGraph(AgentState)

    # Add nodes - these are where call_model and should_continue get used
    graph.add_node("agent", call_model)  # LLM node - calls call_model()
    graph.add_node("tools", ToolNode(tools))  # Tool executor

    # Set entry point
    graph.set_entry_point("agent")

    # After the LLM responds, should_continue() checks if it wants to call a tool
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    # After tool execution, go back to the agent (LLM sees the tool result)
    graph.add_edge("tools", "agent")

    return graph.compile()
