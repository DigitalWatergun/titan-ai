from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from titan.graph.code_agent import code_agent
from titan.graph.research_agent import research_agent
from titan.graph.review_agent import review_agent
from titan.graph.router import pick_agent, route_request
from titan.graph.state import AgentState


def _get_specialist_input(state: AgentState) -> list:
    """Build clean input for a specialist — just context + user question."""
    messages = []

    context_parts = []
    if state.get("research_output"):
        context_parts.append(f"[Paige's research]\n{state['research_output']}")
    if state.get("code_output"):
        context_parts.append(f"[Cody's code changes]\n{state['code_output']}")
    if state.get("review_output"):
        context_parts.append(f"[Mark's review]\n{state['review_output']}")
    if context_parts:
        messages.append(
            SystemMessage(
                content="Previous agent outputs:\n\n" + "\n\n".join(context_parts)
            )
        )

    for msg in reversed(state["messages"]):
        if msg.type == "human":
            messages.append(msg)
            break

    return messages


def invoke_code_agent(state: AgentState) -> dict:
    messages = _get_specialist_input(state)
    result = code_agent.invoke({"messages": messages})
    return {
        "messages": result["messages"],
        "code_output": result["messages"][-1].content,
    }


def invoke_research_agent(state: AgentState) -> dict:
    messages = _get_specialist_input(state)
    result = research_agent.invoke({"messages": messages})
    return {
        "messages": result["messages"],
        "research_output": result["messages"][-1].content,
    }


def invoke_review_agent(state: AgentState) -> dict:
    messages = _get_specialist_input(state)
    result = review_agent.invoke({"messages": messages})
    return {
        "messages": result["messages"],
        "review_output": result["messages"][-1].content,
    }


def build_multi_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router", route_request)
    graph.add_node("code_agent", invoke_code_agent)
    graph.add_node("research_agent", invoke_research_agent)
    graph.add_node("review_agent", invoke_review_agent)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        pick_agent,
        {
            "code": "code_agent",
            "research": "research_agent",
            "review": "review_agent",
            "done": END,
        },
    )

    graph.add_edge("code_agent", "router")
    graph.add_edge("research_agent", "router")
    graph.add_edge("review_agent", "router")

    return graph.compile()
