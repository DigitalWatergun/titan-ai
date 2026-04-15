from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from titan.graph.state import AgentState

router_llm = ChatOpenAI(
    base_url="http://localhost:8001/v1",  # llama.cpp on GPU 0
    api_key=SecretStr("not-needed"),
    model="router",
    temperature=0,
)

ROUTER_PROMPT = """/no_think
You are Bridget, a request router. Based on the user's request and what has been completed so far, pick the NEXT step needed, or "done" if the task is complete.
Categories:
- "code" — writing, editing, or creating code files
- "research" — searching for information, documentation, or understanding concepts
- "review" — reviewing existing code, finding bugs, suggesting improvements
- "done" — the task is fully complete. Use this when the completed work already answers the user's request.

What has been completed so far:
{state_summary}

If the completed work above already answers the user's question, respond with "done".
Respond with ONLY the category name, nothing else."""

MAX_ITERATIONS = 5


def route_request(state: AgentState) -> dict:
    """Classify the next step needed, or decide we're done."""
    iteration = state.get("iteration", 0) + 1

    # Safety: force done after max iterations
    if iteration > MAX_ITERATIONS:
        return {"route": "done", "iteration": iteration}

    # Build summary of what's been done
    summary = []
    if state.get("research_output"):
        summary.append(f"Research completed: {state['research_output'][:200]}")
    if state.get("code_output"):
        summary.append(f"Code written: {state['code_output'][:200]}")
    if state.get("review_output"):
        summary.append(f"Review completed: {state['review_output'][:200]}")

    # If any work has been done and we're past iteration 1, bias toward done
    if summary and iteration > 1:
        state_summary = (
            "\n".join(summary)
            + "\n\nThe above work likely answers the user's request. Only route to another agent if something is clearly missing."
        )
    elif summary:
        state_summary = "\n".join(summary)
    else:
        state_summary = "Nothing yet — this is the first step."

    last_user_msg = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_msg = msg.content
            break

    prompt = ROUTER_PROMPT.format(state_summary=state_summary)
    response = router_llm.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=last_user_msg)]
    )

    route = str(response.content).strip().lower()
    if route not in ("code", "research", "review", "done"):
        route = "code"  # Default fallback

    return {"route": route, "iteration": iteration}


def pick_agent(state: AgentState) -> str:
    """Conditional edge — send to the right specialist or end."""
    return state["route"]
