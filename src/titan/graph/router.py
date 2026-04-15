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
You are Bridget, a request router. Pick the NEXT step needed.

Categories:
- "research" — looking up information ONLY (web search or finding existing code)
- "code" — writing, creating, or modifying ANY file. ALWAYS use this when the user wants a file written.
- "review" — analyzing existing code (only AFTER code exists)
- "done" — task is fully complete

CRITICAL RULES:
- "create a file" → "code" (NEVER "research")
- "write a file" → "code" (NEVER "research")
- "look up X then write file" → first "research", then "code"
- After research is done and a file still needs to be written → ALWAYS route to "code"
- After code is written → "done" (unless review explicitly requested)

What has been completed so far:
{state_summary}

If research is done but the user also asked to write/create a file, the next step is "code".
If the completed work fully answers the user's question, respond with "done".

Respond with ONLY the category name, nothing else."""


MAX_ITERATIONS = 3

# Keywords that signal each agent type in the user's request
CODE_KEYWORDS = [
    "write a file",
    "write file",
    "create a file",
    "create file",
    "modify",
    "edit",
    "implement",
    "save to",
    "save as",
]
REVIEW_KEYWORDS = [
    "review",
    "check the code",
    "find bugs",
    "audit",
]
RESEARCH_KEYWORDS = [
    "look up",
    "search for",
    "find",
    "research",
    "what is",
    "how does",
    "explain",
]


def _detect_required_steps(user_msg: str) -> list[str]:
    """Detect what types of work the user is asking for, in order."""
    msg_lower = user_msg.lower()
    steps = []
    if any(kw in msg_lower for kw in RESEARCH_KEYWORDS):
        steps.append("research")
    if any(kw in msg_lower for kw in CODE_KEYWORDS):
        steps.append("code")
    if any(kw in msg_lower for kw in REVIEW_KEYWORDS):
        steps.append("review")
    return steps


def route_request(state: AgentState) -> dict:
    """Classify the next step needed, or decide we're done.

    Uses deterministic keyword detection on the user's message first.
    Falls back to LLM-based routing for ambiguous requests."""
    iteration = state.get("iteration", 0) + 1

    # Safety: force done after max iterations
    if iteration > MAX_ITERATIONS:
        return {"route": "done", "iteration": iteration}

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_msg = str(msg.content)
            break

    # Deterministic routing based on keywords in user's request
    required_steps = _detect_required_steps(last_user_msg)

    if required_steps:
        # Find the first required step that hasn't been completed
        completed = {
            "research": bool(state.get("research_output")),
            "code": bool(state.get("code_output")),
            "review": bool(state.get("review_output")),
        }
        for step in required_steps:
            if not completed[step]:
                return {"route": step, "iteration": iteration}
        # All required steps done
        return {"route": "done", "iteration": iteration}

    # Fall back to LLM routing for ambiguous requests
    summary = []
    if state.get("research_output"):
        summary.append(f"Research completed: {state['research_output'][:200]}")
    if state.get("code_output"):
        summary.append(f"Code written: {state['code_output'][:200]}")
    if state.get("review_output"):
        summary.append(f"Review completed: {state['review_output'][:200]}")

    if summary and iteration > 1:
        state_summary = (
            "\n".join(summary)
            + "\n\nThe above work likely answers the user's request. Only route to another agent if something is clearly missing."
        )
    elif summary:
        state_summary = "\n".join(summary)
    else:
        state_summary = "Nothing yet — this is the first step."

    prompt = ROUTER_PROMPT.format(state_summary=state_summary)
    response = router_llm.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=last_user_msg)]
    )

    route = str(response.content).strip().lower()
    if route not in ("code", "research", "review", "done"):
        route = "research"  # safer default than "code"

    return {"route": route, "iteration": iteration}


def pick_agent(state: AgentState) -> str:
    """Conditional edge — send to the right specialist or end."""
    return state["route"]
