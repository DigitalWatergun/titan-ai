from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str  # "code", "research", "review", "done"
    working_directory: str
    context: str  # RAG context or intermediate results
    research_output: str  # Paige's findings (empty until she runs)
    code_output: str  # Cody's output (empty until he runs)
    review_output: str  # Mark's feedback (empty until he runs)
    iteration: int  # Track routing cycles to prevent infinite loops
