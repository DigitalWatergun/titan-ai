import logging
from pathlib import Path

import typer
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from titan.graph import build_multi_agent
from titan.graph.state import AgentState
from titan.rag.indexer import index_codebase

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = typer.Typer()
console = Console()


def _get_agent_name(namespace: object) -> str:
    """Extract agent name from subgraph namespace."""
    ns_str = str(namespace)
    if "research_agent" in ns_str:
        return "Paige"
    elif "code_agent" in ns_str:
        return "Cody"
    elif "review_agent" in ns_str:
        return "Mark"
    return "Agent"


@app.command()
def chat(
    working_dir: str = typer.Option(".", help="Working directory for the agent"),
    index: bool = typer.Option(
        False, help="Auto-index the working directory if no index exists"
    ),
):
    """Start an interactive chat with your local coding agent"""
    import os

    os.chdir(working_dir)

    if index:
        index_path = Path(working_dir) / ".titan" / "chromadb"
        if not index_path.exists():
            console.print("No index found, indexing codebase...")
            count = index_codebase(working_dir)
            console.print(f"Indexed [green]{count}[/] chunks into ChromaDB")

    console.print(
        Panel(
            f"Titan - Working in: {os.getcwd()}\nType 'quit' to exit",
            title="Titan",
            border_style="blue",
        )
    )

    agent = build_multi_agent()
    messages: list = []

    while True:
        user_input = console.input("[bold blue]You:[/] ")
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append(HumanMessage(content=user_input))

        # Stream events with subgraphs for real-time tool visibility
        final_messages = None
        input_state: AgentState = {
            "messages": messages + [HumanMessage(content=user_input)],
            "route": "",
            "research_output": "",
            "code_output": "",
            "review_output": "",
            "iteration": 0,
            "working_directory": "",
            "context": "",
        }
        for namespace, event in agent.stream(input_state, subgraphs=True):
            for key, value in event.items():  # type: ignore[misc]
                # Only capture final messages from outer graph events
                if namespace == () and "messages" in value:
                    final_messages = value["messages"]

                # Outer graph: router decisions
                if namespace == () and key == "router":
                    route = value.get("route", "")
                    iteration = value.get("iteration", 0)
                    if route == "done":
                        console.print(
                            f"  [dim]✅ Router → done (after {iteration} steps)[/dim]"
                        )
                    elif route:
                        agent_names = {
                            "code": "Cody",
                            "research": "Paige",
                            "review": "Mark",
                        }
                        name = agent_names.get(route, route)
                        console.print(f"  [dim]🔀 Router → {name}[/dim]")
                        console.print(
                            f"  [dim]    {name} is thinking...[/dim]", end="\r"
                        )

                # Inner subgraph: model calls (tool requests and responses)
                elif len(namespace) > 0 and key == "model":
                    agent_name = _get_agent_name(namespace)
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            console.print(
                                f"  [dim]    {agent_name} → {tc['name']}({tc['args']})[/dim]"
                            )
                    elif last_msg.type == "ai" and last_msg.content:
                        preview = last_msg.content[:80].replace("\n", " ")
                        console.print(f"  [dim]    {agent_name}: {preview}...[/dim]")

                # Inner subgraph: tool execution results
                elif len(namespace) > 0 and key == "tools":
                    for msg in value["messages"]:
                        if hasattr(msg, "name"):
                            console.print(f"  [dim]    ✓ {msg.name} completed[/dim]")

        # Update messages from stream
        if final_messages:
            messages = final_messages

        # Display the final response
        ai_message = messages[-1]
        console.print()
        console.print(Markdown(ai_message.content))
        console.print()


@app.command()
def index(directory: str = typer.Argument(".", help="Directory to index")):
    """Index a codebase into the vector database for RAG search."""

    console.print(f"Indexing [bold]{directory}[/]...")
    count = index_codebase(directory)
    console.print(f"Indexed [green]{count}[/] chunks into ChromaDB")


if __name__ == "__main__":
    app()
