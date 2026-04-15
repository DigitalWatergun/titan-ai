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
    messages = []

    while True:
        user_input = console.input("[bold blue]You:[/] ")
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append(HumanMessage(content=user_input))

        # Stream events to show tool usage in real time
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
        for event in agent.stream(input_state):
            for key, value in event.items():
                if "messages" in value:
                    final_messages = value["messages"]

                if key == "router":
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
                elif key in ("code_agent", "research_agent", "review_agent"):
                    agent_names = {
                        "code_agent": "Cody",
                        "research_agent": "Paige",
                        "review_agent": "Mark",
                    }
                    name = agent_names.get(key, key)
                    for msg in value["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                console.print(
                                    f"  [dim]  {name} → {tc['name']}({tc['args']})[/dim]"
                                )
                        elif msg.type == "tool" and hasattr(msg, "name"):
                            console.print(f"  [dim]  ✓ {msg.name} completed[/dim]")
                        elif msg.type == "ai" and msg.content:
                            preview = msg.content[:80].replace("\n", " ")
                            console.print(f"  [dim]  {name}: {preview}...[/dim]")

        # Get final state with all messages
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
