from pathlib import Path

import typer
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from titan.agent import build_agent
from titan.rag.indexer import index_codebase

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

    agent = build_agent()
    messages = []

    while True:
        user_input = console.input("[bold blue]You:[/] ")
        if user_input.lower() in ("quit", "exit", "q'"):
            break

        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

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
