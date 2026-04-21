import logging
from pathlib import Path

import typer
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from titan.agent import build_agent
from titan.rag.indexer import index_codebase

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
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
    messages: list = []

    while True:
        user_input = console.input("[bold blue]You:[/] ")
        if user_input.startswith("/"):
            command = user_input.strip().lower()
            if command == "/quit":
                break
            elif command == "/clear":
                messages = []
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            elif command == "/help":
                console.print("[dim]/clear — clear conversation[/dim]")
                console.print("[dim]/index — index current directory[/dim]")
                console.print("[dim]/quit  — exit[/dim]")
                continue
            elif command.startswith("/index"):
                parts = user_input.strip().split(maxsplit=1)
                directory = parts[1] if len(parts) > 1 else "."
                console.print(f"Indexing [bold]{directory}[/]...")
                count = index_codebase(directory)
                console.print(f"[dim]Indexed {count} chunks from {directory}[/dim]")
                continue
            else:
                console.print(f"[dim]Unknown command: {command}[/dim]")
                continue

        messages.append(HumanMessage(content=user_input))

        # Stream events for real-time tool visibility
        for event in agent.stream({"messages": messages}):
            for key, value in event.items():
                if key == "agent":
                    last_msg = value["messages"][-1]

                    # Show reasoning if present (Qwen3.5 thinking mode)
                    reasoning = (
                        last_msg.additional_kwargs.get("reasoning_content")
                        if hasattr(last_msg, "additional_kwargs")
                        else None
                    )
                    if reasoning:
                        first_lines = [
                            line for line in reasoning.split("\n") if line.strip()
                        ][:3]
                        for line in first_lines:
                            console.print(
                                f"  [dim italic]💭 Titan thinking: {line[:120]}...[/dim italic]"
                            )

                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            console.print(
                                f"  [dim]  Titan → {tc['name']}({tc['args']})[/dim]"
                            )

                elif key == "tools":
                    for msg in value["messages"]:
                        if hasattr(msg, "name"):
                            console.print(f"  [dim]  ✓ {msg.name} completed[/dim]")

        # Get final messages from the stream
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        # Display the final response
        ai_message = messages[-1]
        console.print()
        console.print(Markdown(ai_message.content))
        console.print()


if __name__ == "__main__":
    app()
