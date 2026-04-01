import typer
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from titan.agent import build_agent

app = typer.Typer()
console = Console()


@app.command()
def chat(working_dir: str = typer.Option(".", help="Working directory for the agent")):
    """Start an interactive chat with your local coding agent"""
    import os

    os.chdir(working_dir)
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
def index():
    print("Not implemented yet")


if __name__ == "__main__":
    app()
