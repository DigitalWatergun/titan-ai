from langchain_core.messages import HumanMessage
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static

from titan.agent import build_agent
from titan.rag.indexer import index_codebase


class TitanApp(App):
    """Titan — Local Coding Assistant TUI"""

    TITLE = "Titan"

    CSS = """
    Screen {
        background: #1e1e1e;
        padding: 1 2;
    }

    #chat-log {
        height: 1fr;
        padding: 1 0;
        border: none;
        background: #1e1e1e;
        color: #ffffff;
        scrollbar-size: 0 0;
    }

    #bottom-area {
        dock: bottom;
        height: auto;
        background: #1e1e1e;
    }

    #input-box {
        background: #2a2a2a;
        border-top: none;
        border-right: none;
        border-bottom: none;
        border-left: tall #4a90c2;
        color: #ffffff;
        padding: 1 2;
    }

    #input-box:focus {
        border-top: none;
        border-right: none;
        border-bottom: none;
        border-left: tall #4a90c2;
    }

    #status-bar {
        height: 1;
        margin-top: 1;
        background: #1e1e1e;
        color: #666666;
        padding: 0 1;
    }

    #status-left {
        width: 1fr;
        height: 1;
        color: #4a90c2;
        background: #1e1e1e;
    }

    #status-right {
        width: auto;
        height: 1;
        color: #666666;
        background: #1e1e1e;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit"),
    ]

    def __init__(self, working_dir: str = "."):
        super().__init__()
        self.working_dir = working_dir
        self.agent = None
        self.messages: list = []

    def compose(self) -> ComposeResult:
        chat_log = RichLog(id="chat-log", wrap=True, highlight=False, markup=True)
        chat_log.can_focus = False
        yield chat_log
        with Vertical(id="bottom-area"):
            yield Input(placeholder="Ask Titan anything...", id="input-box")
            with Horizontal(id="status-bar"):
                yield Static("Titan", id="status-left")
                yield Static("/help", id="status-right")

    def on_mount(self) -> None:
        import os

        os.chdir(self.working_dir)
        self.agent = build_agent()

        log = self.query_one("#chat-log", RichLog)

        cwd = os.getcwd()
        welcome = Text.assemble(
            ("Working in: ", "dim"),
            (cwd, "dim"),
            ("\nType /help for commands, Ctrl+C to quit.", "dim"),
        )
        log.write(
            Panel(
                welcome,
                border_style="#4a90c2",
                title="Titan",
                title_align="left",
                padding=(1, 2),
            )
        )
        log.write("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        input_widget = self.query_one("#input-box", Input)
        input_widget.value = ""

        log = self.query_one("#chat-log", RichLog)

        # Handle slash commands
        if user_input.startswith("/"):
            self._handle_command(user_input, log)
            return

        # Display user message
        log.write("")
        user_msg = Text.assemble(
            ("❯ ", "#4a90c2"),
            (user_input, "#ffffff"),
        )
        log.write(Padding(user_msg, (0, 2), style="on #2a2a2a"), expand=True)
        log.write("")

        # Run agent in background thread so UI doesn't freeze
        self._run_agent(user_input, log)

    def _handle_command(self, user_input: str, log: RichLog) -> None:
        command = user_input.strip().lower()

        if command == "/quit":
            self.exit()
        elif command == "/clear":
            self.messages = []
            log.clear()
            log.write("[dim]Conversation cleared.[/dim]")
        elif command == "/help":
            log.write("\n[bold]Available commands:[/bold]")
            log.write("  [dim]/clear   — clear conversation[/dim]")
            log.write(
                "  [dim]/index   — index current directory (or /index <path>)[/dim]"
            )
            log.write("  [dim]/help    — show this help[/dim]")
            log.write("  [dim]/quit    — exit[/dim]")
        elif command.startswith("/index"):
            import os

            parts = user_input.strip().split(maxsplit=1)
            directory = parts[1] if len(parts) > 1 else "."
            display_dir = os.path.abspath(directory) if directory == "." else directory
            log.write(f"[dim]Indexing {display_dir}...[/dim]")
            self._run_index(directory, display_dir, log)
        else:
            log.write(f"[dim]Unknown command: {command}[/dim]")

    @work(thread=True)
    def _run_index(self, directory: str, display_dir: str, log: RichLog) -> None:
        def on_progress(msg: str) -> None:
            self.call_from_thread(log.write, f"  [#4a90c2]→[/#4a90c2] [dim]{msg}[/dim]")

        try:
            count = index_codebase(directory, on_progress=on_progress)
            self.call_from_thread(
                log.write,
                f"[#4a90c2]Indexed {count} chunks from {display_dir}[/#4a90c2]",
            )
        except Exception as e:
            self.call_from_thread(log.write, f"[red]Indexing failed: {e}[/red]")

    @work(thread=True)
    def _run_agent(self, user_input: str, log: RichLog) -> None:
        self.messages.append(HumanMessage(content=user_input))

        for event in self.agent.stream({"messages": self.messages}):
            for key, value in event.items():
                if key == "agent":
                    last_msg = value["messages"][-1]

                    # Show reasoning
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
                            self.call_from_thread(
                                log.write,
                                f"  [dim italic]💭 {line[:120]}[/dim italic]",
                            )

                    # Show tool calls
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            self.call_from_thread(
                                log.write,
                                f"  [dim]→ {tc['name']}({tc['args']})[/dim]",
                            )

                elif key == "tools":
                    for msg in value["messages"]:
                        if hasattr(msg, "name"):
                            self.call_from_thread(
                                log.write,
                                f"  [dim]✓ {msg.name} completed[/dim]",
                            )

        # Get final response
        result = self.agent.invoke({"messages": self.messages})
        self.messages = result["messages"]

        ai_message = self.messages[-1]
        self.call_from_thread(log.write, "")
        self.call_from_thread(log.write, Markdown(ai_message.content))
        self.call_from_thread(log.write, "")


def run(working_dir: str = "."):
    app = TitanApp(working_dir=working_dir)
    app.run()


if __name__ == "__main__":
    run()
