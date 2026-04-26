import time

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Input, RichLog, Static
from textual.worker import get_current_worker

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
        ("ctrl+c", "safe_quit"),
        ("escape", "interrupt"),
        ("pageup", "scroll_up"),
        ("pagedown", "scroll_down"),
        ("ctrl+o", "toggle_thinking"),
    ]

    def __init__(self, working_dir: str = "."):
        super().__init__()
        self.working_dir = working_dir
        self.agent: CompiledStateGraph | None = None
        self.messages: list = []
        self._thinking_dots = 0
        self._thinking_timer: Timer | None = None
        self._start_time = 0.0
        self._total_tokens = 0
        self._show_full_thinking = False

    def compose(self) -> ComposeResult:
        chat_log = RichLog(id="chat-log", wrap=True, highlight=False, markup=True)
        chat_log.can_focus = False
        yield chat_log
        with Vertical(id="bottom-area"):
            yield Input(placeholder="Ask Titan anything...", id="input-box")
            with Horizontal(id="status-bar"):
                yield Static("Titan", id="status-left")
                yield Static("/help", id="status-right")

    def _start_thinking(self) -> None:
        self._thinking_dots = 0
        self._start_time = time.monotonic()
        self._total_tokens = 0
        self._thinking_timer = self.set_interval(0.4, self._animate_thinking)
        self.query_one("#status-right", Static).update("Press ESC to interrupt")

    def _stop_thinking(self) -> None:
        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        elapsed = time.monotonic() - self._start_time
        self.query_one("#status-left", Static).update(self._format_status(elapsed))
        self.query_one("#status-right", Static).update("/help")

    def _animate_thinking(self) -> None:
        self._thinking_dots = (self._thinking_dots % 3) + 1
        dots = "." * self._thinking_dots
        elapsed = time.monotonic() - self._start_time
        self.query_one("#status-left", Static).update(
            self._format_status(elapsed, f"Thinking{dots}")
        )

    def _format_status(self, elapsed: float, suffix: str = "") -> str:
        h, remainder = divmod(int(elapsed), 3600)
        m, s = divmod(remainder, 60)
        time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        parts = ["Titan", time_str]
        if self._total_tokens > 0:
            parts.append(f"{self._total_tokens:,} tokens")
        if suffix:
            parts.append(suffix)
        return "  ".join(parts)

    def _is_agent_running(self) -> bool:
        return any(w.name == "_run_agent" and w.is_running for w in self.workers)

    def _cancel_agent(self) -> None:
        for worker in self.workers:
            if worker.name == "_run_agent" and worker.is_running:
                worker.cancel()
        self._stop_thinking()
        log = self.query_one("#chat-log", RichLog)
        log.write("[dim]Interrupted.[/dim]")
        log.write("")

    def action_safe_quit(self) -> None:
        if self._is_agent_running():
            self._cancel_agent()
        else:
            self.exit()

    def action_interrupt(self) -> None:
        if self._is_agent_running():
            self._cancel_agent()

    def action_toggle_thinking(self) -> None:
        self._show_full_thinking = not self._show_full_thinking
        mode = "full" if self._show_full_thinking else "summary"
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[dim]Thinking mode: {mode} (Ctrl+O to toggle)[/dim]")

    def action_scroll_up(self) -> None:
        self.query_one("#chat-log", RichLog).scroll_page_up()

    def action_scroll_down(self) -> None:
        self.query_one("#chat-log", RichLog).scroll_page_down()

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

        # Display user message
        log.write("")
        user_msg = Text.assemble(
            ("❯ ", "#4a90c2"),
            (user_input, "#ffffff"),
        )
        log.write(Padding(user_msg, (0, 2), style="on #2a2a2a"), expand=True)
        log.write("")

        # Handle slash commands
        if user_input.startswith("/"):
            self._handle_command(user_input, log)
            return

        # Run agent in background thread so UI doesn't freeze
        self._start_thinking()
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
            log.write("")
            log.write("[bold]Keyboard shortcuts:[/bold]")
            log.write("  [dim]Esc      — interrupt current response[/dim]")
            log.write("  [dim]Ctrl+O   — toggle full/summary thinking mode[/dim]")
            log.write("  [dim]PageUp   — scroll up[/dim]")
            log.write("  [dim]PageDn   — scroll down[/dim]")
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
            import traceback
            from pathlib import Path

            tb = traceback.format_exc()
            log_path = Path("/tmp/titan-error.log")
            log_path.write_text(tb)
            self.call_from_thread(log.write, f"[red]Indexing failed: {e}[/red]")
            self.call_from_thread(
                log.write, f"[dim]Full traceback written to {log_path}[/dim]"
            )

    @work(thread=True)
    def _run_agent(self, user_input: str, log: RichLog) -> None:
        assert self.agent is not None
        self.messages.append(HumanMessage(content=user_input))
        worker = get_current_worker()

        for event in self.agent.stream({"messages": self.messages}):
            if worker.is_cancelled:
                return

            for key, value in event.items():
                if key == "agent":
                    last_msg = value["messages"][-1]

                    # Accumulate tokens
                    usage = getattr(last_msg, "usage_metadata", None)
                    if usage:
                        self._total_tokens += usage.get("total_tokens", 0)

                    # Show reasoning
                    reasoning = (
                        last_msg.additional_kwargs.get("reasoning_content")
                        if hasattr(last_msg, "additional_kwargs")
                        else None
                    )
                    if reasoning and reasoning.strip():
                        if self._show_full_thinking:
                            thought = Text(f"💭 {reasoning}", style="dim italic")
                        else:
                            summary = " ".join(
                                line for line in reasoning.split("\n") if line.strip()
                            )
                            if len(summary) > 150:
                                summary = summary[:150] + "..."
                            thought = Text(f"💭 {summary}", style="dim italic")
                        self.call_from_thread(log.write, Padding(thought, (0, 0, 0, 2)))

                    # Show tool calls
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            tool_text = Text(
                                f"→ {tc['name']}({tc['args']})", style="dim"
                            )
                            self.call_from_thread(
                                log.write, Padding(tool_text, (0, 0, 0, 2))
                            )

                elif key == "tools":
                    for msg in value["messages"]:
                        if hasattr(msg, "name"):
                            self.call_from_thread(
                                log.write,
                                Padding(
                                    Text(f"✓ {msg.name} completed", style="dim"),
                                    (0, 0, 0, 2),
                                ),
                            )

        if worker.is_cancelled:
            return

        # Get final response
        result = self.agent.invoke({"messages": self.messages})
        self.messages = result["messages"]

        ai_message = self.messages[-1]
        usage = getattr(ai_message, "usage_metadata", None)
        if usage:
            self._total_tokens += usage.get("total_tokens", 0)

        self.call_from_thread(log.write, "")
        self.call_from_thread(log.write, Markdown(ai_message.content))
        self.call_from_thread(log.write, "")
        self.call_from_thread(self._stop_thinking)


def run(working_dir: str = "."):
    app = TitanApp(working_dir=working_dir)
    app.run()


if __name__ == "__main__":
    run()
