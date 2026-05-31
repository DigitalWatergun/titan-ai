import asyncio
import os
import time

from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.timer import Timer
from textual.widgets import Input, RichLog, Static

from titan.agents import MAIN_AGENT
from titan.loop import run_turn
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
        self.messages: list[dict] = []
        self._thinking_dots = 0
        self._thinking_timer: Timer | None = None
        self._start_time = 0.0
        self._total_tokens = 0
        self._context_used = 0
        self._context_max = 0
        self._show_full_thinking = False
        self._input_history: list[str] = []
        self._history_index = 0
        self._saved_input = ""

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
        if self._context_max > 0 and self._context_used > 0:
            pct = int(self._context_used / self._context_max * 100)
            parts.append(f"{pct}% ctx")
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

    def _fetch_context_size(self) -> None:
        import os

        import httpx

        llm_url = os.getenv("TITAN_LLM_URL", "http://localhost:8001/v1")
        base_url = llm_url.removesuffix("/v1")
        try:
            resp = httpx.get(f"{base_url}/props", timeout=5)
            data = resp.json()
            settings = data.get("default_generation_settings", {})
            self._context_max = settings.get("n_ctx", 0)
        except Exception:
            self._context_max = 0

    def on_mount(self) -> None:

        os.chdir(self.working_dir)
        self._fetch_context_size()

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

    def on_key(self, event: Key) -> None:
        input_widget = self.query_one("#input-box", Input)
        if event.key == "up":
            if self._input_history and self._history_index > 0:
                if self._history_index == len(self._input_history):
                    self._saved_input = input_widget.value
                self._history_index -= 1
                input_widget.value = self._input_history[self._history_index]
                input_widget.cursor_position = len(input_widget.value)
            event.prevent_default()
        elif event.key == "down":
            if self._history_index < len(self._input_history):
                self._history_index += 1
                if self._history_index == len(self._input_history):
                    input_widget.value = self._saved_input
                else:
                    input_widget.value = self._input_history[self._history_index]
                input_widget.cursor_position = len(input_widget.value)
            event.prevent_default()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        self._input_history.append(user_input)
        self._history_index = len(self._input_history)
        self._saved_input = ""

        input_widget = self.query_one("#input-box", Input)
        input_widget.value = ""

        log = self.query_one("#chat-log", RichLog)

        # Display user message
        log.write("")
        user_msg = Text.assemble(
            ("❯ ", "#4a90c2"),
            (user_input, "#ffffff"),
        )
        log.write(Padding(user_msg, (0), style="on #2a2a2a"), expand=True)
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

    @work
    async def _run_agent(self, user_input: str, log: RichLog) -> None:
        self.messages.append({"role": "user", "content": user_input})

        token_buffer: list[str] = []
        reasoning_buffer: list[str] = []

        def flush_text():
            """Render accumulated tokens as Markdown, then clear the buffer."""
            if reasoning_buffer:
                text = "".join(reasoning_buffer)
                thought = Text(f"💭 {text}", style="dim italic")
                log.write(thought)
                reasoning_buffer.clear()
            if token_buffer:
                text = "".join(token_buffer)
                log.write("")
                log.write(Markdown(text))
                log.write("")
                token_buffer.clear()

        try:
            last_event_type = None
            async for event in run_turn(MAIN_AGENT, self.messages):
                event_type = event[0]
                if last_event_type and last_event_type != event_type:
                    flush_text()
                last_event_type = event_type

                match event:
                    case ("reasoning", text):
                        reasoning_buffer.append(text)
                    case ("token", text):
                        token_buffer.append(text)
                    case ("tool_call", name, args):
                        log.write(Text(f"→ {name}({args})", style="dim"))
                    case ("tool_result", name, result):
                        snippet = result.replace("\n", " ").strip()
                        if len(snippet) > 80:
                            snippet = snippet[:80] + "..."
                        log.write(Text(f"✓ {name} → {snippet}", style="dim"))

            # Loop exited cleanly — flush any final text
            flush_text()

        except asyncio.CancelledError:
            # _cancel_agent already wrote "Interrupted" and stopped the spinner.
            raise

        except Exception as e:
            log.write(f"[red]Agent error: {e}[/red]")

        finally:
            self._stop_thinking()


def run(working_dir: str = "."):
    app = TitanApp(working_dir=working_dir)
    app.run()


if __name__ == "__main__":
    run()
