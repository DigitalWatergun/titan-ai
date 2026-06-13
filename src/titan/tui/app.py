import asyncio
import os
from pathlib import Path

import httpx
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Input, ListView, RichLog
from textual.worker import get_current_worker

from titan.agents import MAIN_AGENT
from titan.chats.store import ChatStore
from titan.chats.titles import generate_title
from titan.loop import run_turn
from titan.rag.indexer import index_codebase
from titan.tui.commands import COMMANDS, CommandContext
from titan.tui.widgets.chat_sidebar import ChatItem, ChatSidebar
from titan.tui.widgets.status_bar import StatusBar

CHATS_DIR = Path.home() / ".titan" / "chats"


class _IndexCancelled(Exception):
    pass


class TitanApp(App):
    """Titan — Local Coding Assistant TUI"""

    TITLE = "Titan"
    CSS_PATH = "titan.tcss"
    BINDINGS = [
        ("ctrl+c", "safe_quit"),
        ("escape", "interrupt"),
        ("pageup", "scroll_up"),
        ("pagedown", "scroll_down"),
        ("ctrl+o", "toggle_thinking"),
        Binding("ctrl+e", "toggle_chat_sidebar", priority=True),
    ]

    def __init__(self, working_dir: str = "."):
        super().__init__()
        self.working_dir = working_dir
        self._store = ChatStore(CHATS_DIR)
        self._show_full_thinking = False
        self._input_history: list[str] = []
        self._history_index = 0
        self._saved_input = ""

    @property
    def messages(self) -> list[dict]:
        return self._store.messages

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            yield ChatSidebar(id="chat-sidebar")
            with Vertical(id="main-area"):
                chat_log = RichLog(
                    id="chat-log", wrap=True, highlight=False, markup=True
                )
                chat_log.can_focus = False
                yield chat_log
                yield Input(placeholder="Ask Titan anything...", id="input-box")
        yield StatusBar()

    def _is_agent_running(self) -> bool:
        return any(w.name == "_run_agent" and w.is_running for w in self.workers)

    def _is_index_running(self) -> bool:
        return any(w.name == "_run_index" and w.is_running for w in self.workers)

    def _cancel_agent(self) -> None:
        for worker in self.workers:
            if worker.name == "_run_agent" and worker.is_running:
                worker.cancel()
        self.query_one(StatusBar).stop_thinking()
        log = self.query_one("#chat-log", RichLog)
        log.write("[dim]Interrupted.[/dim]")
        log.write("")

    def _cancel_index(self) -> None:
        for worker in self.workers:
            if worker.name == "_run_index" and worker.is_running:
                worker.cancel()
        self.query_one(StatusBar).stop_activity()
        self.query_one("#chat-log", RichLog).write("[dim]Indexing interrupted.[/dim]")

    def action_safe_quit(self) -> None:
        if self._is_agent_running():
            self._cancel_agent()
        elif self._is_index_running():
            self._cancel_index()
        else:
            self.exit()

    def action_interrupt(self) -> None:
        if self._is_agent_running():
            self._cancel_agent()
        elif self._is_index_running():
            self._cancel_index()

    def action_toggle_thinking(self) -> None:
        self._show_full_thinking = not self._show_full_thinking
        mode = "full" if self._show_full_thinking else "summary"
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[dim]Thinking mode: {mode} (Ctrl+O to toggle)[/dim]")

    async def action_toggle_chat_sidebar(self) -> None:
        chat_sidebar = self.query_one(ChatSidebar)
        chat_sidebar.display = not chat_sidebar.display
        if chat_sidebar.display:
            await chat_sidebar.populate(self._store.list_chats(cwd=os.getcwd()))
            chat_sidebar.query_one(ListView).focus()
        else:
            self.query_one("#input-box", Input).focus()

    def action_scroll_up(self) -> None:
        self.query_one("#chat-log", RichLog).scroll_page_up()

    def action_scroll_down(self) -> None:
        self.query_one("#chat-log", RichLog).scroll_page_down()

    def _fetch_context_size(self) -> None:
        llm_url = os.getenv("TITAN_LLM_URL", "http://localhost:8001/v1")
        base_url = llm_url.removesuffix("/v1")
        try:
            resp = httpx.get(f"{base_url}/props", timeout=5)
            data = resp.json()
            settings = data.get("default_generation_settings", {})
            n_ctx = settings.get("n_ctx", 0)
        except Exception:
            n_ctx = 0
        self.query_one(StatusBar).n_ctx = n_ctx

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
        self.query_one("#input-box", Input).focus()

    def on_key(self, event: Key) -> None:
        input_widget = self.query_one("#input-box", Input)
        if not input_widget.has_focus:
            return
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
        self._write_user_message(user_input)

        # Handle slash commands
        if user_input.startswith("/"):
            self._handle_command(user_input, log)
            return

        # Run agent in background thread so UI doesn't freeze
        self.query_one(StatusBar).start_thinking()
        self._run_agent(user_input, log)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, ChatItem):
            return
        self._store.load(item.chat_id)
        self._render_chat()
        self.query_one("#input-box", Input).focus()

    def _handle_command(self, user_input: str, log: RichLog) -> None:
        name, _, args = user_input.strip().partition(" ")
        name = name.lower()
        handler = COMMANDS.get(name)
        if handler:
            handler(CommandContext(app=self, args=args, log=log))
        else:
            log.write(f"[dim]Unknown command: {name}[/dim]")

    @work(thread=True)
    def _run_index(self, directory: str, display_dir: str, log: RichLog) -> None:
        worker = get_current_worker()
        bar = self.query_one(StatusBar)
        self.call_from_thread(bar.start_activity, "Indexing")

        def on_progress(msg: str) -> None:
            if worker.is_cancelled:
                raise _IndexCancelled
            self.call_from_thread(log.write, f"  [#4a90c2]→[/#4a90c2] [dim]{msg}[/dim]")

        try:
            count = index_codebase(directory, on_progress=on_progress)
            self.call_from_thread(
                log.write,
                f"[#4a90c2]Indexed {count} chunks from {display_dir}[/#4a90c2]",
            )
        except _IndexCancelled:
            pass
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
        finally:
            self.call_from_thread(bar.stop_activity)

    # --- rendering primitives (shared by live streaming + history replay) ---

    def _write_user_message(self, content: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        log.write(
            Padding(
                Text.assemble(("❯ ", "#4a90c2"), (content, "#ffffff")),
                (0,),
                style="on #2a2a2a",
            ),
            expand=True,
        )
        log.write("")

    def _write_assistant_text(self, content: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        log.write(Markdown(content))
        log.write("")

    def _write_tool_call(self, label: str) -> None:
        self.query_one("#chat-log", RichLog).write(Text(f"→ {label}", style="dim"))

    def _write_tool_result(self, label: str) -> None:
        self.query_one("#chat-log", RichLog).write(Text(f"✓ {label}", style="dim"))

    @staticmethod
    def _snippet(text: str, limit: int = 80) -> str:
        s = text.replace("\n", " ").strip()
        return s[:limit] + "..." if len(s) > limit else s

    def _render_chat(self) -> None:
        self.query_one("#chat-log", RichLog).clear()
        for m in self.messages:
            role, content = m["role"], m.get("content") or ""
            if role == "user":
                self._write_user_message(content)
            elif role == "assistant" and content:
                self._write_assistant_text(content)
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    self._write_tool_call(f"{tc['function']['name']}(...)")
            elif role == "tool":
                self._write_tool_result(self._snippet(content))

    @work
    async def _run_agent(self, user_input: str, log: RichLog) -> None:
        self.messages.append({"role": "user", "content": user_input})

        token_buffer: list[str] = []
        reasoning_buffer: list[str] = []

        def flush_text():
            """Render accumulated tokens as Markdown, then clear the buffer."""
            if reasoning_buffer:
                text = "".join(reasoning_buffer)
                log.write(Text(f"💭 {text}", style="dim italic"))
                reasoning_buffer.clear()
            if token_buffer:
                self._write_assistant_text("".join(token_buffer))
                token_buffer.clear()

        bar = self.query_one(StatusBar)

        try:
            last_event_type = None
            async for event in run_turn(MAIN_AGENT, self.messages):
                event_type = event[0]

                if event_type == "usage":
                    usage = event[1]
                    bar.commit_usage(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                    continue

                if last_event_type and last_event_type != event_type:
                    flush_text()
                last_event_type = event_type

                match event:
                    case ("reasoning", text):
                        reasoning_buffer.append(text)
                        bar.add_live_token()
                    case ("token", text):
                        token_buffer.append(text)
                        bar.add_live_token()
                    case ("tool_call", name, args):
                        self._write_tool_call(f"{name}({args})")
                    case ("tool_result", name, result):
                        self._write_tool_result(f"{name} → {self._snippet(result)}")

            # Loop exited cleanly — flush any final text
            flush_text()

            if not self._store.title:
                self._generate_title_bg()

        except asyncio.CancelledError:
            # _cancel_agent already wrote "Interrupted" and stopped the spinner.
            raise

        except Exception as e:
            log.write(f"[red]Agent error: {e}[/red]")

        finally:
            self._store.save()
            self.query_one(StatusBar).stop_thinking()

    @work
    async def _generate_title_bg(self) -> None:
        chat_id = self._store._id
        title = await generate_title(self.messages, MAIN_AGENT.model_name)
        if self._store._id == chat_id:
            self._store.set_title(title)
            self._store.save()


def run(working_dir: str = "."):
    app = TitanApp(working_dir=working_dir)
    app.run()


if __name__ == "__main__":
    run()
