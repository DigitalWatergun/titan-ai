import asyncio
import os
from pathlib import Path

import httpx
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Input
from textual.worker import get_current_worker

from titan.agents import MAIN_AGENT
from titan.chats.compaction import compact
from titan.chats.rag import (
    count_chat_pieces,
    count_turns,
    index_all_chats,
    index_chat,
    set_current_chat_id,
)
from titan.chats.store import ChatStore
from titan.chats.titles import generate_title
from titan.loop import run_turn
from titan.rag.indexer import (
    collect_and_chunk,
    embed_entries,
)
from titan.tui.commands import COMMANDS, CommandContext
from titan.tui.screens.chat_picker_screen import ChatPickerScreen
from titan.tui.screens.confirm_screen import request_approval
from titan.tui.widgets.chat_log import ChatLog
from titan.tui.widgets.compaction_progress import CompactionProgress
from titan.tui.widgets.status_bar import StatusBar

CHATS_DIR = Path.home() / ".titan" / "chats"
COMPACT_THRESHOLD = 0.8
ESTIMATE_CONFIRM_CHUNKS = 10_000


def _format_size(chunks: int) -> str:
    kb = chunks * 5
    return f"{kb} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"


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
        ("ctrl+y", "copy_mode"),
    ]

    def __init__(self, working_dir: str = "."):
        super().__init__()
        self.working_dir = working_dir
        self._store = ChatStore(CHATS_DIR)
        set_current_chat_id(self._store.id)
        self._show_full_thinking = False
        self._input_history: list[str] = []
        self._history_index = 0
        self._saved_input = ""
        self._compacting = False

    @property
    def messages(self) -> list[dict]:
        return self._store.messages

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="main-area"):
                yield ChatLog(id="chat-log")
                yield CompactionProgress()
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
        log = self.query_one("#chat-log", ChatLog)
        log.write("[dim]Interrupted.[/dim]")
        log.write("")

    def _cancel_index(self) -> None:
        for worker in self.workers:
            if worker.name == "_run_index" and worker.is_running:
                worker.cancel()
        self.query_one(StatusBar).stop_activity()
        self.query_one("#chat-log", ChatLog).write("[dim]Indexing interrupted.[/dim]")

    def _print_estimate(self, entries, log: ChatLog) -> int:
        total = sum(len(chunks) for _, _, chunks in entries)
        for _, path, chunks in entries:
            log.write(
                f"  [#58a6ff]→[/#58a6ff] [dim]{len(chunks)} chunks — {Path(path).name}[/dim]"
            )
        log.write(f"[dim]Estimated ~{total} chunks, ~{_format_size(total)}[/dim]")
        return total

    async def _confirm_over_threshold(
        self, total: int, unit: str, log: ChatLog
    ) -> bool:
        # if total <= ESTIMATE_CONFIRM_CHUNKS:
        #     return True
        if (
            await request_approval(
                self, f"Index ~{total} {unit} (~{_format_size(total)})?"
            )
            == "yes"
        ):
            return True
        log.write("[dim]Cancelled - nothing embedded.[/dim]")
        return False

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
        log = self.query_one("#chat-log", ChatLog)
        log.write(f"[dim]Thinking mode: {mode} (Ctrl+O to toggle)[/dim]")

    def action_scroll_up(self) -> None:
        self.query_one("#chat-log", ChatLog).scroll_page_up()

    def action_scroll_down(self) -> None:
        self.query_one("#chat-log", ChatLog).scroll_page_down()

    def action_copy_mode(self) -> None:
        self.query_one("#chat-log", ChatLog).enter_copy_mode()

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

        log = self.query_one("#chat-log", ChatLog)

        cwd = os.getcwd()
        welcome = Text.assemble(
            ("Working in: ", "dim"),
            (cwd, "dim"),
            ("\nType /help for commands, Ctrl+C to quit.", "dim"),
        )
        log.write(
            Panel(
                welcome,
                border_style="#58a6ff",
                title="Titan",
                title_align="left",
                padding=(1, 2),
            )
        )
        log.write("")
        self.query_one("#input-box", Input).focus()

    def on_key(self, event: Key) -> None:
        input_widget = self.query_one("#input-box", Input)
        if self.focused is not input_widget:
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

        log = self.query_one("#chat-log", ChatLog)

        # Display user message
        self._write_user_message(user_input)

        # Handle slash commands
        if user_input.startswith("/"):
            self._handle_command(user_input, log)
            return

        # Run agent in background thread so UI doesn't freeze
        self.query_one(StatusBar).start_thinking()
        self._run_agent(user_input, log)

    def _handle_command(self, user_input: str, log: ChatLog) -> None:
        name, _, args = user_input.strip().partition(" ")
        name = name.lower()
        handler = COMMANDS.get(name)
        if handler:
            handler(CommandContext(app=self, args=args, log=log))
        else:
            log.write(f"[dim]Unknown command: {name}[/dim]")

    @work
    async def _run_index(
        self, directory: Path, collection_type: str, log: ChatLog
    ) -> None:
        worker = get_current_worker()
        bar = self.query_one(StatusBar)

        def on_progress(msg: str) -> None:
            if worker.is_cancelled:
                raise _IndexCancelled
            self.call_from_thread(log.write, f"  [#58a6ff]→[/#58a6ff] [dim]{msg}[/dim]")

        try:
            if collection_type == "chats":
                base = directory
                total = await asyncio.to_thread(count_chat_pieces, base)
                log.write(
                    f"[dim]Estimated ~{total} chat pieces, ~{_format_size(total)}[/dim]"
                )
                if not await self._confirm_over_threshold(total, "chat pieces", log):
                    return
                bar.start_activity("Indexing")
                results = await asyncio.to_thread(
                    lambda: {"chats": index_all_chats(base, on_progress)}
                )
            else:
                entries = await asyncio.to_thread(
                    collect_and_chunk, directory, collection_type, on_progress
                )
                total = self._print_estimate(entries, log)
                if not await self._confirm_over_threshold(total, "chunks", log):
                    return
                bar.start_activity("Indexing")
                results = await asyncio.to_thread(embed_entries, entries, on_progress)

            total = sum(results.values())
            for path, n in results.items():
                log.write(
                    f"  [#58a6ff]→[/#58a6ff] [dim]{n} chunks — {Path(path).name}[/dim]"
                )
            log.write(
                f"[#58a6ff]Indexed {total} chunks across {len(results)} collection(s)[/#58a6ff]"
            )
        except _IndexCancelled:
            pass
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            log_path = Path("/tmp/titan-error.log")
            log_path.write_text(tb)
            log.write(f"[red]Indexing failed: {e}[/red]")
            log.write(f"[dim]Full traceback written to {log_path}[/dim]")
        finally:
            bar.stop_activity()

    # --- rendering primitives (shared by live streaming + history replay) ---

    def _write_user_message(self, content: str) -> None:
        log = self.query_one("#chat-log", ChatLog)
        log.write("")
        log.write(
            Padding(
                Text.assemble(("❯ ", "#58a6ff"), (content, "#ffffff")),
                (0,),
                style="on #30363d",
            ),
        )
        log.write("")

    def _write_assistant_text(self, content: str) -> None:
        log = self.query_one("#chat-log", ChatLog)
        log.write("")
        log.write_markdown(content)
        log.write("")

    def _write_tool_call(self, label: str) -> None:
        self.query_one("#chat-log", ChatLog).write(Text(f"→ {label}", style="dim"))

    def _write_tool_result(self, label: str) -> None:
        self.query_one("#chat-log", ChatLog).write(Text(f"✓ {label}", style="dim"))

    @staticmethod
    def _snippet(text: str, limit: int = 80) -> str:
        s = text.replace("\n", " ").strip()
        return s[:limit] + "..." if len(s) > limit else s

    def _render_chat(self) -> None:
        self.query_one("#chat-log", ChatLog).clear()
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
    async def _run_agent(self, user_input: str, log: ChatLog) -> None:
        self.messages.append({"role": "user", "content": user_input})
        working_context = self._store.working_context()
        base_len = len(working_context)

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
        synced = False

        try:
            last_event_type = None
            async for event in run_turn(MAIN_AGENT, working_context):
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
            self.messages.extend(working_context[base_len:])
            synced = True
            flush_text()

            if not self._store.title:
                self._generate_title_bg()

            if bar.n_ctx and bar.last_prompt_tokens / bar.n_ctx >= COMPACT_THRESHOLD:
                await self._compact_chat()

        except asyncio.CancelledError:
            # _cancel_agent already wrote "Interrupted" and stopped the spinner.
            raise

        except Exception as e:
            log.write(f"[red]Agent error: {e}[/red]")

        finally:
            if not synced:
                self.messages.extend(working_context[base_len:])
            self._store.save()
            self.query_one(StatusBar).stop_thinking()

    @work
    async def _generate_title_bg(self) -> None:
        chat_id = self._store.id
        title = await generate_title(self.messages, MAIN_AGENT.model_name)
        if self._store.id == chat_id:
            self._store.set_title(title)
            self._store.save()

    async def _compact_chat(self) -> None:
        if self._compacting:
            return
        self._compacting = True
        progress_wgt = self.query_one(CompactionProgress)
        input_wgt = self.query_one("#input-box", Input)
        input_wgt.disabled = True
        progress_wgt.start()
        try:
            result = await compact(self._store)
            if result:
                delta, new_cov, summary_text = result
                old_cov = self._store.compaction["covers_through"]
                turn_offset = count_turns(self._store.messages[:old_cov])
                await asyncio.to_thread(
                    index_chat,
                    self._store.id,
                    self._store.title,
                    self._store.cwd,
                    delta,
                    turn_offset=turn_offset,
                )
                self._store.set_compaction(summary_text, new_cov)
                self._store.save()
        finally:
            self._compacting = False
            progress_wgt.stop()
            input_wgt.disabled = False
            input_wgt.focus()
        self.query_one("#chat-log", ChatLog).write("[dim]Chat compacted.[/dim]")

    @work
    async def _resume_chat(self) -> None:
        metas = self._store.list_chats(cwd=os.getcwd())
        if not metas:
            self.query_one("#chat-log", ChatLog).write("[dim]No saved chats.[/dim]")
            return
        chat_id = await self.push_screen_wait(ChatPickerScreen(metas))
        if chat_id is None:
            return
        self._store.load(chat_id)
        set_current_chat_id(self._store.id)
        self._render_chat()
        self.query_one("#input-box", Input).focus()


def run(working_dir: str = "."):
    app = TitanApp(working_dir=working_dir)
    app.run()


if __name__ == "__main__":
    run()
