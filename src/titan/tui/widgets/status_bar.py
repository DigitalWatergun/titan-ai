import time

from textual.containers import Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static


class StatusBar(Horizontal):
    """Bottom status bar: title, elapsed time, token count, context %, thinking spinner."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        margin-top: 1;
        background: #1e1e1e;
        color: #666666;
        padding: 0 1;
    }
    StatusBar > #status-left {
        width: 1fr;
        height: 1;
        color: #4a90c2;
        background: #1e1e1e;
    }
    StatusBar > #status-right {
        width: auto;
        height: 1;
        color: #666666;
        background: #1e1e1e;
    }
    """

    total_completion_tokens: reactive[int] = reactive(0)
    last_prompt_tokens: reactive[int] = reactive(0)
    n_ctx: reactive[int] = reactive(0)
    thinking: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timer: Timer | None = None
        self._start_time = 0.0
        self._elapsed = 0.0
        self._dots = 0
        self._live_tokens = 0

    def compose(self):
        yield Static("Titan", id="status-left")
        yield Static("/help", id="status-right")

    def start_thinking(self) -> None:
        self._start_time = time.monotonic()
        self._elapsed = 0.0
        self._dots = 0
        self._live_tokens = 0
        self.thinking = True
        self._timer = self.set_interval(0.4, self._tick_animation)
        self.query_one("#status-right", Static).update("Press ESC to interrupt")

    def add_live_token(self, n: int = 1) -> None:
        self._live_tokens += n
        self._refresh_display()

    def commit_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._live_tokens = 0
        self.last_prompt_tokens = prompt_tokens
        self.total_completion_tokens += completion_tokens

    def _tick_animation(self) -> None:
        self._dots = (self._dots % 3) + 1
        self._refresh_display()

    def stop_thinking(self) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None
        if self.thinking:
            self._elapsed = time.monotonic() - self._start_time
        self.thinking = False
        self._refresh_display()
        self.query_one("#status-right", Static).update("/help")

    def reset_stats(self) -> None:
        self._live_tokens = 0
        self.total_completion_tokens = 0
        self.last_prompt_tokens = 0
        self._elapsed = 0.0
        self._refresh_display()

    def watch_total_completion_tokens(self, *_):
        self._refresh_display()

    def watch_last_prompt_tokens(self, *_):
        self._refresh_display()

    def watch_n_ctx(self, *_):
        self._refresh_display()

    def _refresh_display(self) -> None:
        elapsed = (time.monotonic() - self._start_time) if self.thinking else self._elapsed
        h, remainder = divmod(int(elapsed), 3600)
        m, s = divmod(remainder, 60)
        time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        parts = ["Titan", time_str]
        shown_tokens = self.total_completion_tokens + self._live_tokens
        if shown_tokens > 0:
            parts.append(f"{shown_tokens:,} tokens")
        if self.n_ctx > 0 and self.last_prompt_tokens > 0:
            pct = int(self.last_prompt_tokens / self.n_ctx * 100)
            parts.append(f"{pct}% ctx")
        if self.thinking:
            parts.append(f"Thinking{'.' * self._dots}")
        self.query_one("#status-left", Static).update("  ".join(parts))
