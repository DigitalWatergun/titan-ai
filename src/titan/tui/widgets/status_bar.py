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

    total_tokens: reactive[int] = reactive(0)
    last_prompt_tokens: reactive[int] = reactive(0)
    n_ctx: reactive[int] = reactive(0)
    thinking: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timer: Timer | None = None
        self._start_time = 0.0
        self._dots = 0

    def compose(self):
        yield Static("Titan", id="status-left")
        yield Static("/help", id="status-right")

    def start_thinking(self) -> None:
        self._start_time = time.monotonic()
        self._dots = 0
        self.thinking = True
        self._timer = self.set_interval(0.4, self._tick_animation)
        self.query_one("#status-right", Static).update("Press ESC to interrupt")

    def _tick_animation(self) -> None:
        self._dots = (self._dots % 3) + 1
        self._refresh_display(f"Thinking{'.' * self._dots}")

    def stop_thinking(self) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None
        self.thinking = False
        self._refresh_display()
        self.query_one("#status-right", Static).update("/help")

    def watch_total_tokens(self, *_):
        self._refresh_display()

    def watch_last_prompt_tokens(self, *_):
        self._refresh_display()

    def watch_n_ctx(self, *_):
        self._refresh_display()

    def _refresh_display(self, suffix: str = "") -> None:
        elapsed = time.monotonic() - self._start_time if self.thinking else 0
        h, remainder = divmod(int(elapsed), 3600)
        m, s = divmod(remainder, 60)
        time_str = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        parts = ["Titan", time_str]
        if self.total_tokens > 0:
            parts.append(f"{self.total_tokens:,} tokens")
        if self.n_ctx > 0 and self.last_prompt_tokens > 0:
            pct = int(self.last_prompt_tokens / self.n_ctx * 100)
            parts.append(f"{pct}% ctx")
        if suffix:
            parts.append(suffix)
        self.query_one("#status-left", Static).update("  ".join(parts))
