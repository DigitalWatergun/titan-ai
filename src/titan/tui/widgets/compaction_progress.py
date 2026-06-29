import time

from textual.containers import Horizontal
from textual.widgets import ProgressBar, Static


class CompactionProgress(Horizontal):
    DEFAULT_CSS = "CompactionProgress { display: none; height: 1; }"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timer = None
        self._start = 0.0

    def compose(self):
        yield Static("Compacting chat…", id="compact-label")
        yield ProgressBar(total=None, show_eta=False, show_percentage=False)

    def start(self) -> None:
        self._start = time.monotonic()
        self.display = True
        self._timer = self.set_interval(1.0, self._tick)

    def stop(self) -> None:
        if self._timer:
            self._timer.stop()
        self.display = False

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._start)
        m, s = divmod(elapsed, 60)
        label = f"Compacting chat… ({m}m {s:02d}s)" if m else f"Compacting chat… ({s}s)"
        self.query_one("#compact-label", Static).update(label)
