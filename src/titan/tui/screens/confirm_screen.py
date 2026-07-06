from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList


class ConfirmScreen(ModalScreen[str]):
    DEFAULT_CSS = """
    ConfirmScreen > #confirm-bar {
        dock: bottom;
        width: 100%;
        height: auto;
        padding: 1 2;
    }
    ConfirmScreen #confirm-message {
        padding-bottom: 1;
    }
    ConfirmScreen OptionList {
        height: auto;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self, message: str, body=None, options=(("Yes", "yes"), ("No", "no"))):
        super().__init__()
        self._message = message
        self._body = body
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-bar"):
            yield Label(self._message, id="confirm-message")
            if self._body is not None:
                yield self._body
            yield OptionList(
                *[f"{i}. {label}" for i, (label, _) in enumerate(self._options, 1)]
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._options[event.option_index][1])

    def on_key(self, event: Key) -> None:
        for i, (label, value) in enumerate(self._options, 1):
            if event.key == str(i) or event.key == label[0].lower():
                event.stop()
                self.dismiss(value)
                return

    def action_cancel(self) -> None:
        self.dismiss(self._options[-1][1])


async def request_approval(
    app,
    message: str,
    body=None,
    options=(("Yes", "yes"), ("No", "no")),
    force: bool = False,
) -> str:
    if not force and getattr(app, "auto_accept", False):
        return options[0][1]
    return await app.push_screen_wait(ConfirmScreen(message, body, options))
