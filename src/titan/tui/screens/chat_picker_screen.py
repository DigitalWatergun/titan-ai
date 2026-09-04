from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from titan.chats.store import ChatMeta


class ChatPickerItem(ListItem):
    def __init__(self, meta: ChatMeta) -> None:
        super().__init__(Label(meta.title or "Untitled"))
        self.chat_id = meta.id


class ChatPickerScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("j", "cursor_down", ""),
        ("k", "cursor_up", ""),
        ("G", "cursor_bottom", ""),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, metas: list[ChatMeta]) -> None:
        super().__init__()
        self._metas = metas
        self._pending_g = False

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label("Resume a conversation", id="picker-title")
            yield ListView(
                *[ChatPickerItem(meta) for meta in self._metas], id="picker-list"
            )

    def on_mount(self) -> None:
        self.query_one("#picker-list", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ChatPickerItem):
            self.dismiss(event.item.chat_id)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        self.query_one("#picker-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#picker-list", ListView).action_cursor_up()

    def action_cursor_bottom(self) -> None:
        self.query_one("#picker-list", ListView).index = len(self._metas) - 1

    def on_key(self, event: Key) -> None:
        if event.key == "g":
            if self._pending_g:
                self._pending_g = False
                self.query_one("#picker-list", ListView).index = 0
            else:
                self._pending_g = True
        else:
            self._pending_g = False
