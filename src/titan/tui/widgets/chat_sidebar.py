from textual.containers import Vertical
from textual.widgets import Label, ListItem, ListView

from titan.chats.store import ChatMeta


class ChatItem(ListItem):
    """ListView row that remembers which chat it represents by id."""

    def __init__(self, meta: ChatMeta) -> None:
        super().__init__(Label(meta.title or "Untitled"))
        self.chat_id = meta.id


class ChatSidebar(Vertical):
    """Left-docked side panel listing saved chats."""

    def compose(self):
        yield Label("Recent Chats", id="chat-sidebar-title")
        yield ListView(id="chat-list")

    async def populate(self, metas: list[ChatMeta]) -> None:
        list_view = self.query_one("#chat-list", ListView)
        await list_view.clear()
        await list_view.extend(ChatItem(m) for m in metas)
        list_view.index = 0 if metas else None
