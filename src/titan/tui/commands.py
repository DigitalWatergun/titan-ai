import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from textual.widgets import RichLog

from titan.chats.rag import set_current_chat_id
from titan.rag.store import client
from titan.tui.widgets.status_bar import StatusBar

# Need to write this here for type checking due to circular dependency. Not needed for runtime
if TYPE_CHECKING:
    from titan.tui.app import TitanApp


@dataclass(frozen=True)
class CommandContext:
    """Everything a slash-command handler might need. Built fresh per invocation."""

    app: "TitanApp"
    args: str
    log: RichLog


CommandHandler = Callable[["CommandContext"], None]
COMMANDS: dict[str, CommandHandler] = {}


def command(name: str) -> Callable[[CommandHandler], CommandHandler]:
    """Decorator: register a function as the handler for the `name` (e.g. '/clear')."""

    def decorator(fn: CommandHandler) -> CommandHandler:
        COMMANDS[name] = fn
        return fn

    return decorator


@command("/quit")
def quit_app(ctx: CommandContext) -> None:
    ctx.app.exit()


@command("/clear")
def clear_chat(ctx: CommandContext) -> None:
    ctx.app._store.new()
    set_current_chat_id(ctx.app._store.id)
    ctx.app.query_one(StatusBar).reset_stats()
    ctx.log.clear()
    ctx.log.write("[dim]Chat cleared.[/dim]")


@command("/help")
def show_help(ctx: CommandContext) -> None:
    ctx.log.write("\n[bold]Available commands:[/bold]")
    ctx.log.write("  [dim]/clear        — clear chat[/dim]")
    ctx.log.write(
        "  [dim]/index        — index current directory (or /index <path>)[/dim]"
    )
    ctx.log.write("  [dim]/index vault  — index the Obsidian vault (.md only)[/dim]")
    ctx.log.write("  [dim]/index list   — show indexed projects[/dim]")
    ctx.log.write(
        "  [dim]/index delete — remove an index (/index delete <name|all>)[/dim]"
    )
    ctx.log.write("  [dim]/help         — show this help[/dim]")
    ctx.log.write("  [dim]/quit         — exit[/dim]")
    ctx.log.write("")
    ctx.log.write("[bold]Keyboard shortcuts:[/bold]")
    ctx.log.write("  [dim]Esc           — interrupt current response[/dim]")
    ctx.log.write("  [dim]Ctrl+O        — toggle full/summary thinking mode[/dim]")
    ctx.log.write("  [dim]Ctrl+E        — toggle recent chats[/dim]")
    ctx.log.write("  [dim]PageUp        — scroll up[/dim]")
    ctx.log.write("  [dim]PageDn        — scroll down[/dim]")


@command("/index")
def index_directory(ctx: CommandContext) -> None:
    arg = ctx.args.strip()
    parts = arg.split(maxsplit=1)
    verb = parts[0] if parts else ""

    if verb == "list":
        for c in client().list_collections():
            m = c.metadata or {}
            ctx.log.write(f"  {m.get('project_path', '?')} — {c.count()} chunks")
        return

    if verb == "vault":
        directory = Path(os.environ["OBSIDIAN_VAULT_DIR"]).expanduser().resolve()
        ctx.log.write(f"[dim]Indexing {directory}...[/dim]")
        ctx.app._run_index(directory, "vault", ctx.log)
        return

    if verb in ("delete", "rm"):
        target = parts[1].strip() if len(parts) > 1 else os.getcwd()
        c = client()
        cols = c.list_collections()
        if target == "all":
            for col in cols:
                c.delete_collection(col.name)
            ctx.log.write(f"[dim]Deleted {len(cols)} index(es).[/dim]")
            return
        matches = [
            col
            for col in cols
            if target == col.name
            or target in (col.metadata or {}).get("project_path", "")
        ]
        if not matches:
            ctx.log.write(f"[dim]No index matching '{target}'.[/dim]")
        elif len(matches) > 1:
            ctx.log.write(
                f"[dim]'{target}' matches {len(matches)} — be more specific:[/dim]"
            )
            for col in matches:
                ctx.log.write(
                    f"  {col.name} — {(col.metadata or {}).get('project_path', '?')}"
                )
        else:
            c.delete_collection(matches[0].name)
            ctx.log.write(
                f"[dim]Deleted {(matches[0].metadata or {}).get('project_path', '?')}[/dim]"
            )
        return

    if verb == "chats":
        ctx.log.write("[dim]Indexing chats...[/dim]")
        ctx.app._run_index(str(ctx.app._store.base_dir), "chats", ctx.log)
        return

    directory = arg or "."
    vault_dir = os.environ.get("OBSIDIAN_VAULT_DIR")
    here = Path(directory).expanduser().resolve()
    if vault_dir and here == Path(vault_dir).expanduser().resolve():
        ctx.log.write("[dim]Detected vault — indexing notes (.md only)...[/dim]")
        ctx.app._run_index(directory, "vault", ctx.log)
        return

    display_dir = os.path.abspath(directory) if directory == "." else directory
    ctx.log.write(f"[dim]Indexing {display_dir}...[/dim]")
    ctx.app._run_index(directory, "codebase", ctx.log)


@command("/compact")
def compact_now(ctx: CommandContext) -> None:
    if ctx.app._is_agent_running() or ctx.app._compacting:
        ctx.log.write("[dim]Busy — try again in a moment.[/dim]")
        return
    ctx.app.run_worker(ctx.app._compact_chat())
    ctx.log.write("[dim]Compacting chat...[/dim]")
