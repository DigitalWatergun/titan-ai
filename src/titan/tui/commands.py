from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from textual.widgets import RichLog

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
def clear_conversation(ctx: CommandContext) -> None:
    ctx.app.messages = []
    ctx.app.query_one(StatusBar).reset_stats()
    ctx.log.clear()
    ctx.log.write("[dim]Conversation cleared.[/dim]")


@command("/help")
def show_help(ctx: CommandContext) -> None:
    ctx.log.write("\n[bold]Available commands:[/bold]")
    ctx.log.write("  [dim]/clear   — clear conversation[/dim]")
    ctx.log.write("  [dim]/index   — index current directory (or /index <path>)[/dim]")
    ctx.log.write("  [dim]/help    — show this help[/dim]")
    ctx.log.write("  [dim]/quit    — exit[/dim]")
    ctx.log.write("")
    ctx.log.write("[bold]Keyboard shortcuts:[/bold]")
    ctx.log.write("  [dim]Esc      — interrupt current response[/dim]")
    ctx.log.write("  [dim]Ctrl+O   — toggle full/summary thinking mode[/dim]")
    ctx.log.write("  [dim]PageUp   — scroll up[/dim]")
    ctx.log.write("  [dim]PageDn   — scroll down[/dim]")


@command("/index")
def index_directory(ctx: CommandContext) -> None:
    import os

    directory = ctx.args.strip() or "."
    display_dir = os.path.abspath(directory) if directory == "." else directory
    ctx.log.write(f"[dim]Indexing {display_dir}...[/dim]")
    ctx.app._run_index(directory, display_dir, ctx.log)
