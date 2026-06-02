import logging

import typer
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers_modules").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    working_dir: str = typer.Option(".", help="Working directory for the agent"),
):
    """Titan — Local Coding Assistant"""
    from titan.tui.app import run

    run(working_dir=working_dir)


if __name__ == "__main__":
    app()
