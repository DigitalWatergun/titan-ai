import typer

app = typer.Typer()


@app.command()
def chat():
    print("Titan is running!")


if __name__ == "__main__":
    app()
