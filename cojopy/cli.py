"""Console script for cojopy."""

import typer

app = typer.Typer()


def main():
    """Main entrypoint."""
    typer.echo("cojopy")
    typer.echo("=" * len("cojopy"))
    typer.echo("Conditional Analysis with LD Matrix")


if __name__ == "__main__":
    app(main)
