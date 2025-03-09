"""Console script for cojopy."""

import logging

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from cojopy import __version__
from cojopy.cojopy import COJO

app = typer.Typer()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(context_settings=CONTEXT_SETTINGS, add_completion=False)


@app.callback(invoke_without_command=True, no_args_is_help=True)
def main(
    version: bool = typer.Option(False, "--version", "-V", help="Show version."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose info."),
):
    """COJO: Conditional & Joint Association Analysis."""
    console = Console()
    console.rule("[bold blue]COJO[/bold blue]")
    console.print(f"Version: {__version__}", justify="center")
    console.print("Author: Jianhua Wang", justify="center")
    console.print("Email: jianhua.mert@gmail.com", justify="center")
    if version:
        typer.echo(f"COJO version: {__version__}")
        raise typer.Exit()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Verbose mode is on.")
    else:
        for name in [
            "COJO",
        ]:
            logging.getLogger(name).setLevel(logging.INFO)


@app.command(name="slct", help="Conditional selection of SNPs.")
def slct(
    sumstats: str = typer.Argument(..., help="Path to the summary statistics file."),
    ld_matrix: str = typer.Argument(..., help="Path to the LD matrix file."),
    p_cutoff: float = typer.Option(5e-8, "--p-cutoff", "-p", help="P-value cutoff."),
    collinear_cutoff: float = typer.Option(
        0.9, "--collinear-cutoff", "-c", help="Collinearity cutoff."
    ),
    window_size: int = typer.Option(
        10000000, "--window-size", "-w", help="Window size."
    ),
    output: str = typer.Option(..., "--output", "-o", help="Path to the output file."),
):
    """Perform conditional selection of SNPs using COJO algorithm."""
    c = COJO(
        p_cutoff=p_cutoff, collinear_cutoff=collinear_cutoff, window_size=window_size
    )
    sumstats_df = pd.read_csv(sumstats, sep="\t")
    ld_matrix_array = np.loadtxt(ld_matrix)
    cojo_result = c.conditional_selection(sumstats_df, ld_matrix_array)
    cojo_result.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    app(main)
