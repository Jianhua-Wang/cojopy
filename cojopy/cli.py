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
    sumstats: str = typer.Option(
        ...,
        "--sumstats",
        "-s",
        help="Path to the summary statistics file. The file should have the same columns as COJO input: SNP, A1, A2, b, se, p, freq, N",
    ),
    ld_matrix: str = typer.Option(
        ...,
        "--ld-matrix",
        "-l",
        help="Path to the LD matrix file.",
    ),
    output: str = typer.Option(..., "--output", "-o", help="Path to the output file."),
    ld_freq: str = typer.Option(
        None,
        "--ld-freq",
        "-f",
        help="Path to the LD frequency file. The file should have the following columns: SNP, freq. Use freq in sumstats if ld_freq is not provided.",
    ),
    p_cutoff: float = typer.Option(5e-8, "--p-cutoff", "-p", help="P-value cutoff."),
    collinear_cutoff: float = typer.Option(0.9, "--collinear-cutoff", "-c", help="Collinearity cutoff."),
    window_size: int = typer.Option(10000000, "--window-size", "-w", help="Window size."),
    maf_cutoff: float = typer.Option(0.01, "--maf-cutoff", "-m", help="Minor allele frequency cutoff."),
    diff_freq_cutoff: float = typer.Option(
        0.2, "--diff-freq-cutoff", "-d", help="Difference in minor allele frequency cutoff."
    ),
):
    """Perform conditional selection of SNPs using COJO algorithm."""
    c = COJO(
        p_cutoff=p_cutoff,
        collinear_cutoff=collinear_cutoff,
        window_size=window_size,
        maf_cutoff=maf_cutoff,
        diff_freq_cutoff=diff_freq_cutoff,
    )
    cojo_result = c.conditional_selection(sumstats, ld_matrix, ld_freq)
    cojo_result.to_csv(output, sep="\t", index=False, float_format="%.6g")


@app.command(name="joint", help="Joint analysis of SNPs.")
def joint(
    sumstats: str = typer.Option(
        ...,
        "--sumstats",
        "-s",
        help="Path to the summary statistics file. The file should have the same columns as COJO input: SNP, A1, A2, b, se, p, freq, N",
    ),
    ld_matrix: str = typer.Option(
        ...,
        "--ld-matrix",
        "-l",
        help="Path to the LD matrix file.",
    ),
    output: str = typer.Option(..., "--output", "-o", help="Path to the output file."),
    ld_freq: str = typer.Option(
        None,
        "--ld-freq",
        "-f",
        help="Path to the LD frequency file. The file should have the following columns: SNP, freq. Use freq in sumstats if ld_freq is not provided.",
    ),
    extract_snps: str = typer.Option(
        None,
        "--extract-snps",
        "-e",
        help="Path to the file containing the SNPs to extract. Each SNP should be in a new line.",
    ),
    p_cutoff: float = typer.Option(5e-8, "--p-cutoff", "-p", help="P-value cutoff."),
    collinear_cutoff: float = typer.Option(0.9, "--collinear-cutoff", "-c", help="Collinearity cutoff."),
    window_size: int = typer.Option(10000000, "--window-size", "-w", help="Window size."),
    maf_cutoff: float = typer.Option(0.01, "--maf-cutoff", "-m", help="Minor allele frequency cutoff."),
    diff_freq_cutoff: float = typer.Option(
        0.2, "--diff-freq-cutoff", "-d", help="Difference in minor allele frequency cutoff."
    ),
):
    """Perform joint analysis of SNPs using COJO algorithm."""
    logger = logging.getLogger("COJO")
    c = COJO(
        p_cutoff=p_cutoff,
        collinear_cutoff=collinear_cutoff,
        window_size=window_size,
        maf_cutoff=maf_cutoff,
        diff_freq_cutoff=diff_freq_cutoff,
    )
    joint_result = c.run_joint_analysis(sumstats, ld_matrix, extract_snps, ld_freq)
    logger.info("Joint analysis complete. Writing results to %s", output)
    joint_result.to_csv(output, sep="\t", index=False, float_format="%.6g")


@app.command(name="cond", help="Conditional analysis of SNPs.")
def cond(
    sumstats: str = typer.Option(
        ...,
        "--sumstats",
        "-s",
        help="Path to the summary statistics file. The file should have the same columns as COJO input: SNP, A1, A2, b, se, p, freq, N",
    ),
    ld_matrix: str = typer.Option(
        ...,
        "--ld-matrix",
        "-l",
        help="Path to the LD matrix file.",
    ),
    output: str = typer.Option(..., "--output", "-o", help="Path to the output file."),

    cond_snps: str = typer.Option(
        ...,
        "--cond-snps",
        "-c",
        help="Path to the file containing the SNPs to condition on. Each SNP should be in a new line.",
    ),
    extract_snps: str = typer.Option(
        None,
        "--extract-snps",
        "-e",
        help="Path to the file containing the SNPs to extract. Each SNP should be in a new line.",
    ),
    ld_freq: str = typer.Option(
        None,
        "--ld-freq",
        "-f",
        help="Path to the LD frequency file. The file should have the following columns: SNP, freq. Use freq in sumstats if ld_freq is not provided.",
    ),
    p_cutoff: float = typer.Option(5e-8, "--p-cutoff", "-p", help="P-value cutoff."),
    collinear_cutoff: float = typer.Option(0.9, "--collinear-cutoff", "-c", help="Collinearity cutoff."),
    window_size: int = typer.Option(10000000, "--window-size", "-w", help="Window size."),
    maf_cutoff: float = typer.Option(0.01, "--maf-cutoff", "-m", help="Minor allele frequency cutoff."),
    diff_freq_cutoff: float = typer.Option(
        0.2, "--diff-freq-cutoff", "-d", help="Difference in minor allele frequency cutoff."
    ),
):
    """Perform conditional analysis of SNPs using COJO algorithm."""
    logger = logging.getLogger("COJO")
    c = COJO(
        p_cutoff=p_cutoff,
        collinear_cutoff=collinear_cutoff,
        window_size=window_size,
        maf_cutoff=maf_cutoff,
        diff_freq_cutoff=diff_freq_cutoff,
    )
    cond_result = c.run_conditional_analysis(sumstats, ld_matrix, cond_snps, extract_snps, ld_freq)
    logger.info("Conditional analysis complete. Writing results to %s", output)
    cond_result.to_csv(output, sep="\t", index=False, float_format="%.6g")


if __name__ == "__main__":
    app(main)
