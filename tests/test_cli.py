#!/usr/bin/env python
"""Tests for `cojopy` CLI."""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cojopy.cli import app

runner = CliRunner()


@pytest.fixture
def sample_sumstats():
    """Create sample summary statistics for testing."""
    return pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "A1": ["A", "C", "G", "T"],
            "A2": ["G", "T", "A", "C"],
            "b": [0.5, 0.3, 0.5, 0.1],
            "se": [0.1, 0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-8, 1e-5],
            "freq": [0.3, 0.4, 0.5, 0.6],
            "N": [1000, 1000, 1000, 1000],
        }
    )


@pytest.fixture
def sample_ld_matrix():
    """Create sample LD matrix for testing."""
    return np.array(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.1, 1.0, 0.4, 0.5],
            [0.2, 0.4, 1.0, 0.6],
            [0.3, 0.5, 0.6, 1.0],
        ],
    )


@pytest.fixture
def temp_files(sample_sumstats, sample_ld_matrix):
    """Create temporary files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save sumstats
        sumstats_path = os.path.join(temp_dir, "sumstats.txt")
        sample_sumstats.to_csv(sumstats_path, sep="\t", index=False)

        # Save LD matrix
        ld_path = os.path.join(temp_dir, "ld.txt")
        np.savetxt(ld_path, sample_ld_matrix)

        # Create output path
        output_path = os.path.join(temp_dir, "output.txt")

        yield {
            "sumstats_path": sumstats_path,
            "ld_path": ld_path,
            "output_path": output_path,
            "temp_dir": temp_dir,
        }


def test_cli_version():
    """Test CLI version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "COJO version:" in result.stdout


def test_cli_help():
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "COJO: Conditional & Joint Association Analysis" in result.stdout


def test_cli_slct_command(temp_files):
    """Test CLI slct command."""
    result = runner.invoke(
        app,
        [
            "slct",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])


def test_cli_slct_command_with_ld_freq(temp_files, sample_sumstats):
    """Test CLI slct command with LD frequency file."""
    # Create LD frequency file with small differences
    ld_freq_path = os.path.join(temp_files["temp_dir"], "ld_freq.txt")
    ld_freq = pd.DataFrame({"SNP": sample_sumstats["SNP"], "freq": [0.31, 0.41, 0.51, 0.61]})
    ld_freq.to_csv(ld_freq_path, sep="\t", index=False)

    result = runner.invoke(
        app,
        [
            "slct",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
            "--ld-freq",
            ld_freq_path,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])


def test_cli_joint_command(temp_files):
    """Test CLI joint command."""
    # Create extract SNPs file
    extract_snps_path = os.path.join(temp_files["temp_dir"], "extract_snps.txt")
    with open(extract_snps_path, "w") as f:
        f.write("rs1\nrs2\n")

    result = runner.invoke(
        app,
        [
            "joint",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
            "--extract-snps",
            extract_snps_path,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])


def test_cli_cond_command(temp_files):
    """Test CLI cond command."""
    # Create conditioning SNPs file
    cond_snps_path = os.path.join(temp_files["temp_dir"], "cond_snps.txt")
    with open(cond_snps_path, "w") as f:
        f.write("rs1\nrs2\n")

    result = runner.invoke(
        app,
        [
            "cond",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
            "--cond-snps",
            cond_snps_path,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])


def test_cli_cond_command_with_extract(temp_files):
    """Test CLI cond command with SNP extraction."""
    # Create conditioning and extract SNPs files
    cond_snps_path = os.path.join(temp_files["temp_dir"], "cond_snps.txt")
    extract_snps_path = os.path.join(temp_files["temp_dir"], "extract_snps.txt")

    with open(cond_snps_path, "w") as f:
        f.write("rs1\n")
    with open(extract_snps_path, "w") as f:
        f.write("rs3\n")

    result = runner.invoke(
        app,
        [
            "cond",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
            "--cond-snps",
            cond_snps_path,
            "--extract-snps",
            extract_snps_path,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])


def test_cli_missing_required_args():
    """Test CLI with missing required arguments."""
    result = runner.invoke(app, ["slct"])
    assert result.exit_code != 0
    assert "Missing option" in result.stdout


def test_cli_invalid_file_paths():
    """Test CLI with invalid file paths."""
    result = runner.invoke(
        app,
        [
            "slct",
            "--sumstats",
            "nonexistent.txt",
            "--ld-matrix",
            "nonexistent.txt",
            "--output",
            "output.txt",
        ],
    )
    assert result.exit_code != 0


def test_cli_verbose_mode():
    """Test CLI verbose mode."""
    with patch("logging.getLogger") as mock_logger:
        result = runner.invoke(app, ["--verbose"])
        assert result.exit_code == 0
        mock_logger.assert_called()


def test_cli_slct_command_with_collinearity(temp_files):
    """Test CLI slct command with high collinearity."""
    # Modify LD matrix to create high collinearity
    ld_matrix = pd.read_csv(temp_files["ld_path"], sep="\t", index_col=0)
    ld_matrix.iloc[:] = 0.95
    ld_matrix.to_csv(temp_files["ld_path"], sep="\t")

    result = runner.invoke(
        app,
        [
            "slct",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
            "--collinear-cutoff",
            "0.9",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])


def test_cli_slct_command_with_window(temp_files):
    """Test CLI slct command with window size."""
    result = runner.invoke(
        app,
        [
            "slct",
            "--sumstats",
            temp_files["sumstats_path"],
            "--ld-matrix",
            temp_files["ld_path"],
            "--output",
            temp_files["output_path"],
            "--window-size",
            "1000",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(temp_files["output_path"])
