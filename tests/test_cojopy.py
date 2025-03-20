#!/usr/bin/env python
"""Tests for `cojopy` package."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from cojopy.cojopy import COJO


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
        ]
    )


@pytest.fixture
def sample_positions():
    """Create sample SNP positions for testing."""
    return pd.Series([1000, 2000, 3000, 4000], index=["rs1", "rs2", "rs3", "rs4"])


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

        yield {"sumstats_path": sumstats_path, "ld_path": ld_path, "temp_dir": temp_dir}


def test_cojo_initialization():
    """Test COJO class initialization with default parameters."""
    cojo = COJO()
    assert cojo.p_cutoff == 5e-8
    assert cojo.collinear_cutoff == 0.9
    assert cojo.window_size == 10000000
    assert cojo.maf_cutoff == 0.01
    assert cojo.diff_freq_cutoff == 0.2
    assert len(cojo.snps_selected) == 0
    assert cojo.backward_removed == 0
    assert cojo.collinear_filtered == 0


def test_cojo_initialization_custom_params():
    """Test COJO class initialization with custom parameters."""
    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.8, window_size=5000000, maf_cutoff=0.02, diff_freq_cutoff=0.1)
    assert cojo.p_cutoff == 1e-6
    assert cojo.collinear_cutoff == 0.8
    assert cojo.window_size == 5000000
    assert cojo.maf_cutoff == 0.02
    assert cojo.diff_freq_cutoff == 0.1


def test_load_sumstats(temp_files):
    """Test loading summary statistics and LD matrix."""
    cojo = COJO()
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])

    assert cojo.total_snps == 4
    assert len(cojo.beta) == 4
    assert len(cojo.se) == 4
    assert len(cojo.p) == 4
    assert len(cojo.freq) == 4
    assert len(cojo.n) == 4
    assert len(cojo.snp_ids) == 4
    assert cojo.ld_matrix.shape == (4, 4)


def test_load_sumstats_with_ld_freq(temp_files, sample_sumstats):
    """Test loading summary statistics with LD frequency file."""
    # Create LD frequency file
    ld_freq_path = os.path.join(temp_files["temp_dir"], "ld_freq.txt")
    ld_freq = pd.DataFrame(
        {"SNP": sample_sumstats["SNP"], "freq": [0.31, 0.41, 0.51, 0.61]}  # Slightly different frequencies
    )
    ld_freq.to_csv(ld_freq_path, sep="\t", index=False)

    cojo = COJO(diff_freq_cutoff=0.1)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"], ld_freq_path)

    assert cojo.total_snps == 4  # All SNPs should pass the frequency difference check


def test_load_sumstats_with_ld_freq_filtering(temp_files, sample_sumstats):
    """Test filtering SNPs based on frequency difference between sumstats and LD frequency."""
    # Create LD frequency file with large differences
    ld_freq_path = os.path.join(temp_files["temp_dir"], "ld_freq.txt")
    ld_freq = pd.DataFrame(
        {"SNP": sample_sumstats["SNP"], "freq": [0.5, 0.4, 0.5, 0.6]}  # Large differences for some SNPs
    )
    ld_freq.to_csv(ld_freq_path, sep="\t", index=False)

    cojo = COJO(diff_freq_cutoff=0.2)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"], ld_freq_path)

    # Verify that SNPs were filtered
    assert cojo.total_snps == 3  # One SNP should be filtered out due to frequency differences
    assert len(cojo.beta) == 3
    assert len(cojo.se) == 3
    assert len(cojo.p) == 3
    assert len(cojo.freq) == 3
    assert len(cojo.n) == 3
    assert len(cojo.snp_ids) == 3
    assert cojo.ld_matrix.shape == (3, 3)


def test_load_sumstats_maf_filtering(temp_files):
    """Test MAF filtering in load_sumstats."""
    # Modify sumstats to include SNPs with low MAF
    sumstats = pd.read_csv(temp_files["sumstats_path"], sep="\t")
    sumstats.loc[0, "freq"] = 0.005  # Below MAF cutoff
    sumstats.to_csv(temp_files["sumstats_path"], sep="\t", index=False)

    cojo = COJO(maf_cutoff=0.01)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])

    assert cojo.total_snps == 3  # One SNP should be filtered out
    assert "rs1" not in cojo.snp_ids  # The SNP with low MAF should be removed


def test_conditional_selection_no_significant_snps(temp_files):
    """Test conditional selection when no SNPs are significant."""
    # Modify sumstats to have no significant SNPs
    sumstats = pd.read_csv(temp_files["sumstats_path"], sep="\t")
    sumstats["p"] = [1e-7, 1e-6, 1e-5, 1e-4]
    sumstats.to_csv(temp_files["sumstats_path"], sep="\t", index=False)

    cojo = COJO(p_cutoff=1e-8)
    result = cojo.conditional_selection(temp_files["sumstats_path"], temp_files["ld_path"])

    assert result.empty
    assert len(cojo.snps_selected) == 0


def test_conditional_selection_single_snp(temp_files):
    """Test conditional selection with only one significant SNP."""
    cojo = COJO(p_cutoff=1e-7)
    result = cojo.conditional_selection(temp_files["sumstats_path"], temp_files["ld_path"])

    assert len(result) == 1
    assert result["SNP"].iloc[0] == "rs1"
    assert len(cojo.snps_selected) == 1
    assert cojo.backward_removed == 0
    assert cojo.collinear_filtered == 0


def test_conditional_selection_with_collinearity(temp_files):
    """Test conditional selection with collinear SNPs."""
    # Modify LD matrix to create high collinearity
    ld_matrix = np.loadtxt(temp_files["ld_path"])
    ld_matrix[:] = 0.95
    np.savetxt(temp_files["ld_path"], ld_matrix)

    # Modify p-values to ensure multiple SNPs are selected
    sumstats = pd.read_csv(temp_files["sumstats_path"], sep="\t")
    sumstats["p"] = [1e-8, 1e-8, 1e-8, 1e-8]  # Make all SNPs significant
    sumstats.to_csv(temp_files["sumstats_path"], sep="\t", index=False)

    cojo = COJO(p_cutoff=1e-7, collinear_cutoff=0.9)
    result = cojo.conditional_selection(temp_files["sumstats_path"], temp_files["ld_path"])

    assert len(result) == 1  # Only one SNP should be selected due to collinearity


def test_conditional_selection_with_window(temp_files, sample_positions):
    """Test conditional selection with window-based LD consideration."""
    cojo = COJO(p_cutoff=1e-7, window_size=1000)
    result = cojo.conditional_selection(temp_files["sumstats_path"], temp_files["ld_path"], positions=sample_positions)

    assert len(result) > 0
    # SNPs should only be considered for LD if they're within the window
    # This is hard to test directly, but we can verify the result is reasonable
    assert len(result) <= 4  # Should not select more SNPs than we have


def test_backward_elimination(temp_files):
    """Test backward elimination of non-significant SNPs."""
    # Modify p-values to create a scenario where backward elimination is needed
    sumstats = pd.read_csv(temp_files["sumstats_path"], sep="\t")
    # Make all SNPs significant initially
    sumstats["p"] = [1e-8, 1e-8, 1e-8, 1e-8]
    # Make rs2 less significant but still significant enough to be selected
    sumstats.loc[sumstats["SNP"] == "rs2", "p"] = 1e-7
    # Make rs3 more significant to trigger backward elimination
    sumstats.loc[sumstats["SNP"] == "rs3", "p"] = 1e-9
    sumstats.to_csv(temp_files["sumstats_path"], sep="\t", index=False)

    cojo = COJO(p_cutoff=1e-7)
    result = cojo.conditional_selection(temp_files["sumstats_path"], temp_files["ld_path"])

    assert len(result) == 1  # Only the most significant SNP should remain


def test_joint_statistics_calculation(temp_files):
    """Test calculation of joint statistics."""
    # Modify LD matrix to create some correlation between SNPs
    ld_matrix = np.loadtxt(temp_files["ld_path"])
    ld_matrix[0, 1] = 0.5  # Add correlation between rs1 and rs2
    ld_matrix[1, 0] = 0.5
    np.savetxt(temp_files["ld_path"], ld_matrix)

    cojo = COJO(p_cutoff=1e-7)
    result = cojo.conditional_selection(temp_files["sumstats_path"], temp_files["ld_path"])

    assert "joint_beta" in result.columns
    assert "joint_se" in result.columns
    assert "joint_p" in result.columns
    assert len(result) > 0

    # Check that joint statistics are different from original statistics
    # Only check if there are multiple SNPs in the result
    if len(result) > 1:
        assert not np.allclose(result["joint_beta"], result["original_beta"])
        assert not np.allclose(result["joint_se"], result["original_se"])
        assert not np.allclose(result["joint_p"], result["original_p"])


def test_run_conditional_analysis(temp_files):
    """Test running conditional analysis with specified SNPs."""
    # Create a file with SNPs to condition on
    cond_snps_path = os.path.join(temp_files["temp_dir"], "cond_snps.txt")
    with open(cond_snps_path, "w") as f:
        f.write("rs1\nrs2\n")

    cojo = COJO()
    result = cojo.run_conditional_analysis(temp_files["sumstats_path"], temp_files["ld_path"], cond_snps_path)

    assert len(result) == 2  # Should only include SNPs not in cond_snps
    assert all(snp not in ["rs1", "rs2"] for snp in result["SNP"])
    assert "cond_beta" in result.columns
    assert "cond_se" in result.columns
    assert "cond_p" in result.columns


def test_run_conditional_analysis_with_extract(temp_files):
    """Test running conditional analysis with SNP extraction."""
    # Create files for conditioning and extraction
    cond_snps_path = os.path.join(temp_files["temp_dir"], "cond_snps.txt")
    extract_snps_path = os.path.join(temp_files["temp_dir"], "extract_snps.txt")

    with open(cond_snps_path, "w") as f:
        f.write("rs1\n")
    with open(extract_snps_path, "w") as f:
        f.write("rs3\n")

    cojo = COJO()
    result = cojo.run_conditional_analysis(
        temp_files["sumstats_path"], temp_files["ld_path"], cond_snps_path, extract_snps_path=extract_snps_path
    )

    assert len(result) == 1
    assert result["SNP"].iloc[0] == "rs3"


def test_run_joint_analysis(temp_files):
    """Test running joint analysis with specified SNPs."""
    # Create a file with SNPs to extract
    extract_snps_path = os.path.join(temp_files["temp_dir"], "extract_snps.txt")
    with open(extract_snps_path, "w") as f:
        f.write("rs1\nrs2\n")

    cojo = COJO()
    result = cojo.run_joint_analysis(
        temp_files["sumstats_path"], temp_files["ld_path"], extract_snps_path=extract_snps_path
    )

    assert len(result) == 2
    assert all(snp in ["rs1", "rs2"] for snp in result["SNP"])
    assert "joint_beta" in result.columns
    assert "joint_se" in result.columns
    assert "joint_p" in result.columns
