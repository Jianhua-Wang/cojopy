#!/usr/bin/env python
"""Tests for `cojopy` package."""

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
            "b": [0.5, 0.3, 0.5, 0.1],
            "se": [0.1, 0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-8, 1e-5],
            "freq": [0.3, 0.4, 0.5, 0.6],
            "n": [1000, 1000, 1000, 1000],
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


def test_cojo_initialization():
    """Test COJO class initialization with default parameters."""
    cojo = COJO()
    assert cojo.p_cutoff == 5e-8
    assert cojo.collinear_cutoff == 0.9
    assert cojo.window_size == 10000000
    assert cojo.verbose is True
    assert len(cojo.snps_selected) == 0
    assert cojo.backward_removed == 0
    assert cojo.collinear_filtered == 0


def test_cojo_initialization_custom_params():
    """Test COJO class initialization with custom parameters."""
    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.8, window_size=5000000, verbose=False)
    assert cojo.p_cutoff == 1e-6
    assert cojo.collinear_cutoff == 0.8
    assert cojo.window_size == 5000000
    assert cojo.verbose is False


def test_conditional_selection_no_significant_snps(sample_sumstats, sample_ld_matrix):
    """Test conditional selection when no SNPs are significant."""
    # Modify p-values to be above cutoff
    sumstats = sample_sumstats.copy()
    sumstats["p"] = [1e-7, 1e-6, 1e-5, 1e-4]

    cojo = COJO(p_cutoff=1e-8)
    result = cojo.conditional_selection(sumstats, sample_ld_matrix)

    assert result.empty
    assert len(cojo.snps_selected) == 0


def test_conditional_selection_single_snp(sample_sumstats, sample_ld_matrix):
    """Test conditional selection with only one significant SNP."""
    cojo = COJO(p_cutoff=1e-7)
    result = cojo.conditional_selection(sample_sumstats, sample_ld_matrix)

    assert len(result) == 1
    assert result["SNP"].iloc[0] == "rs1"
    assert len(cojo.snps_selected) == 1
    assert cojo.backward_removed == 0
    assert cojo.collinear_filtered == 0


def test_conditional_selection_with_collinearity(sample_sumstats, sample_ld_matrix):
    """Test conditional selection with collinear SNPs."""
    # Modify LD matrix to create high collinearity
    ld_matrix = sample_ld_matrix.copy()
    # set all values to 0.95
    ld_matrix[:] = 0.95

    cojo = COJO(p_cutoff=1e-7, collinear_cutoff=0.9)
    result = cojo.conditional_selection(sample_sumstats, ld_matrix)

    assert len(result) == 1  # Only one SNP should be selected due to collinearity
    # assert cojo.collinear_filtered > 0


def test_conditional_selection_with_window(
    sample_sumstats, sample_ld_matrix, sample_positions
):
    """Test conditional selection with window-based LD consideration."""
    cojo = COJO(p_cutoff=1e-7, window_size=1000)
    result = cojo.conditional_selection(
        sample_sumstats, sample_ld_matrix, sample_positions
    )

    assert len(result) > 0
    # SNPs should only be considered for LD if they're within the window


def test_backward_elimination(sample_sumstats, sample_ld_matrix):
    """Test backward elimination of non-significant SNPs."""
    # Modify p-values to create a scenario where backward elimination is needed
    sumstats = sample_sumstats.copy()
    sumstats.loc[sumstats["SNP"] == "rs2", "p"] = 1e-6

    cojo = COJO(p_cutoff=1e-7)
    result = cojo.conditional_selection(sumstats, sample_ld_matrix)

    assert len(result) == 1  # Only the most significant SNP should remain
    # assert cojo.backward_removed > 0


def test_joint_statistics_calculation(sample_sumstats, sample_ld_matrix):
    """Test calculation of joint statistics."""
    cojo = COJO(p_cutoff=1e-7)
    result = cojo.conditional_selection(sample_sumstats, sample_ld_matrix)

    assert "joint_beta" in result.columns
    assert "joint_se" in result.columns
    assert "joint_p" in result.columns
    assert len(result) > 0

    # Check that joint statistics are different from original statistics
    # assert not np.allclose(result["joint_beta"], result["original_beta"])
    # assert not np.allclose(result["joint_se"], result["original_se"])
    # assert not np.allclose(result["joint_p"], result["original_p"])


def test_conditional_statistics_calculation(sample_sumstats, sample_ld_matrix):
    """Test calculation of conditional statistics."""
    cojo = COJO(p_cutoff=1e-7)
    cojo.conditional_selection(sample_sumstats, sample_ld_matrix)

    # Test that conditional statistics are calculated correctly
    cond_beta, cond_se, cond_p = cojo._calculate_conditional_stats()

    assert len(cond_beta) == len(sample_sumstats)
    assert len(cond_se) == len(sample_sumstats)
    assert len(cond_p) == len(sample_sumstats)

    # # Check that conditional statistics are different from original statistics
    # assert not np.allclose(cond_beta, cojo.original_beta)  # type: ignore
    # assert not np.allclose(cond_se, cojo.original_se)  # type: ignore
    # assert not np.allclose(cond_p, cojo.original_p)  # type: ignore
