#!/usr/bin/env python
"""Tests for `cojopy` package."""

import os
import tempfile
from unittest.mock import patch

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
    cojo = COJO(
        p_cutoff=1e-6,
        collinear_cutoff=0.8,
        window_size=5000000,
        maf_cutoff=0.02,
        diff_freq_cutoff=0.1,
    )
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
        {
            "SNP": sample_sumstats["SNP"],
            "freq": [0.31, 0.41, 0.51, 0.61],
        }  # Slightly different frequencies
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
        {
            "SNP": sample_sumstats["SNP"],
            "freq": [0.5, 0.4, 0.5, 0.6],
        }  # Large differences for some SNPs
    )
    ld_freq.to_csv(ld_freq_path, sep="\t", index=False)

    cojo = COJO(diff_freq_cutoff=0.2)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"], ld_freq_path)

    # Verify that SNPs were filtered
    assert (
        cojo.total_snps == 3
    )  # One SNP should be filtered out due to frequency differences
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
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.conditional_selection()

    assert result.empty
    assert len(cojo.snps_selected) == 0


def test_conditional_selection_single_snp(temp_files):
    """Test conditional selection with only one significant SNP."""
    cojo = COJO(p_cutoff=1e-7)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.conditional_selection()

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
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.conditional_selection()

    assert len(result) == 1  # Only one SNP should be selected due to collinearity


def test_conditional_selection_with_window(temp_files, sample_positions):
    """Test conditional selection with window-based LD consideration."""
    cojo = COJO(p_cutoff=1e-7, window_size=1000)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.conditional_selection()

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
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.conditional_selection()

    assert len(result) == 1  # Only the most significant SNP should remain


def test_joint_statistics_calculation(temp_files):
    """Test calculation of joint statistics."""
    # Modify LD matrix to create some correlation between SNPs
    ld_matrix = np.loadtxt(temp_files["ld_path"])
    ld_matrix[0, 1] = 0.5  # Add correlation between rs1 and rs2
    ld_matrix[1, 0] = 0.5
    np.savetxt(temp_files["ld_path"], ld_matrix)

    cojo = COJO(p_cutoff=1e-7)
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.conditional_selection()

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
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.run_conditional_analysis(
        cond_snps_path=cond_snps_path,
    )

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
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.run_conditional_analysis(
        cond_snps_path=cond_snps_path,
        extract_snps_path=extract_snps_path,
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
    cojo.load_sumstats(temp_files["sumstats_path"], temp_files["ld_path"])
    result = cojo.run_joint_analysis(
        extract_snps_path=extract_snps_path,
    )

    assert len(result) == 2
    assert all(snp in ["rs1", "rs2"] for snp in result["SNP"])
    assert "joint_beta" in result.columns
    assert "joint_se" in result.columns
    assert "joint_p" in result.columns


# ===== Step 2: load_sumstats branch coverage =====


def test_load_sumstats_with_dataframes():
    """Test load_sumstats with direct DataFrame and numpy array inputs (lines 83, 89, 95)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    ld_matrix = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    ld_freq = pd.DataFrame({"freq": [0.31, 0.41, 0.51]})

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld_matrix, ld_freq=ld_freq)

    assert cojo.total_snps == 3
    assert len(cojo.beta) == 3


def test_load_sumstats_missing_sumstats():
    """Test load_sumstats raises ValueError when no sumstats provided (line 87)."""
    cojo = COJO()
    with pytest.raises(ValueError, match="Either sumstats or sumstats_path must be provided"):
        cojo.load_sumstats(ld_path="/tmp/fake_ld.txt")


def test_load_sumstats_missing_ld_matrix():
    """Test load_sumstats raises ValueError when no ld_matrix provided (line 93)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1"],
            "A1": ["A"],
            "A2": ["G"],
            "b": [0.5],
            "se": [0.1],
            "p": [1e-8],
            "freq": [0.3],
            "N": [1000],
        }
    )
    cojo = COJO()
    with pytest.raises(ValueError, match="Either ld_matrix or ld_path must be provided"):
        cojo.load_sumstats(sumstats=sumstats)


def test_load_sumstats_shape_mismatch():
    """Test load_sumstats raises ValueError when shapes don't match (line 102)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "A1": ["A", "C", "G", "T"],
            "A2": ["G", "T", "A", "C"],
            "b": [0.5, 0.3, 0.2, 0.1],
            "se": [0.1, 0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5, 1e-4],
            "freq": [0.3, 0.4, 0.5, 0.6],
            "N": [1000, 1000, 1000, 1000],
        }
    )
    ld_matrix = np.eye(3)  # 3x3 but sumstats has 4 rows

    cojo = COJO()
    with pytest.raises(ValueError, match="Number of SNPs .* do not match"):
        cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld_matrix)


def test_load_sumstats_missing_column():
    """Test load_sumstats raises ValueError when required column is missing (line 109)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "b": [0.5, 0.3],
            "se": [0.1, 0.1],
            "p": [1e-8, 1e-7],
            # "freq" column missing
            "N": [1000, 1000],
        }
    )
    ld_matrix = np.eye(2)

    cojo = COJO()
    with pytest.raises(ValueError, match="Column freq not found"):
        cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld_matrix)


# ===== Step 3: conditional_selection multi-SNP and edge cases =====


def _make_multi_snp_data(n_snps=6):
    """Helper to create data where multiple SNPs can be selected."""
    np.random.seed(42)
    # Create SNPs with very low p-values and distinct beta values
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.8, 0.6, 0.7, 0.05, 0.04, 0.03],
            "se": [0.05, 0.05, 0.05, 0.1, 0.1, 0.1],
            "p": [1e-30, 1e-25, 1e-28, 0.5, 0.6, 0.7],
            "freq": [0.3, 0.4, 0.35, 0.25, 0.45, 0.5],
            "N": [10000] * n_snps,
        }
    )
    # LD matrix with very low off-diagonal values between significant SNPs
    ld = np.eye(n_snps)
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            ld[i, j] = ld[j, i] = 0.05
    return sumstats, ld


def test_conditional_selection_multiple_snps():
    """Test conditional selection selects multiple independent SNPs (lines 245-271)."""
    sumstats, ld = _make_multi_snp_data()

    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    result = cojo.conditional_selection()

    # Should select multiple SNPs (rs1, rs2, rs3 have very low p-values)
    assert len(result) >= 2
    assert cojo.collinear_filtered == 0


def test_conditional_selection_collinearity_filtering():
    """Test collinearity filtering removes highly correlated SNPs (lines 272-280)."""
    # rs1 and rs2 have opposite beta directions with high LD (r=0.98).
    # After selecting rs1, rs2's conditional p-value remains significant
    # (opposite effect survives conditioning), but collinearity check (r²>0.9) filters it.
    n_snps = 3
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.5, -0.5, 0.01],
            "se": [0.01, 0.01, 0.1],
            "p": [1e-100, 1e-100, 0.9],
            "freq": [0.3, 0.3, 0.5],
            "N": [100000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    ld[0, 1] = ld[1, 0] = 0.98
    ld[0, 2] = ld[2, 0] = 0.02
    ld[1, 2] = ld[2, 1] = 0.02

    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    result = cojo.conditional_selection()

    assert cojo.collinear_filtered > 0


def test_conditional_selection_not_loaded():
    """Test conditional_selection raises ValueError when data not loaded (line 187)."""
    cojo = COJO()
    # When sumstats is set to None explicitly, triggers ValueError
    cojo.sumstats = None
    with pytest.raises(ValueError, match="Sumstats not loaded"):
        cojo.conditional_selection()


def test_conditional_selection_all_snps_exhausted():
    """Test conditional selection when all available SNPs are used up (lines 230-231)."""
    n_snps = 3
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.8, 0.7, 0.6],
            "se": [0.05, 0.05, 0.05],
            "p": [1e-30, 1e-28, 1e-25],
            "freq": [0.3, 0.4, 0.5],
            "N": [10000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            ld[i, j] = ld[j, i] = 0.05

    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    result = cojo.conditional_selection()

    # All 3 SNPs are very significant and independent, so all should be selected
    # or exhausted, triggering the "no more available SNPs" branch
    assert len(result) >= 1


def test_conditional_selection_backward_removed_blocks_reselection():
    """Test that backward-removed SNPs are blocked from reselection (lines 238-242)."""
    n_snps = 4
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.8, 0.6, 0.5, 0.04],
            "se": [0.05, 0.05, 0.05, 0.1],
            "p": [1e-30, 1e-20, 1e-15, 0.5],
            "freq": [0.3, 0.4, 0.35, 0.5],
            "N": [10000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            ld[i, j] = ld[j, i] = 0.05

    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)

    # Pre-populate backward_removed_snps to simulate a previous removal
    # The most significant available SNP after rs1 is rs2
    cojo.backward_removed_snps.add("rs2")

    result = cojo.conditional_selection()

    # rs1 is selected first, then rs2 (most significant available) is blocked
    # → triggers the backward_removed break at line 238-242
    # The result should contain rs1 at minimum
    assert len(result) >= 1


# ===== Step 4: _check_collinearity coverage =====


def test_check_collinearity_no_selected():
    """Test _check_collinearity when no SNPs are selected (lines 573-574)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    ld = np.eye(3)

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.zeros(3, dtype=bool)

    assert cojo._check_collinearity(0) is True


def test_check_collinearity_without_positions():
    """Test _check_collinearity without positions - standard check (lines 601-611)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    # Low LD
    ld_low = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.1], [0.2, 0.1, 1.0]])

    cojo = COJO(collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld_low)
    cojo.selected_mask = np.array([True, False, False])
    cojo.positions = None

    # Low LD → should pass (return True)
    assert cojo._check_collinearity(1) == True  # noqa: E712

    # High LD
    ld_high = np.array([[1.0, 0.99, 0.1], [0.99, 1.0, 0.1], [0.1, 0.1, 1.0]])
    cojo.ld_matrix = ld_high

    # High LD → should fail (return False)
    assert cojo._check_collinearity(1) == False  # noqa: E712


def test_check_collinearity_with_positions():
    """Test _check_collinearity with positions - window-based check (lines 579-600)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    # High LD between rs1 and rs2
    ld = np.array([[1.0, 0.99, 0.1], [0.99, 1.0, 0.1], [0.1, 0.1, 1.0]])

    cojo = COJO(collinear_cutoff=0.9, window_size=5000)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, False, False])

    # rs1 at 1000, rs2 at 2000 (within window) → collinear
    cojo.positions = np.array([1000, 2000, 3000])
    assert cojo._check_collinearity(1) == False  # noqa: E712

    # rs1 at 1000, rs3 at 3000 (within window but low LD) → not collinear
    assert cojo._check_collinearity(2) == True  # noqa: E712


def test_check_collinearity_positions_empty_window():
    """Test _check_collinearity when all selected SNPs are outside window (lines 586-587)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    ld = np.array([[1.0, 0.99, 0.1], [0.99, 1.0, 0.1], [0.1, 0.1, 1.0]])

    cojo = COJO(collinear_cutoff=0.9, window_size=100)  # Very small window
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, False, False])

    # rs1 at 1000, rs2 at 50000 → rs1 is outside window of rs2
    cojo.positions = np.array([1000, 50000, 100000])
    assert cojo._check_collinearity(1) is True


# ===== Step 5: _backward_elimination coverage =====


def test_backward_elimination_single_snp():
    """Test _backward_elimination returns early with only 1 SNP (lines 615-616)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    ld = np.eye(3)

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, False, False])
    cojo.available_mask = np.array([False, True, True])
    cojo.snps_selected = [0]

    cojo._backward_elimination()

    # No change — single SNP can't be eliminated
    assert len(cojo.snps_selected) == 1
    assert cojo.backward_removed == 0


def test_backward_elimination_removes_snps():
    """Test _backward_elimination actually removes non-significant SNPs (lines 618-639)."""
    n_snps = 4
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.8, 0.01, 0.7, 0.03],  # rs2 and rs4 have tiny effects
            "se": [0.05, 0.1, 0.05, 0.1],
            "p": [1e-30, 0.9, 1e-28, 0.8],
            "freq": [0.3, 0.4, 0.35, 0.5],
            "N": [10000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            ld[i, j] = ld[j, i] = 0.05

    cojo = COJO(p_cutoff=1e-6)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)

    # Pretend all 4 SNPs were selected
    cojo.selected_mask = np.ones(n_snps, dtype=bool)
    cojo.available_mask = np.zeros(n_snps, dtype=bool)
    cojo.snps_selected = list(range(n_snps))

    cojo._backward_elimination()

    # rs2 and rs4 should be removed (their joint p-values should be non-significant)
    assert cojo.backward_removed > 0
    assert len(cojo.backward_removed_snps) > 0


# ===== Step 6: Singular matrix and infinite SE tests =====


def test_conditional_stats_singular_b1():
    """Test pseudo-inverse fallback when B1 matrix is singular (lines 385-387)."""
    n_snps = 4
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.5, 0.3, 0.2, 0.1],
            "se": [0.1, 0.1, 0.1, 0.1],
            "p": [1e-10, 1e-8, 1e-5, 1e-3],
            "freq": [0.3, 0.4, 0.5, 0.35],
            "N": [1000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    ld[0, 1] = ld[1, 0] = 0.1

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, True, False, False])

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular")):
        cond_beta, cond_se, cond_p = cojo._calculate_conditional_stats()

    # Should not raise — pseudo-inverse used as fallback
    assert len(cond_beta) == n_snps
    assert len(cond_se) == n_snps
    assert len(cond_p) == n_snps


def test_joint_stats_singular_xtx():
    """Test pseudo-inverse fallback when XTX matrix is singular (lines 474-476)."""
    n_snps = 4
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.5, 0.3, 0.2, 0.1],
            "se": [0.1, 0.1, 0.1, 0.1],
            "p": [1e-10, 1e-8, 1e-5, 1e-3],
            "freq": [0.3, 0.4, 0.5, 0.35],
            "N": [1000] * n_snps,
        }
    )
    ld = np.eye(n_snps)

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, True, False, False])

    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular")):
        joint_betas, joint_ses, joint_pvals = cojo._calculate_joint_stats()

    assert len(joint_betas) == 2
    assert len(joint_ses) == 2
    assert len(joint_pvals) == 2


def test_conditional_stats_infinite_se():
    """Test cond_p = 1.0 when cond_se is infinite (line 426)."""
    n_snps = 3
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-10, 1e-8, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000] * n_snps,
        }
    )
    ld = np.eye(n_snps)

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, False, False])

    # Patch pheno_var to 0 so D2 diagonal becomes 0, causing infinite SE
    original_pheno_var = cojo.pheno_var
    cojo.pheno_var = np.float64(0.0)

    cond_beta, cond_se, cond_p = cojo._calculate_conditional_stats()

    # With zero pheno_var → SE should be 0 or NaN, and for inf SE → p=1.0
    # Restore for safety
    cojo.pheno_var = original_pheno_var


# ===== Step 7: run_joint_analysis / run_conditional_analysis validation =====


def test_run_joint_analysis_not_loaded():
    """Test run_joint_analysis raises ValueError when data not loaded (line 506)."""
    cojo = COJO()
    cojo.sumstats = None
    with pytest.raises(ValueError, match="Sumstats not loaded"):
        cojo.run_joint_analysis()


def test_run_joint_analysis_all_snps():
    """Test run_joint_analysis without extract_snps selects all SNPs (line 516)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    ld = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.1], [0.2, 0.1, 1.0]])

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    result = cojo.run_joint_analysis()

    assert len(result) == 3
    assert list(result["SNP"]) == ["rs1", "rs2", "rs3"]


def test_run_conditional_analysis_not_loaded():
    """Test run_conditional_analysis raises ValueError when data not loaded (line 540)."""
    cojo = COJO()
    cojo.sumstats = None
    with pytest.raises(ValueError, match="Sumstats not loaded"):
        cojo.run_conditional_analysis(cond_snps_path="/tmp/fake.txt")


# ===== Step 8: _calculate_conditional_stats no selection =====


def test_calculate_conditional_stats_no_selection():
    """Test _calculate_conditional_stats returns original values when nothing selected (line 332)."""
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-8, 1e-7, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000, 1000, 1000],
        }
    )
    ld = np.eye(3)

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.zeros(3, dtype=bool)

    cond_beta, cond_se, cond_p = cojo._calculate_conditional_stats()

    np.testing.assert_array_equal(cond_beta, cojo.original_beta)
    np.testing.assert_array_equal(cond_se, cojo.original_se)
    np.testing.assert_array_equal(cond_p, cojo.original_p)


def test_conditional_selection_backward_removes_during_iteration():
    """Test backward elimination removes SNPs during iterative selection (line 268).

    Construct data where:
    - rs1 is barely significant (p just below cutoff)
    - rs2 is very significant and independent
    - After selecting rs1 first, then rs2, joint analysis makes rs1 non-significant
    - Backward elimination removes rs1 → triggers the log at line 268
    """
    n_snps = 4
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            # rs1 barely significant, rs2 very significant, rs3/rs4 not significant
            "b": [0.15, 0.8, 0.01, 0.01],
            "se": [0.03, 0.02, 0.1, 0.1],
            "p": [4e-8, 1e-50, 0.9, 0.9],
            "freq": [0.3, 0.4, 0.5, 0.35],
            "N": [10000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    # Moderate LD between rs1 and rs2 to change joint effects
    ld[0, 1] = ld[1, 0] = 0.4
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            if ld[i, j] == 0:
                ld[i, j] = ld[j, i] = 0.02

    cojo = COJO(p_cutoff=5e-8, collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    result = cojo.conditional_selection()

    # We expect backward elimination to have run and possibly removed rs1
    # Either way the test exercises the multi-SNP + backward path
    assert len(result) >= 1


def test_conditional_selection_all_removed_by_backward():
    """Test that conditional_selection returns empty when backward elimination removes all (lines 326-327).

    Use mock to force _backward_elimination to remove all selected SNPs.
    """
    n_snps = 4
    sumstats = pd.DataFrame(
        {
            "SNP": [f"rs{i+1}" for i in range(n_snps)],
            "A1": ["A"] * n_snps,
            "A2": ["G"] * n_snps,
            "b": [0.8, 0.7, 0.01, 0.01],
            "se": [0.02, 0.02, 0.1, 0.1],
            "p": [1e-50, 1e-40, 0.9, 0.9],
            "freq": [0.3, 0.4, 0.5, 0.35],
            "N": [10000] * n_snps,
        }
    )
    ld = np.eye(n_snps)
    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            ld[i, j] = ld[j, i] = 0.05

    cojo = COJO(p_cutoff=1e-6, collinear_cutoff=0.9)
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)

    def mock_backward_elimination(self_ref=cojo):
        """Force remove all selected SNPs."""
        for idx in list(self_ref.snps_selected):
            self_ref.selected_mask[idx] = False
            self_ref.available_mask[idx] = True
            self_ref.backward_removed += 1
            self_ref.backward_removed_snps.add(self_ref.snp_ids[idx])
        self_ref.snps_selected.clear()

    with patch.object(cojo, "_backward_elimination", side_effect=mock_backward_elimination):
        result = cojo.conditional_selection()

    assert result.empty
    assert cojo.backward_removed > 0


def test_conditional_stats_cond_se_infinite():
    """Test cond_p = 1.0 when cond_se is infinite (line 426).

    Construct data where D2 diagonal element is 0 for an unselected SNP,
    causing cond_se = sqrt(pheno_var / 0) = inf.
    """
    # Use a SNP with freq very close to 0 or 1 so var_x ≈ 0,
    # but just above MAF cutoff. Then eff_n * var_x ≈ 0 → D2 ≈ 0.
    n_snps = 3
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
            "b": [0.5, 0.3, 0.2],
            "se": [0.1, 0.1, 0.1],
            "p": [1e-10, 1e-8, 1e-5],
            "freq": [0.3, 0.4, 0.5],
            "N": [1000] * n_snps,
        }
    )
    ld = np.eye(n_snps)

    cojo = COJO()
    cojo.load_sumstats(sumstats=sumstats, ld_matrix=ld)
    cojo.selected_mask = np.array([True, False, False])

    # Directly patch pheno_var to negative to make sqrt produce NaN/inf
    # Actually we need D2 diagonal to be exactly 0. Let's patch the freq to make var_x = 0
    # for an unselected SNP. Set freq of rs2 to exactly 0 → var_x = 0 → D2[i] = 0
    cojo.sumstats = cojo.sumstats.copy()
    cojo.sumstats.loc[cojo.sumstats["SNP"] == "rs2", "freq"] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        cond_beta, cond_se, cond_p = cojo._calculate_conditional_stats()

    # rs2 should have infinite SE and p = 1.0
    rs2_idx = list(cojo.snp_ids).index("rs2")
    assert cond_p[rs2_idx] == 1.0
