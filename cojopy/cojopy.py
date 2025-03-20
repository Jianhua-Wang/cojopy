"""Main module."""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


class COJO:
    """Class for Conditional & Joint Association Analysis using summary statistics and LD matrix as input."""

    def __init__(
        self,
        p_cutoff: float = 5e-8,
        collinear_cutoff: float = 0.9,
        window_size: int = 10000000,
        maf_cutoff: float = 0.01,
        diff_freq_cutoff: float = 0.05,
    ):
        """Initialize the COJO analysis parameters.

        Parameters
        ----------
        p_cutoff : float
            Path to the LD frequency file
        p_cutoff : float
            P-value threshold for selecting SNPs
        collinear_cutoff : float
            Threshold for collinearity between SNPs (r²)
        window_size : int
            Window size in base pairs to consider LD between SNPs
        maf_cutoff : float
            Minor allele frequency threshold for selecting SNPs
        diff_freq_cutoff : float
            Difference in minor allele frequency threshold for selecting SNPs
        """
        self.p_cutoff = p_cutoff
        self.collinear_cutoff = collinear_cutoff
        self.window_size = window_size
        self.snps_selected: list[int] = []
        self.backward_removed = 0
        self.collinear_filtered = 0
        self.maf_cutoff = maf_cutoff
        self.diff_freq_cutoff = diff_freq_cutoff

        # Set up logging
        self.logger = logging.getLogger("COJO")

    def load_sumstats(self, sumstats_path: str, ld_path: str, ld_freq_path: Optional[str] = None):
        """
        Load summary statistics and LD matrix.

        Parameters
        ----------
        sumstats_path : str
            Path to the summary statistics file
            The file should have the same columns as COJO input: SNP, A1, A2, b, se, p, freq, N
        ld_path : str
            Path to the LD matrix file
        ld_freq_path : str, optional
            Path to the LD frequency file. The file should have the following columns: SNP, freq
            Use freq in sumstats if ld_freq_path is not provided.
        """
        self.logger.info("Loading summary statistics and LD matrix")
        self.sumstats = pd.read_csv(sumstats_path, sep="\t")
        self.ld_matrix = np.loadtxt(ld_path)

        if self.sumstats.shape[0] != self.ld_matrix.shape[0]:
            raise ValueError("Number of SNPs in summary statistics and LD matrix do not match")

        required_columns = ["SNP", "A1", "A2", "b", "se", "p", "freq", "N"]
        for col in required_columns:
            if col not in self.sumstats.columns:
                raise ValueError(f"Column {col} not found in summary statistics file")

        # Filter SNPs based on MAF cutoff, delete those SNPs both in sumstats and ld_matrix
        maf_mask = (self.sumstats["freq"] >= self.maf_cutoff) & (self.sumstats["freq"] <= 1 - self.maf_cutoff)
        self.logger.info("Filtering SNPs based on MAF cutoff: %d SNPs removed", np.sum(~maf_mask))
        self.sumstats = self.sumstats[maf_mask]
        self.ld_matrix = self.ld_matrix[maf_mask, :][:, maf_mask]

        if ld_freq_path is not None:
            self.ld_freq = pd.read_csv(ld_freq_path, sep="\t")["freq"].values
            freq_diff = np.abs(self.sumstats["freq"].values - self.ld_freq)  # type: ignore
            freq_diff_mask = freq_diff < self.diff_freq_cutoff
            self.logger.info("Filtering SNPs based on difference in frequency: %d SNPs removed", np.sum(~freq_diff_mask))
            self.sumstats = self.sumstats[freq_diff_mask]
            self.ld_matrix = self.ld_matrix[freq_diff_mask, :][:, freq_diff_mask]
        else:
            self.ld_freq = self.sumstats["freq"].values
            self.logger.info("Using freq in sumstats as LD frequency")

        self.logger.info("After filtering, %d SNPs left", self.sumstats.shape[0])
        self.total_snps = self.sumstats.shape[0]
        self.freq = self.ld_freq
        self.n = self.sumstats["N"].values
        self.beta = self.sumstats["b"].values
        self.se = self.sumstats["se"].values
        self.p = self.sumstats["p"].values
        self.snp_ids = self.sumstats["SNP"].values
        self.original_beta = self.beta.copy()
        self.original_se = self.se.copy()
        self.original_p = self.p.copy()
        self.pheno_var = self._cal_pheno_var(
            self.freq,  # type: ignore
            self.beta,  # type: ignore
            self.se,  # type: ignore
            self.n,  # type: ignore
        )
        self.positions = None  # TODO: add positions
        self.logger.info("Phenotype variance: %g", self.pheno_var)

    def conditional_selection(
        self,
        sumstats_path: str,
        ld_path: str,
        ld_freq_path: Optional[str] = None,
        positions: Union[pd.Series, None] = None,
    ) -> pd.DataFrame:
        """
        Perform stepwise model selection to identify independent associated signals.

        Parameters
        ----------
        sumstats : pandas.DataFrame
            Summary statistics containing columns: SNP, beta, se, p, freq, n
        ld_matrix : numpy.ndarray
            LD correlation matrix for the SNPs
        positions : pandas.Series, optional
            Positions of SNPs (for window-based LD consideration)

        Returns
        -------
        result : pandas.DataFrame
            Selected SNPs with their conditional and joint effects
        """
        self.logger.info(
            "Starting COJO analysis with p-value cutoff: %g, collinearity cutoff: %g",
            self.p_cutoff,
            self.collinear_cutoff,
        )

        self.load_sumstats(sumstats_path, ld_path, ld_freq_path)

        # Create masks for different operations
        self.selected_mask = np.zeros(self.total_snps, dtype=bool)  # Mask for selected SNPs
        self.available_mask = np.ones(self.total_snps, dtype=bool)  # Mask for available SNPs

        self.logger.info("Analyzing %d SNPs", self.total_snps)

        # Start with the most significant SNP
        min_p_idx = np.argmin(self.p)  # type: ignore
        if self.p[min_p_idx] > self.p_cutoff:
            self.logger.info("No significant SNPs found (minimum p-value: %g)", self.p[min_p_idx])
            return pd.DataFrame()  # No significant SNPs

        # Select the first SNP
        self.selected_mask[min_p_idx] = True
        self.available_mask[min_p_idx] = False
        self.snps_selected.append(min_p_idx)  # type: ignore
        self.logger.info(
            "Iteration 1: Selected SNP %s with p-value %g",
            self.snp_ids[min_p_idx],
            self.p[min_p_idx],
        )

        # Iterative model selection
        continue_selection = True
        iteration = 2
        while continue_selection:
            self.logger.info("Iteration %d: Calculating conditional statistics", iteration)
            # Calculate conditional p-values for remaining SNPs
            cond_betas, cond_ses, cond_pvals = self._calculate_conditional_stats()

            # Find the most significant SNP that passes the threshold
            available_indices = np.where(self.available_mask)[0]
            if len(available_indices) == 0:
                self.logger.info("No more available SNPs to test")
                break

            cond_p_subset = cond_pvals[available_indices]
            min_cond_p_idx = available_indices[np.argmin(cond_p_subset)]

            if cond_pvals[min_cond_p_idx] < self.p_cutoff:
                self.logger.info(
                    "Found significant SNP %s with conditional p-value %g",
                    self.snp_ids[min_cond_p_idx],
                    cond_pvals[min_cond_p_idx],
                )

                # Check collinearity
                if self._check_collinearity(min_cond_p_idx):
                    # Add the SNP to the model
                    self.selected_mask[min_cond_p_idx] = True
                    self.available_mask[min_cond_p_idx] = False
                    self.snps_selected.append(min_cond_p_idx)
                    self.logger.info(
                        "Added SNP %s to the model (total selected: %d): selected SNPs: %s",
                        self.snp_ids[min_cond_p_idx],
                        len(self.snps_selected),
                        ", ".join(self.snp_ids[self.selected_mask]),
                    )

                    # Backward elimination step
                    prev_selected = len(self.snps_selected)
                    self._backward_elimination()
                    if prev_selected > len(self.snps_selected):
                        self.logger.info(
                            "Backward elimination removed %d SNPs",
                            prev_selected - len(self.snps_selected),
                        )
                else:
                    # Skip this SNP due to collinearity
                    self.available_mask[min_cond_p_idx] = False
                    self.collinear_filtered += 1
                    self.logger.info(
                        "SNP %s filtered due to collinearity (r² > %g)",
                        self.snp_ids[min_cond_p_idx],
                        self.collinear_cutoff,
                    )
            else:
                # No more significant SNPs
                self.logger.info(
                    "No more significant SNPs (minimum conditional p-value: %g)",
                    cond_pvals[min_cond_p_idx],
                )
                continue_selection = False

            iteration += 1

        # Final joint analysis
        if len(self.snps_selected) > 0:
            self.logger.info(
                "Performing final joint analysis with %d selected SNPs",
                len(self.snps_selected),
            )
            joint_betas, joint_ses, joint_pvals = self._calculate_joint_stats()

            # Prepare results
            result = pd.DataFrame(
                {
                    "SNP": [self.snp_ids[i] for i in self.snps_selected],
                    "original_beta": [self.original_beta[i] for i in self.snps_selected],
                    "original_se": [self.original_se[i] for i in self.snps_selected],
                    "original_p": [self.original_p[i] for i in self.snps_selected],
                    "joint_beta": joint_betas,
                    "joint_se": joint_ses,
                    "joint_p": joint_pvals,
                }
            )

            self.logger.info(
                "COJO analysis complete. Selected %d independent SNPs.",
                len(self.snps_selected),
            )
            self.logger.info("Filtered %d SNPs due to collinearity.", self.collinear_filtered)
            self.logger.info("Removed %d SNPs during backward elimination.", self.backward_removed)

            return result
        else:
            self.logger.info("No SNPs selected in the final model.")
            return pd.DataFrame()

    def _calculate_conditional_stats(self):
        """Calculate conditional statistics for all SNPs given the currently selected SNPs using the improved implementation from Yang et al. (2012)."""
        if sum(self.selected_mask) == 0:
            return self.original_beta, self.original_se, self.original_p

        # Get indices of selected and unselected SNPs
        selected_indices = np.where(self.selected_mask)[0]
        unselected_indices = np.where(~self.selected_mask)[0]

        # Extract data for selected and unselected SNPs
        beta_selected = self.original_beta[selected_indices]
        beta_unselected = self.original_beta[unselected_indices]
        se_selected = self.original_se[selected_indices]
        se_unselected = self.original_se[unselected_indices]
        n_selected = self.n[selected_indices]
        n_unselected = self.n[unselected_indices]
        freq_selected = self.sumstats["freq"].values[selected_indices]
        freq_unselected = self.sumstats["freq"].values[unselected_indices]

        # Calculate phenotype variance using all SNPs
        pheno_var = self.pheno_var

        # Calculate effective sample sizes
        eff_n_selected = self._cal_effective_n(pheno_var, beta_selected, se_selected, freq_selected)
        eff_n_unselected = self._cal_effective_n(pheno_var, beta_unselected, se_unselected, freq_unselected)

        # Compute genotype variances
        var_x_selected = 2 * freq_selected * (1 - freq_selected)
        var_x_unselected = 2 * freq_unselected * (1 - freq_unselected)

        # Construct D matrices
        D1 = np.diag(eff_n_selected * var_x_selected)
        D2 = np.diag(eff_n_unselected * var_x_unselected)

        # Extract submatrices from LD matrix
        R11 = self.ld_matrix[np.ix_(selected_indices, selected_indices)]  # B1
        R21 = self.ld_matrix[np.ix_(unselected_indices, selected_indices)]  # C

        # Construct B1 matrix
        B1 = np.zeros((len(selected_indices), len(selected_indices)))
        for j in range(len(selected_indices)):
            for k in range(len(selected_indices)):
                if j == k:
                    B1[j, k] = eff_n_selected[j] * var_x_selected[j]
                else:
                    B1[j, k] = (
                        min(eff_n_selected[j], eff_n_selected[k])
                        * R11[j, k]
                        * np.sqrt(var_x_selected[j] * var_x_selected[k])
                    )

        # Calculate inverse of B1
        try:
            B1_inv = np.linalg.inv(B1)
        except np.linalg.LinAlgError:
            self.logger.warning("B1 matrix is singular, using pseudo-inverse")
            B1_inv = np.linalg.pinv(B1)

        # Initialize arrays for conditional statistics
        cond_beta = np.zeros(self.total_snps)
        cond_se = np.zeros(self.total_snps)
        cond_p = np.ones(self.total_snps)

        # Calculate conditional statistics for each unselected SNP
        for i, idx in enumerate(unselected_indices):
            # Construct C vector for this SNP
            C = np.zeros(len(selected_indices))
            for k in range(len(selected_indices)):
                C[k] = (
                    min(eff_n_unselected[i], eff_n_selected[k])
                    * R21[i, k]
                    * np.sqrt(var_x_unselected[i] * var_x_selected[k])
                )

            # Calculate conditional beta
            Z_Bi = C @ B1_inv
            adjustment = (Z_Bi * D1.diagonal()) @ beta_selected / D2.diagonal()[i]
            cond_beta[idx] = beta_unselected[i] - adjustment

            # Calculate conditional standard error
            cond_se[idx] = np.sqrt(pheno_var / D2.diagonal()[i])

            # Calculate p-value
            if cond_se[idx] < np.inf:
                cond_p[idx] = self._calculate_p_value(cond_beta[idx], cond_se[idx])
            else:
                cond_p[idx] = 1.0

            self.logger.debug(
                f"SNP {self.snp_ids[idx]} has p-value {cond_p[idx]:.2e}, beta {cond_beta[idx]:.5f}, se {cond_se[idx]:.5f}"
            )

        return cond_beta, cond_se, cond_p

    def _calculate_joint_stats(self):
        """Calculate joint statistics for all selected SNPs using the improved implementation from Yang et al. (2012)."""
        selected_indices = np.where(self.selected_mask)[0]
        self.logger.info("Calculating joint statistics for %d SNPs", len(selected_indices))
        n_snp = len(selected_indices)

        # Extract relevant data for selected SNPs
        beta_selected = self.original_beta[selected_indices]
        se_selected = self.original_se[selected_indices]
        freq_selected = self.sumstats["freq"].values[selected_indices]
        ld_selected = self.ld_matrix[np.ix_(selected_indices, selected_indices)]

        # Calculate effective sample sizes
        eff_n = self._cal_effective_n(self.pheno_var, beta_selected, se_selected, freq_selected)

        # Construct D matrix (diagonal)
        var_x = 2 * freq_selected * (1 - freq_selected)
        D = np.diag(eff_n * var_x)

        # Construct X'X matrix
        XTX = np.zeros((n_snp, n_snp))
        for j in range(n_snp):
            for k in range(n_snp):
                if j == k:
                    XTX[j, k] = eff_n[j] * var_x[j]
                else:
                    XTX[j, k] = min(eff_n[j], eff_n[k]) * ld_selected[j, k] * np.sqrt(var_x[j] * var_x[k])

        # Compute joint effects
        try:
            XTX_inv = np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
            self.logger.warning("XTX matrix is singular, using pseudo-inverse")
            XTX_inv = np.linalg.pinv(XTX)

        # Calculate joint effects using Equation (12)
        joint_betas = XTX_inv @ D @ beta_selected

        # Calculate standard errors using Equation (12)
        var_joint = self.pheno_var * XTX_inv.diagonal()
        joint_ses = np.sqrt(var_joint)

        # Calculate p-values
        joint_pvals = self._calculate_p_value(joint_betas, joint_ses)

        return joint_betas, joint_ses, joint_pvals

    def run_joint_analysis(self, sumstats_path: str, ld_path: str, extract_snps_path: Optional[str] = None, ld_freq_path: Optional[str] = None):
        """Run joint analysis for all selected SNPs."""
        self.load_sumstats(sumstats_path, ld_path, ld_freq_path)
        if extract_snps_path is not None:
            extract_snps = []
            with open(extract_snps_path, "r") as f:
                for line in f:
                    extract_snps.append(line.strip())
            self.logger.info("Extracting SNPs from %s", extract_snps_path)
            self.selected_mask = np.isin(self.snp_ids, extract_snps)  # type: ignore
        else:
            self.selected_mask = np.ones(self.total_snps, dtype=bool)  # type: ignore
        joint_betas, joint_ses, joint_pvals = self._calculate_joint_stats()
        return pd.DataFrame(
            {
                "SNP": self.snp_ids[self.selected_mask],
                "original_beta": self.original_beta[self.selected_mask],
                "original_se": self.original_se[self.selected_mask],
                "original_p": self.original_p[self.selected_mask],
                "joint_beta": joint_betas,
                "joint_se": joint_ses,
                "joint_p": joint_pvals,
            }
        )

    def _check_collinearity(self, new_snp_idx):
        """Check for collinearity between the new SNP and already selected SNPs.

        Returns False if collinearity is detected, True otherwise.
        """
        if sum(self.selected_mask) == 0:
            return True

        selected_indices = np.where(self.selected_mask)[0]

        # Consider window size if positions are provided
        if self.positions is not None:
            in_window = (
                np.abs(self.positions[new_snp_idx] - self.positions[selected_indices])  # type: ignore
                <= self.window_size
            )
            selected_indices_in_window = selected_indices[in_window]

            if len(selected_indices_in_window) == 0:
                return True

            # Check only SNPs within the window
            ld_subset = self.ld_matrix[new_snp_idx, selected_indices_in_window]
            max_r2 = np.max(ld_subset**2)

            if len(selected_indices_in_window) > 0:
                max_r2_idx = selected_indices_in_window[np.argmax(ld_subset**2)]
                self.logger.debug(
                    "Max r² = %g with SNP %s (within %d bp window)",
                    max_r2,
                    self.snp_ids[max_r2_idx],
                    self.window_size,
                )
        else:
            # Check all selected SNPs
            ld_subset = self.ld_matrix[new_snp_idx, selected_indices]
            max_r2 = np.max(ld_subset**2)

            max_r2_idx = selected_indices[np.argmax(ld_subset**2)]
            self.logger.debug("Max r² = %g with SNP %s", max_r2, self.snp_ids[max_r2_idx])

        return max_r2 < self.collinear_cutoff

    def _backward_elimination(self):
        """Perform backward elimination to remove SNPs that are no longer significant after adding a new SNP."""
        if len(self.snps_selected) <= 1:
            return

        # Calculate joint statistics
        joint_betas, joint_ses, joint_pvals = self._calculate_joint_stats()

        # Find SNPs that are no longer significant
        to_remove = []
        for i, idx in enumerate(self.snps_selected):
            if joint_pvals[i] > self.p_cutoff:
                to_remove.append((idx, i))  # Store both original index and position in selected list
                self.logger.info(
                    "SNP %s no longer significant in joint model (p-value: %g)",
                    self.snp_ids[idx],
                    joint_pvals[i],
                )

        # Remove SNPs in reverse order (to avoid index issues)
        for idx, pos in sorted(to_remove, key=lambda x: x[1], reverse=True):
            self.selected_mask[idx] = False
            self.available_mask[idx] = True  # Make it available again
            self.snps_selected.pop(pos)
            self.backward_removed += 1

    def _calculate_p_value(self, beta: np.ndarray, se: np.ndarray) -> np.ndarray:
        z_scores = beta / se
        log_sf = norm.logsf(abs(z_scores))
        log_p = np.log(2) + log_sf
        return np.exp(log_p)

    def _cal_pheno_var(self, freq: np.ndarray, beta: np.ndarray, se: np.ndarray, n: np.ndarray) -> np.float64:
        """Calculate the phenotype variance using Equation (8) from Yang et al. (2012)."""
        var_x = 2 * freq * (1 - freq)
        Vp_buf = var_x * n * se**2 + var_x * beta**2 * n / (n - 1)
        return np.median(Vp_buf)

    def _cal_effective_n(self, pheno_var: np.float64, beta: np.ndarray, se: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """Calculate the effective sample size using Equation (13) from Yang et al. (2012)."""
        var_x = 2 * freq * (1 - freq)
        eff_n = (pheno_var - var_x * beta**2) / (var_x * se**2) + 1
        return eff_n
