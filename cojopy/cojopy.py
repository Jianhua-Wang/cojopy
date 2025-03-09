"""Main module."""

import logging
from typing import Union

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
        verbose: bool = True,
    ):
        """Initialize the COJO analysis parameters.

        Parameters
        ----------
        p_cutoff : float
            P-value threshold for selecting SNPs
        collinear_cutoff : float
            Threshold for collinearity between SNPs (r²)
        window_size : int
            Window size in base pairs to consider LD between SNPs
        verbose : bool
            Whether to print logging information
        """
        self.p_cutoff = p_cutoff
        self.collinear_cutoff = collinear_cutoff
        self.window_size = window_size
        self.snps_selected: list[int] = []
        self.backward_removed = 0
        self.collinear_filtered = 0
        self.verbose = verbose

        # Set up logging
        self.logger = logging.getLogger("COJO")

    def conditional_selection(
        self,
        sumstats: pd.DataFrame,
        ld_matrix: np.ndarray,
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

        # Extract necessary columns from summary statistics
        beta = sumstats["b"].values
        se = sumstats["se"].values
        p_values = sumstats["p"].values
        n = (
            sumstats["n"].values
            if "n" in sumstats.columns
            else np.ones(len(beta)) * sumstats.get("n_samples", 100000)
        )
        snp_ids = sumstats["SNP"].values

        # Store original data
        self.original_beta = beta.copy()
        self.original_se = se.copy()
        self.original_p = p_values.copy()
        self.ld_matrix = ld_matrix
        self.n = n
        self.sumstats = sumstats
        self.snp_ids = snp_ids
        self.positions = positions

        # Create masks for different operations
        self.total_snps = len(beta)
        self.selected_mask = np.zeros(
            self.total_snps, dtype=bool
        )  # Mask for selected SNPs
        self.available_mask = np.ones(
            self.total_snps, dtype=bool
        )  # Mask for available SNPs

        self.logger.info("Analyzing %d SNPs", self.total_snps)

        # Start with the most significant SNP
        min_p_idx = np.argmin(p_values)  # type: ignore
        if p_values[min_p_idx] > self.p_cutoff:
            self.logger.info(
                "No significant SNPs found (minimum p-value: %g)", p_values[min_p_idx]
            )
            return pd.DataFrame()  # No significant SNPs

        # Select the first SNP
        self.selected_mask[min_p_idx] = True
        self.available_mask[min_p_idx] = False
        self.snps_selected.append(min_p_idx)  # type: ignore
        self.logger.info(
            "Iteration 1: Selected SNP %s with p-value %g",
            snp_ids[min_p_idx],
            p_values[min_p_idx],
        )

        # Iterative model selection
        continue_selection = True
        iteration = 2
        while continue_selection:
            self.logger.info(
                "Iteration %d: Calculating conditional statistics", iteration
            )
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
                    snp_ids[min_cond_p_idx],
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
                        snp_ids[min_cond_p_idx],
                        len(self.snps_selected),
                        ", ".join(snp_ids[self.selected_mask]),
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
                        snp_ids[min_cond_p_idx],
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
                    "original_beta": [
                        self.original_beta[i] for i in self.snps_selected
                    ],
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
            self.logger.info(
                "Filtered %d SNPs due to collinearity.", self.collinear_filtered
            )
            self.logger.info(
                "Removed %d SNPs during backward elimination.", self.backward_removed
            )

            return result
        else:
            self.logger.info("No SNPs selected in the final model.")
            return pd.DataFrame()

    def _calculate_conditional_stats(self):
        """Calculate conditional statistics for all SNPs given the currently selected SNPs."""
        if sum(self.selected_mask) == 0:
            return self.original_beta, self.original_se, self.original_p

        # Indices of selected SNPs
        selected_indices = np.where(self.selected_mask)[0]

        # Extract the sub-matrix of LD for selected SNPs
        ld_selected = self.ld_matrix[np.ix_(selected_indices, selected_indices)]

        # For numerical stability, add a small value to diagonal
        np.fill_diagonal(ld_selected, ld_selected.diagonal() + 1e-6)

        D_N = np.ones(len(selected_indices))

        # Calculate inverse of LD matrix
        try:
            ld_inv = np.linalg.inv(ld_selected)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            self.logger.warning("LD matrix is singular, using pseudo-inverse")
            ld_inv = np.linalg.pinv(ld_selected)

        # Selected beta values
        beta_selected = self.original_beta[selected_indices]

        # Initialize arrays for conditional statistics
        cond_beta = np.zeros(self.total_snps)
        cond_se = np.zeros(self.total_snps)
        cond_p = np.ones(self.total_snps)

        # Calculate conditional statistics for each unselected SNP
        for i in range(self.total_snps):
            if self.selected_mask[i]:
                # For selected SNPs, use joint statistics later
                continue

            # Get LD between this SNP and selected SNPs
            ld_with_selected = self.ld_matrix[i, selected_indices]
            B2 = 1.0
            # maf = self.sumstats["freq"][i]
            # Nd = self.n[0]
            # B2 = 2 * maf * (1 - maf) * Nd

            Z_Bi = ld_with_selected @ ld_inv
            adjustment = (Z_Bi * D_N) @ beta_selected / B2
            cond_beta[i] = self.original_beta[i] - adjustment

            # Calculate conditional standard error using the formula from the paper
            # var(b2|b1) = [D2 - D2X'1(X'1X1)^(-1)X1D2]σ²c
            # Here simplified: D2 is 1, X'1X1 is the LD matrix of selected SNPs
            var_reduction = np.dot(np.dot(ld_with_selected, ld_inv), ld_with_selected)
            # Keep the residual variance constant as mentioned in the paper
            cond_se[i] = (
                self.original_se[i] / np.sqrt(1 - var_reduction)
                if var_reduction < 0.9
                else np.inf
            )

            # Calculate p-value
            if cond_se[i] < np.inf:
                cond_p[i] = self._calculate_p_value(cond_beta[i], cond_se[i])
            else:
                cond_p[i] = 1.0
            self.logger.debug(
                f"SNP {self.snp_ids[i]} has p-value {cond_p[i]:.2e}, beta {cond_beta[i]:.5f}, se {cond_se[i]:.5f}"
            )

        return cond_beta, cond_se, cond_p

    def _calculate_joint_stats(self):
        """Calculate joint statistics for all selected SNPs."""
        selected_indices = np.where(self.selected_mask)[0]

        # Extract the LD matrix for selected SNPs
        ld_selected = self.ld_matrix[np.ix_(selected_indices, selected_indices)]

        # Add small value to diagonal for numerical stability
        np.fill_diagonal(ld_selected, ld_selected.diagonal() + 1e-6)

        # Calculate inverse of LD matrix
        try:
            ld_inv = np.linalg.inv(ld_selected)
        except np.linalg.LinAlgError:
            self.logger.warning(
                "LD matrix for joint analysis is singular, using pseudo-inverse"
            )
            ld_inv = np.linalg.pinv(ld_selected)

        # Selected beta values
        beta_selected = self.original_beta[selected_indices]

        # Calculate joint effects: b = (X'X)^(-1)X'y
        joint_betas = np.dot(ld_inv, beta_selected)

        # Calculate standard errors
        var_joint = np.zeros(len(selected_indices))
        for i in range(len(selected_indices)):
            var_joint[i] = self.original_se[selected_indices[i]] ** 2 * ld_inv[i, i]
        joint_ses = np.sqrt(var_joint)

        # Calculate p-values
        joint_pvals = self._calculate_p_value(joint_betas, joint_ses)

        return joint_betas, joint_ses, joint_pvals

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

            if self.verbose and len(selected_indices_in_window) > 0:
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

            if self.verbose:
                max_r2_idx = selected_indices[np.argmax(ld_subset**2)]
                self.logger.debug(
                    "Max r² = %g with SNP %s", max_r2, self.snp_ids[max_r2_idx]
                )

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
                to_remove.append(
                    (idx, i)
                )  # Store both original index and position in selected list
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
