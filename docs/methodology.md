# Methodology: LD-based Conditional & Joint Analysis

This document provides a self-contained mathematical derivation of the formulas
used in cojopy, following Yang et al. (2012) *Nature Genetics*.
Each section maps equations to the corresponding code paths so that readers can
trace every quantity from theory to implementation.

---

## 1. Notation

| Symbol | Meaning | Code variable |
|--------|---------|---------------|
| $y$ | Phenotype vector ($N \times 1$) | — (not stored; summary-level only) |
| $x_j$ | Genotype vector for SNP $j$ ($N \times 1$), coded as 0/1/2 dosage | — |
| $X$ | Genotype matrix ($N \times p$) | — |
| $\hat{\beta}_j$ | Marginal OLS effect of SNP $j$ | `beta`, `original_beta` |
| $\text{SE}(\hat{\beta}_j)$ | Standard error of $\hat{\beta}_j$ | `se`, `original_se` |
| $p_j$ | P-value for $\hat{\beta}_j$ | `p`, `original_p` |
| $f_j$ | Effect-allele frequency of SNP $j$ | `freq` |
| $N_j$ | Sample size for SNP $j$ | `n` |
| $r_{jk}$ | LD (Pearson) correlation between SNPs $j$ and $k$ | `ld_matrix[j, k]` |
| $V_p$ | Phenotype variance | `pheno_var` |
| $\text{Var}(x_j)$ | Genotype variance $= 2f_j(1-f_j)$ under HWE | `var_x` |
| $D$ | Diagonal matrix with $D_{jj} = N_j \cdot \text{Var}(x_j)$ | `D` |
| $\hat{\beta}_J$ | Joint (multi-SNP) effect estimates | `joint_betas` |
| $\hat{\beta}_{i \mid S}$ | Conditional effect of SNP $i$ given selected set $S$ | `cond_beta` |
| $S$ | Set of selected (conditioning) SNP indices | `selected_indices` |

---

## 2. Linear Regression Model

### 2.1 Single-SNP marginal regression

For each SNP $j$, the marginal model is:

$$
y = x_j \beta_j + \epsilon_j, \qquad \epsilon_j \sim \mathcal{N}(0, \sigma_{\epsilon_j}^2 I)
$$

The OLS estimate is:

$$
\hat{\beta}_j = \frac{x_j' y}{x_j' x_j}
$$

### 2.2 Multi-SNP joint regression

When fitting $p$ SNPs simultaneously:

$$
y = X \beta + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma_\epsilon^2 I)
$$

the OLS normal equations give:

$$
X'X \hat{\beta}_J = X'y
$$

so

$$
\hat{\beta}_J = (X'X)^{-1} X'y
$$

### 2.3 Marginal vs. joint effects

From the marginal estimate for SNP $j$:

$$
\hat{\beta}_j = \frac{x_j' y}{x_j' x_j}
$$

Multiplying both sides by $(x_j' x_j)$:

$$
(x_j' x_j)\hat{\beta}_j = x_j' y
$$

In matrix form for all SNPs:

$$
D \hat{\beta} \approx X'y
$$

where $D = \text{diag}(x_1'x_1, \ldots, x_p'x_p)$ collects the diagonal
entries of $X'X$. This identity bridges marginal and joint effects:

$$
\hat{\beta}_J = (X'X)^{-1} D \hat{\beta}
$$

!!! note "Code reference"
    This relationship is the foundation of the entire COJO method — it allows
    computing joint effects from marginal summary statistics plus an LD matrix,
    without access to individual-level data.

---

## 3. From Individual Data to Summary Statistics

### 3.1 OLS expressions

From standard OLS theory:

$$
\hat{\beta}_j = \frac{x_j' y}{x_j' x_j}, \qquad
\text{SE}(\hat{\beta}_j) = \sqrt{\frac{\sigma_\epsilon^2}{x_j' x_j}}
$$

Since $x_j' x_j = N_j \cdot \text{Var}(x_j)$ when $x_j$ is centred, and
writing $\sigma_\epsilon^2 \approx V_p$ (the residual variance is approximated
by the phenotype variance for small per-SNP effects):

$$
\text{SE}(\hat{\beta}_j) \approx \sqrt{\frac{V_p}{N_j \cdot \text{Var}(x_j)}}
$$

### 3.2 The $X'X$ matrix and LD

The off-diagonal entries of $X'X$ relate to LD:

$$
(X'X)_{jk} = \sum_{i=1}^{N} x_{ij} x_{ik}
\approx N \cdot r_{jk} \cdot \sqrt{\text{Var}(x_j) \cdot \text{Var}(x_k)}
$$

where $r_{jk}$ is the Pearson correlation between genotype vectors $x_j$ and
$x_k$. When SNPs $j$ and $k$ have different sample sizes (as in meta-analysis),
$N$ is replaced by $\min(N_j, N_k)$.

---

## 4. Phenotype Variance Estimation — Eq. (8)

### 4.1 Derivation

Rearranging the SE expression from Section 3.1:

$$
\text{SE}(\hat{\beta}_j)^2 = \frac{V_p}{N_j \cdot \text{Var}(x_j)}
$$

However, this ignores the variance explained by SNP $j$ itself.
A more accurate decomposition (Yang et al. 2012, Eq. 8) accounts for the
SNP's own contribution:

$$
V_p = \text{Var}(x_j) \cdot N_j \cdot \text{SE}(\hat{\beta}_j)^2
    + \text{Var}(x_j) \cdot \hat{\beta}_j^2 \cdot \frac{N_j}{N_j - 1}
$$

Each SNP yields an independent estimate of $V_p$. The median across all SNPs
provides a robust estimator:

$$
\hat{V}_p = \text{median}_j \left\{
  \text{Var}(x_j) \cdot N_j \cdot \text{SE}(\hat{\beta}_j)^2
  + \text{Var}(x_j) \cdot \hat{\beta}_j^2 \cdot \frac{N_j}{N_j - 1}
\right\}
$$

### 4.2 Why median?

The median is preferred over the mean because a small number of SNPs with large
effects can produce extreme per-SNP $V_p$ estimates. The median is resistant to
such outliers.

!!! note "Code reference"
    **`_cal_pheno_var()`** — [`cojopy.py:661-667`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L661-L667)

    ```python
    var_x = 2 * freq * (1 - freq)
    Vp_buf = var_x * n * se**2 + var_x * beta**2 * n / (n - 1)
    return np.median(Vp_buf)
    ```

---

## 5. Effective Sample Size — Eq. (13)

### 5.1 Motivation

In meta-analysis, each SNP may be measured in a different subset of studies,
so $N_j$ varies. Even when $N$ is constant, the "effective" sample size for
reconstructing $(X'X)_{jj}$ must account for the variance explained by the
SNP itself.

### 5.2 Derivation

Starting from the per-SNP variance decomposition (Eq. 8):

$$
V_p = \text{Var}(x_j) \cdot N_j \cdot \text{SE}(\hat{\beta}_j)^2
    + \text{Var}(x_j) \cdot \hat{\beta}_j^2 \cdot \frac{N_j}{N_j - 1}
$$

For $N_j \gg 1$, $\frac{N_j}{N_j-1} \approx 1$, so approximately:

$$
V_p \approx \text{Var}(x_j) \cdot N_j \cdot \text{SE}(\hat{\beta}_j)^2
    + \text{Var}(x_j) \cdot \hat{\beta}_j^2
$$

Solving for $N_j$:

$$
N_j = \frac{V_p - \text{Var}(x_j) \cdot \hat{\beta}_j^2}
           {\text{Var}(x_j) \cdot \text{SE}(\hat{\beta}_j)^2} + 1
$$

Given the global estimate $\hat{V}_p$ from Section 4, this formula yields the
effective sample size for each SNP.

!!! note "Code reference"
    **`_cal_effective_n()`** — [`cojopy.py:669-680`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L669-L680)

    ```python
    var_x = 2 * freq * (1 - freq)
    eff_n = (pheno_var - var_x * beta**2) / (var_x * se**2) + 1
    ```

---

## 6. Reconstructing $X'X$ from Summary Statistics

With $\hat{V}_p$ and per-SNP effective $N_j$ in hand, we can reconstruct the
$X'X$ matrix entirely from summary statistics and an external LD reference.

### 6.1 Genotype variance under HWE

$$
\text{Var}(x_j) = 2 f_j (1 - f_j)
$$

### 6.2 Off-diagonal entries

$$
(X'X)_{jk} = \min(N_j, N_k) \cdot r_{jk}
              \cdot \sqrt{\text{Var}(x_j) \cdot \text{Var}(x_k)},
\qquad j \neq k
$$

The $\min$ accounts for different sample sizes: only the overlapping samples
contribute to the cross-product.

### 6.3 Diagonal entries

$$
(X'X)_{jj} = N_j \cdot \text{Var}(x_j)
$$

### 6.4 The $D$ matrix

The $D$ matrix is simply the diagonal of $X'X$:

$$
D = \text{diag}\bigl(N_1 \cdot \text{Var}(x_1),\; \ldots,\; N_p \cdot \text{Var}(x_p)\bigr)
$$

!!! note "Code reference"
    **`_calculate_joint_stats()`** — [`cojopy.py:458-469`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L458-L469)

    ```python
    XTX = np.zeros((n_snp, n_snp))
    for j in range(n_snp):
        for k in range(n_snp):
            if j == k:
                XTX[j, k] = eff_n[j] * var_x[j]
            else:
                XTX[j, k] = (
                    min(eff_n[j], eff_n[k])
                    * ld_selected[j, k]
                    * np.sqrt(var_x[j] * var_x[k])
                )
    ```

---

## 7. Joint Analysis — Eq. (12)

### 7.1 Joint effect estimates

From Section 2.3, the joint effects are:

$$
\hat{\beta}_J = (X'X)^{-1} D \hat{\beta}
$$

where $\hat{\beta}$ is the vector of marginal effects, $D$ is the diagonal
matrix from Section 6.4, and $X'X$ is the reconstructed matrix from Section 6.

### 7.2 Derivation

Starting from the normal equations:

$$
X'X \hat{\beta}_J = X'y
$$

We showed in Section 2.3 that $D\hat{\beta} \approx X'y$.
Substituting:

$$
X'X \hat{\beta}_J = D \hat{\beta}
$$

$$
\hat{\beta}_J = (X'X)^{-1} D \hat{\beta}
$$

### 7.3 Joint standard errors

The variance-covariance matrix of $\hat{\beta}_J$ is:

$$
\text{Var}(\hat{\beta}_J) = \sigma_\epsilon^2 (X'X)^{-1} \approx V_p (X'X)^{-1}
$$

Therefore:

$$
\text{SE}(\hat{\beta}_{J,j}) = \sqrt{V_p \cdot \bigl[(X'X)^{-1}\bigr]_{jj}}
$$

### 7.4 Joint P-values

$$
z_j = \frac{\hat{\beta}_{J,j}}{\text{SE}(\hat{\beta}_{J,j})}, \qquad
p_j = 2\,\Phi(-|z_j|)
$$

where $\Phi$ is the standard normal CDF.

!!! note "Code reference"
    **`_calculate_joint_stats()`** — [`cojopy.py:471-509`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L471-L509)

    ```python
    # Joint effects (Eq. 12)
    joint_betas = XTX_inv @ D @ beta_selected

    # Joint SEs
    var_joint = self.pheno_var * XTX_inv.diagonal()
    joint_ses = np.sqrt(var_joint)
    ```

---

## 8. Conditional Analysis

### 8.1 Setup

Partition the SNPs into a selected (conditioning) set $S$ and the remaining
set $S^c$. We want the effect of each SNP $i \in S^c$ conditional on all SNPs
in $S$.

Define the block structure of $X'X$:

$$
X'X = \begin{pmatrix}
B_1 & C' \\
C   & B_2
\end{pmatrix}
$$

where:

- $B_1 = (X'X)_{S,S}$ — the $|S| \times |S|$ sub-matrix among selected SNPs
- $C_i = (X'X)_{i,S}$ — the row vector linking SNP $i$ to the selected set
- $B_{2,ii} = (X'X)_{ii} = D_{ii}$ — the diagonal entry for SNP $i$

### 8.2 Conditional effect

For a single SNP $i \in S^c$, the conditional effect given $S$ is:

$$
\hat{\beta}_{i \mid S}
= \hat{\beta}_i
  - \frac{C_i \, B_1^{-1} \, D_S \, \hat{\beta}_S}{D_{ii}}
$$

where:

- $\hat{\beta}_i$ is the marginal effect of SNP $i$
- $\hat{\beta}_S$ is the vector of marginal effects for SNPs in $S$
- $D_S = \text{diag}(N_j \cdot \text{Var}(x_j))_{j \in S}$ is the diagonal sub-matrix for the selected set
- $D_{ii} = N_i \cdot \text{Var}(x_i)$

### 8.3 Derivation sketch

From the partitioned normal equations, the full system is:

$$
\begin{pmatrix}
B_1 & C_i' \\
C_i & D_{ii}
\end{pmatrix}
\begin{pmatrix}
\hat{\beta}_{S \mid i} \\
\hat{\beta}_{i \mid S}
\end{pmatrix}
=
\begin{pmatrix}
D_S \hat{\beta}_S \\
D_{ii} \hat{\beta}_i
\end{pmatrix}
$$

From the second row:

$$
C_i \hat{\beta}_{S \mid i} + D_{ii} \hat{\beta}_{i \mid S} = D_{ii} \hat{\beta}_i
$$

From the first row, $\hat{\beta}_{S \mid i} \approx B_1^{-1} D_S \hat{\beta}_S$
(treating the contribution of single SNP $i$ as negligible). Substituting:

$$
D_{ii} \hat{\beta}_{i \mid S} = D_{ii} \hat{\beta}_i - C_i B_1^{-1} D_S \hat{\beta}_S
$$

$$
\hat{\beta}_{i \mid S} = \hat{\beta}_i - \frac{C_i B_1^{-1} D_S \hat{\beta}_S}{D_{ii}}
$$

### 8.4 Conditional standard error

$$
\text{SE}(\hat{\beta}_{i \mid S}) = \sqrt{\frac{V_p}{D_{ii}}}
$$

This follows from the fact that, after conditioning, the residual variance for
SNP $i$ is approximately $V_p / D_{ii}$, treating the selected SNPs as fixed
covariates.

### 8.5 Conditional P-value

$$
z_i = \frac{\hat{\beta}_{i \mid S}}{\text{SE}(\hat{\beta}_{i \mid S})}, \qquad
p_i = 2\,\Phi(-|z_i|)
$$

!!! note "Code reference"
    **`_calculate_conditional_stats()`** — [`cojopy.py:329-433`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L329-L433)

    ```python
    # C_i @ B1_inv
    Z_Bi = C @ B1_inv
    # Conditional beta
    adjustment = (Z_Bi * D1.diagonal()) @ beta_selected / D2.diagonal()[i]
    cond_beta[idx] = beta_unselected[i] - adjustment
    # Conditional SE
    cond_se[idx] = np.sqrt(pheno_var / D2.diagonal()[i])
    ```

---

## 9. Stepwise Selection Algorithm

The stepwise model selection procedure combines conditional analysis (forward
step) and joint analysis (backward elimination) in an iterative loop.

### 9.1 Forward step

1. **Initialise**: select the SNP with the smallest marginal P-value if
   $p < p_\text{cutoff}$ (default $5 \times 10^{-8}$).
2. **Iterate**: compute conditional statistics for all remaining SNPs given
   the current selected set $S$; identify the SNP with the smallest
   conditional P-value.
3. **Collinearity check**: before adding a candidate SNP, verify
   $r^2 < r^2_\text{cutoff}$ (default 0.9) with every SNP already in $S$.
   If the check fails, exclude the candidate and continue.
4. **Add**: if the conditional P-value passes the threshold and collinearity
   check, add the SNP to $S$.

### 9.2 Backward elimination

After each forward addition, perform joint analysis on all SNPs in $S$.
Remove any SNP whose joint P-value exceeds $p_\text{cutoff}$. Record removed
SNPs to prevent re-selection in subsequent iterations (avoids infinite loops).

### 9.3 Termination

The algorithm stops when:

- No remaining SNP has a conditional P-value below $p_\text{cutoff}$, or
- The candidate SNP was previously removed by backward elimination, or
- No available SNPs remain.

### 9.4 Final output

After convergence, a final joint analysis is run on the selected set to
produce the reported joint effect estimates, SEs, and P-values.

!!! note "Code reference"
    **`conditional_selection()`** — [`cojopy.py:156-327`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L156-L327)

    The forward step is the `while continue_selection` loop; backward elimination
    is handled by `_backward_elimination()` at [`cojopy.py:627-653`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L627-L653).

---

## 10. Implementation Notes

### 10.1 Log-space P-value computation

For highly significant associations, P-values can underflow to zero in
64-bit floating point. cojopy computes P-values in log space to avoid this:

$$
\log p = \log 2 + \log \Phi(-|z|)
$$

and then exponentiates, clamping to the smallest representable positive float
(`np.finfo(np.float64).tiny`).

!!! note "Code reference"
    **`_calculate_p_value()`** — [`cojopy.py:655-659`](https://github.com/Jianhua-Wang/cojopy/blob/main/cojopy/cojopy.py#L655-L659)

    ```python
    log_sf = norm.logsf(abs(z_scores))
    log_p = np.log(2) + log_sf
    return np.maximum(np.exp(log_p), np.finfo(np.float64).tiny)
    ```

### 10.2 Singular matrix handling

When the selected SNP set includes near-collinear variants, $X'X$ or $B_1$
may be numerically singular. In such cases, the code falls back to the
Moore-Penrose pseudo-inverse (`np.linalg.pinv`) and logs a warning.

### 10.3 Negative variance

If $(X'X)^{-1}_{jj} < 0$ (which can occur with ill-conditioned matrices),
the joint SE becomes undefined. The code sets SE = $\infty$ and P = 1.0 for
such SNPs, effectively excluding them from interpretation.

---

## 11. Reference

Yang J, Ferreira T, Morris AP, et al. Conditional and joint multiple-SNP
analysis of GWAS summary statistics identifies additional variants influencing
complex traits. *Nature Genetics*. 2012;44(4):369-375.
doi:[10.1038/ng.2213](https://doi.org/10.1038/ng.2213)
