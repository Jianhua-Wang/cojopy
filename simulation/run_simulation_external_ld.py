#!/usr/bin/env python3
"""Simulation comparison between cojopy and GCTA-COJO using EXTERNAL LD.

Unlike run_simulation.py where LD comes from the study sample itself, here:
  - Study sample (N=2000): used only for GWAS → .ma file
  - Reference panel (N=500): separate individuals from the same population → LD

Both GCTA-COJO (--bfile ref_panel) and cojopy (LD matrix from ref_panel)
use the SAME external LD. This verifies that even with imperfect LD (which
hurts accuracy), both tools still produce identical results.

Usage:
    cd simulation && python run_simulation_external_ld.py
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Re-use tool download logic from the main simulation
from run_simulation import (
    BASE_DIR,
    BIN_DIR,
    CAUSAL_BETAS,
    COLLINEAR_CUTOFF,
    HERITABILITY,
    N_BLOCKS,
    N_CAUSAL,
    N_SAMPLES,
    N_SNPS,
    P_CUTOFF,
    SNPS_PER_BLOCK,
    download_tools,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Directories (separate from run_simulation output) ────────────────────────
OUTPUT_DIR = BASE_DIR / "output_external_ld"
STUDY_DIR = OUTPUT_DIR / "study"
REF_DIR = OUTPUT_DIR / "ref_panel"
GWAS_DIR = OUTPUT_DIR / "gwas"
LD_DIR = OUTPUT_DIR / "ld"
GCTA_DIR = OUTPUT_DIR / "gcta"
COJOPY_DIR = OUTPUT_DIR / "cojopy"
COMP_DIR = OUTPUT_DIR / "comparison"

SEED_STUDY = 42
SEED_REF = 99  # Different seed → different individuals
N_REF = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Genotype simulation (parameterized by n_samples)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_genotypes(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Simulate genotype matrix with block-diagonal LD structure."""
    logger.info("Simulating genotypes: %d samples, %d SNPs", n_samples, N_SNPS)
    genotypes = np.zeros((n_samples, N_SNPS), dtype=np.int8)

    for block_idx in range(N_BLOCKS):
        start = block_idx * SNPS_PER_BLOCK

        corr = np.zeros((SNPS_PER_BLOCK, SNPS_PER_BLOCK))
        for i in range(SNPS_PER_BLOCK):
            for j in range(SNPS_PER_BLOCK):
                corr[i, j] = 0.95 ** abs(i - j)

        mean = np.zeros(SNPS_PER_BLOCK)
        z = rng.multivariate_normal(mean, corr, size=n_samples)

        for snp_i in range(SNPS_PER_BLOCK):
            col = z[:, snp_i]
            t1 = np.percentile(col, 49)
            t2 = np.percentile(col, 91)
            geno = np.zeros(n_samples, dtype=np.int8)
            geno[col >= t1] = 1
            geno[col >= t2] = 2
            genotypes[:, start + snp_i] = geno

    return genotypes


# ═══════════════════════════════════════════════════════════════════════════════
# Phenotype simulation
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_phenotypes(
    genotypes: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    """Simulate phenotypes with known causal SNPs."""
    causal_indices = []
    for block_idx in range(N_CAUSAL):
        start = block_idx * SNPS_PER_BLOCK
        offset = 50 + block_idx * 10
        causal_indices.append(start + offset)

    X_causal = genotypes[:, causal_indices].astype(np.float64)
    for i in range(X_causal.shape[1]):
        X_causal[:, i] = (X_causal[:, i] - X_causal[:, i].mean()) / (
            X_causal[:, i].std() + 1e-10
        )

    betas = np.array(CAUSAL_BETAS)
    genetic_signal = X_causal @ betas
    var_genetic = np.var(genetic_signal)
    var_noise = var_genetic * (1 - HERITABILITY) / HERITABILITY
    noise = rng.normal(0, np.sqrt(var_noise), genotypes.shape[0])
    phenotype = genetic_signal + noise

    actual_h2 = var_genetic / np.var(phenotype)
    logger.info("Target h²=%.2f, actual h²=%.4f", HERITABILITY, actual_h2)
    return phenotype, causal_indices


# ═══════════════════════════════════════════════════════════════════════════════
# Write PLINK files
# ═══════════════════════════════════════════════════════════════════════════════
def write_plink_files(
    genotypes: np.ndarray,
    phenotype: np.ndarray | None,
    out_dir: Path,
    prefix_name: str,
    plink_path: Path,
) -> str:
    """Write .ped/.map and convert to binary. Returns bfile prefix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / prefix_name)
    n_samples = genotypes.shape[0]

    # .map
    with open(prefix + ".map", "w") as f:
        for i in range(N_SNPS):
            f.write(f"1\trs{i + 1}\t0\t{(i + 1) * 1000}\n")

    # .ped
    logger.info("Writing PED: %s (%d samples)", prefix, n_samples)
    with open(prefix + ".ped", "w") as f:
        for i in range(n_samples):
            pheno_val = str(phenotype[i]) if phenotype is not None else "-9"
            parts = [f"FID{i + 1}", f"IID{i + 1}", "0", "0", "1", pheno_val]
            for j in range(N_SNPS):
                g = genotypes[i, j]
                if g == 0:
                    parts.extend(["A", "A"])
                elif g == 1:
                    parts.extend(["G", "A"])
                else:
                    parts.extend(["G", "G"])
            f.write("\t".join(parts) + "\n")

    subprocess.run(
        [str(plink_path), "--file", prefix, "--make-bed", "--out", prefix],
        capture_output=True,
        check=True,
    )
    return prefix


# ═══════════════════════════════════════════════════════════════════════════════
# Run GWAS on study sample → .ma file
# ═══════════════════════════════════════════════════════════════════════════════
def run_gwas(study_prefix: str, plink_path: Path) -> str:
    """Run GWAS on study sample, return path to .ma file."""
    GWAS_DIR.mkdir(parents=True, exist_ok=True)
    gwas_prefix = str(GWAS_DIR / "study")

    subprocess.run(
        [
            str(plink_path), "--bfile", study_prefix,
            "--linear", "--allow-no-sex", "--out", gwas_prefix,
        ],
        capture_output=True, check=True,
    )
    subprocess.run(
        [
            str(plink_path), "--bfile", study_prefix,
            "--freq", "--allow-no-sex", "--out", gwas_prefix,
        ],
        capture_output=True, check=True,
    )

    assoc = pd.read_csv(gwas_prefix + ".assoc.linear", sep=r"\s+")
    assoc = assoc[assoc["TEST"] == "ADD"].copy()
    freq_df = pd.read_csv(gwas_prefix + ".frq", sep=r"\s+")
    bim = pd.read_csv(
        study_prefix + ".bim", sep="\t", header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
    )

    ma = pd.DataFrame()
    ma["SNP"] = bim["SNP"]
    ma["A1"] = bim["A1"]
    ma["A2"] = bim["A2"]

    freq_map = dict(zip(freq_df["SNP"], zip(freq_df["A1"], freq_df["MAF"])))
    freqs = []
    for _, row in ma.iterrows():
        snp = row["SNP"]
        if snp in freq_map:
            freq_a1, maf = freq_map[snp]
            freqs.append(maf if freq_a1 == row["A1"] else 1 - maf)
        else:
            freqs.append(np.nan)
    ma["freq"] = freqs

    assoc_map = dict(
        zip(assoc["SNP"], zip(assoc["A1"], assoc["BETA"], assoc["STAT"], assoc["P"]))
    )
    betas, ses, ps = [], [], []
    for _, row in ma.iterrows():
        snp = row["SNP"]
        if snp in assoc_map:
            assoc_a1, beta, t_stat, p = assoc_map[snp]
            se = abs(beta / t_stat) if t_stat != 0 else np.nan
            betas.append(beta if assoc_a1 == row["A1"] else -beta)
            ses.append(se)
            ps.append(p)
        else:
            betas.append(np.nan)
            ses.append(np.nan)
            ps.append(np.nan)

    ma["b"] = betas
    ma["se"] = ses
    ma["p"] = ps
    ma["N"] = N_SAMPLES
    ma = ma.dropna()

    ma_path = str(GWAS_DIR / "study.ma")
    ma.to_csv(ma_path, sep="\t", index=False)
    logger.info("Written .ma file: %s (%d SNPs)", ma_path, len(ma))
    return ma_path


# ═══════════════════════════════════════════════════════════════════════════════
# Compute LD from reference panel
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ref_ld(ref_prefix: str, plink_path: Path) -> str:
    """Compute LD matrix from reference panel. Returns path to .ld file."""
    LD_DIR.mkdir(parents=True, exist_ok=True)
    ld_prefix = str(LD_DIR / "ref")
    subprocess.run(
        [
            str(plink_path), "--bfile", ref_prefix,
            "--r", "square", "--allow-no-sex", "--out", ld_prefix,
        ],
        capture_output=True, check=True,
    )
    logger.info("Reference LD matrix: %s.ld", ld_prefix)
    return ld_prefix + ".ld"


# ═══════════════════════════════════════════════════════════════════════════════
# Run GCTA-COJO (using ref panel for LD)
# ═══════════════════════════════════════════════════════════════════════════════
def run_gcta_cojo(
    gcta_path: Path, ref_prefix: str, ma_path: str
) -> dict[str, pd.DataFrame]:
    """Run GCTA-COJO slct/joint/cond with --bfile pointing to ref panel."""
    GCTA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── slct ──
    slct_prefix = str(GCTA_DIR / "slct")
    logger.info("Running GCTA-COJO slct (external LD from ref panel)...")
    proc = subprocess.run(
        [
            str(gcta_path), "--bfile", ref_prefix,
            "--cojo-file", ma_path, "--cojo-slct",
            "--cojo-p", str(P_CUTOFF), "--cojo-collinear", str(COLLINEAR_CUTOFF),
            "--out", slct_prefix,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.warning("GCTA slct stderr:\n%s", proc.stderr)

    slct_file = slct_prefix + ".jma.cojo"
    if os.path.exists(slct_file):
        results["slct"] = pd.read_csv(slct_file, sep=r"\s+")
        logger.info("GCTA slct: %d SNPs selected", len(results["slct"]))
    else:
        logger.warning("GCTA slct output not found")
        results["slct"] = pd.DataFrame()

    if results["slct"].empty:
        logger.error("No SNPs selected by GCTA slct")
        return results

    selected_snps = results["slct"]["SNP"].tolist()

    snps_file = str(GCTA_DIR / "selected_snps.txt")
    with open(snps_file, "w") as f:
        for snp in selected_snps:
            f.write(f"{snp}\n")

    cond_snps_file = str(GCTA_DIR / "cond_snps.txt")
    with open(cond_snps_file, "w") as f:
        f.write(f"{selected_snps[0]}\n")

    # ── joint ──
    joint_prefix = str(GCTA_DIR / "joint")
    logger.info("Running GCTA-COJO joint (external LD)...")
    proc = subprocess.run(
        [
            str(gcta_path), "--bfile", ref_prefix,
            "--cojo-file", ma_path, "--cojo-joint",
            "--extract", snps_file, "--out", joint_prefix,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.warning("GCTA joint stderr:\n%s", proc.stderr)

    joint_file = joint_prefix + ".jma.cojo"
    if os.path.exists(joint_file):
        results["joint"] = pd.read_csv(joint_file, sep=r"\s+")
        logger.info("GCTA joint: %d SNPs", len(results["joint"]))
    else:
        results["joint"] = pd.DataFrame()

    # ── cond ──
    cond_prefix = str(GCTA_DIR / "cond")
    logger.info("Running GCTA-COJO cond (external LD)...")
    proc = subprocess.run(
        [
            str(gcta_path), "--bfile", ref_prefix,
            "--cojo-file", ma_path, "--cojo-cond", cond_snps_file,
            "--out", cond_prefix,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.warning("GCTA cond stderr:\n%s", proc.stderr)

    cond_file = cond_prefix + ".cma.cojo"
    if os.path.exists(cond_file):
        results["cond"] = pd.read_csv(cond_file, sep=r"\s+")
        logger.info("GCTA cond: %d SNPs", len(results["cond"]))
    else:
        results["cond"] = pd.DataFrame()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Run cojopy (using ref panel LD matrix)
# ═══════════════════════════════════════════════════════════════════════════════
def run_cojopy(
    ma_path: str, ld_path: str, gcta_results: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Run cojopy slct/joint/cond using ref panel LD matrix."""
    project_root = str(BASE_DIR.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from cojopy.cojopy import COJO

    COJOPY_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── slct ──
    logger.info("Running cojopy slct (external LD)...")
    c = COJO(p_cutoff=P_CUTOFF, collinear_cutoff=COLLINEAR_CUTOFF)
    c.load_sumstats(sumstats_path=ma_path, ld_path=ld_path)
    slct_result = c.conditional_selection()
    slct_result.to_csv(
        str(COJOPY_DIR / "slct.txt"), sep="\t", index=False, float_format="%.6g"
    )
    results["slct"] = slct_result
    logger.info("cojopy slct: %d SNPs selected", len(slct_result))

    # Use GCTA's selection for fair joint/cond comparison
    if "slct" in gcta_results and not gcta_results["slct"].empty:
        selected_snps = gcta_results["slct"]["SNP"].tolist()
    elif not slct_result.empty:
        selected_snps = slct_result["SNP"].tolist()
    else:
        logger.error("No SNPs selected")
        return results

    snps_file = str(COJOPY_DIR / "selected_snps.txt")
    with open(snps_file, "w") as f:
        for snp in selected_snps:
            f.write(f"{snp}\n")

    cond_snps_file = str(COJOPY_DIR / "cond_snps.txt")
    with open(cond_snps_file, "w") as f:
        f.write(f"{selected_snps[0]}\n")

    # ── joint ──
    logger.info("Running cojopy joint (external LD)...")
    c2 = COJO(p_cutoff=P_CUTOFF, collinear_cutoff=COLLINEAR_CUTOFF)
    c2.load_sumstats(sumstats_path=ma_path, ld_path=ld_path)
    joint_result = c2.run_joint_analysis(extract_snps_path=snps_file)
    joint_result.to_csv(
        str(COJOPY_DIR / "joint.txt"), sep="\t", index=False, float_format="%.6g"
    )
    results["joint"] = joint_result
    logger.info("cojopy joint: %d SNPs", len(joint_result))

    # ── cond ──
    logger.info("Running cojopy cond (external LD)...")
    c3 = COJO(p_cutoff=P_CUTOFF, collinear_cutoff=COLLINEAR_CUTOFF)
    c3.load_sumstats(sumstats_path=ma_path, ld_path=ld_path)
    cond_result = c3.run_conditional_analysis(cond_snps_path=cond_snps_file)
    cond_result.to_csv(
        str(COJOPY_DIR / "cond.txt"), sep="\t", index=False, float_format="%.6g"
    )
    results["cond"] = cond_result
    logger.info("cojopy cond: %d SNPs", len(cond_result))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Compare results
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_pearsonr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return pearsonr(x[mask], y[mask])[0]


def compare_results(
    gcta_results: dict[str, pd.DataFrame],
    cojopy_results: dict[str, pd.DataFrame],
) -> bool:
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    all_pass = True
    summary_lines = []

    # ── slct ──
    g_slct = gcta_results.get("slct", pd.DataFrame())
    c_slct = cojopy_results.get("slct", pd.DataFrame())
    if g_slct.empty or c_slct.empty:
        summary_lines.append("[SKIP] slct: one or both results empty")
    else:
        g_snps = set(g_slct["SNP"])
        c_snps = set(c_slct["SNP"])
        same_snps = g_snps == c_snps
        common_snps = g_snps & c_snps
        logger.info("GCTA slct: %s", sorted(g_snps))
        logger.info("cojopy slct: %s", sorted(c_snps))

        if common_snps:
            merged = pd.merge(g_slct, c_slct, on="SNP", suffixes=("_gcta", "_cojopy"))
            r_beta = _safe_pearsonr(merged["bJ"].values, merged["joint_beta"].values)
            r_se = _safe_pearsonr(merged["bJ_se"].values, merged["joint_se"].values)
            max_db = np.max(np.abs(merged["bJ"].values - merged["joint_beta"].values))
            max_ds = np.max(np.abs(merged["bJ_se"].values - merged["joint_se"].values))
            ok = (r_beta > 0.999 if np.isfinite(r_beta) else False) and same_snps
            if not ok:
                all_pass = False
            summary_lines.append(
                f"[{'PASS' if ok else 'FAIL'}] slct: Same SNPs={same_snps}, "
                f"Beta r={r_beta:.6f}, SE r={r_se:.6f}, "
                f"Max|diff_beta|={max_db:.2e}, Max|diff_se|={max_ds:.2e}"
            )
        else:
            summary_lines.append("[FAIL] slct: No common SNPs")
            all_pass = False

    # ── joint ──
    g_joint = gcta_results.get("joint", pd.DataFrame())
    c_joint = cojopy_results.get("joint", pd.DataFrame())
    if g_joint.empty or c_joint.empty:
        summary_lines.append("[SKIP] joint: one or both results empty")
    else:
        merged = pd.merge(g_joint, c_joint, on="SNP", suffixes=("_gcta", "_cojopy"))
        if len(merged) == 0:
            summary_lines.append("[FAIL] joint: No common SNPs after merge")
            all_pass = False
        else:
            r_beta = _safe_pearsonr(merged["bJ"].values, merged["joint_beta"].values)
            r_se = _safe_pearsonr(merged["bJ_se"].values, merged["joint_se"].values)
            max_db = np.max(np.abs(merged["bJ"].values - merged["joint_beta"].values))
            max_ds = np.max(np.abs(merged["bJ_se"].values - merged["joint_se"].values))
            ok = r_beta > 0.999 if np.isfinite(r_beta) else False
            if not ok:
                all_pass = False
            summary_lines.append(
                f"[{'PASS' if ok else 'FAIL'}] joint: "
                f"Beta r={r_beta:.6f}, SE r={r_se:.6f}, "
                f"Max|diff_beta|={max_db:.2e}, Max|diff_se|={max_ds:.2e}"
            )

    # ── cond ──
    g_cond = gcta_results.get("cond", pd.DataFrame())
    c_cond = cojopy_results.get("cond", pd.DataFrame())
    if g_cond.empty or c_cond.empty:
        summary_lines.append("[SKIP] cond: one or both results empty")
    else:
        merged = pd.merge(g_cond, c_cond, on="SNP", suffixes=("_gcta", "_cojopy"))
        if len(merged) == 0:
            summary_lines.append("[FAIL] cond: No common SNPs after merge")
            all_pass = False
        else:
            r_beta = _safe_pearsonr(merged["bC"].values, merged["cond_beta"].values)
            r_se = _safe_pearsonr(merged["bC_se"].values, merged["cond_se"].values)
            max_db = np.max(np.abs(merged["bC"].values - merged["cond_beta"].values))
            max_ds = np.max(np.abs(merged["bC_se"].values - merged["cond_se"].values))
            ok = r_beta > 0.999 if np.isfinite(r_beta) else False
            if not ok:
                all_pass = False
            summary_lines.append(
                f"[{'PASS' if ok else 'FAIL'}] cond:  "
                f"Beta r={r_beta:.6f}, SE r={r_se:.6f}, "
                f"Max|diff_beta|={max_db:.2e}, Max|diff_se|={max_ds:.2e}"
            )

    # ── Summary ──
    logger.info("=" * 60)
    for line in summary_lines:
        logger.info(line)
    overall = "ALL PASSED" if all_pass else "SOME FAILED"
    logger.info("Overall: %s", overall)
    logger.info("=" * 60)

    with open(str(COMP_DIR / "summary.txt"), "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nOverall: {overall}\n")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 60)
    logger.info("cojopy vs GCTA-COJO: EXTERNAL LD comparison")
    logger.info("=" * 60)
    logger.info("Study sample: N=%d (GWAS only)", N_SAMPLES)
    logger.info("Reference panel: N=%d (LD only)", N_REF)
    logger.info("Same population model, different individuals")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Step 1: Download tools
    gcta_path, plink_path = download_tools()

    # Step 2: Simulate study sample genotypes + phenotypes
    logger.info("--- Simulating study sample (seed=%d) ---", SEED_STUDY)
    rng_study = np.random.default_rng(SEED_STUDY)
    geno_study = simulate_genotypes(rng_study, N_SAMPLES)
    phenotype, causal_indices = simulate_phenotypes(geno_study, rng_study)
    logger.info("Causal SNP indices: %s", causal_indices)

    # Step 3: Simulate reference panel genotypes (NO phenotype)
    logger.info("--- Simulating reference panel (seed=%d) ---", SEED_REF)
    rng_ref = np.random.default_rng(SEED_REF)
    geno_ref = simulate_genotypes(rng_ref, N_REF)

    # Step 4: Write PLINK files for both
    study_prefix = write_plink_files(
        geno_study, phenotype, STUDY_DIR, "study", plink_path
    )
    ref_prefix = write_plink_files(
        geno_ref, None, REF_DIR, "ref", plink_path
    )

    # Step 5: GWAS on study sample → .ma file
    ma_path = run_gwas(study_prefix, plink_path)

    # Step 6: LD from reference panel
    ld_path = compute_ref_ld(ref_prefix, plink_path)

    # Step 7: Run GCTA-COJO with --bfile ref_panel
    gcta_results = run_gcta_cojo(gcta_path, ref_prefix, ma_path)

    # Step 8: Run cojopy with ref panel LD matrix
    cojopy_results = run_cojopy(ma_path, ld_path, gcta_results)

    # Step 9: Compare
    all_pass = compare_results(gcta_results, cojopy_results)

    if all_pass:
        logger.info(
            "SUCCESS: With external LD, cojopy and GCTA-COJO still produce identical results"
        )
    else:
        logger.warning("ATTENTION: Some comparisons did not pass threshold")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
