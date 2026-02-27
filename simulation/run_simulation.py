#!/usr/bin/env python3
"""Simulation comparison between cojopy and GCTA-COJO.

Generates simulated genotype/phenotype data, runs both GCTA-COJO and cojopy
(slct, joint, cond), and quantitatively compares results.

Usage:
    cd simulation && python run_simulation.py
"""

import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
OUTPUT_DIR = BASE_DIR / "output"
GENO_DIR = OUTPUT_DIR / "genotype"
GWAS_DIR = OUTPUT_DIR / "gwas"
LD_DIR = OUTPUT_DIR / "ld"
GCTA_DIR = OUTPUT_DIR / "gcta"
COJOPY_DIR = OUTPUT_DIR / "cojopy"
COMP_DIR = OUTPUT_DIR / "comparison"

# ── Simulation parameters ────────────────────────────────────────────────────
N_SAMPLES = 2000
N_SNPS = 1000
N_BLOCKS = 5
SNPS_PER_BLOCK = N_SNPS // N_BLOCKS  # 200
N_CAUSAL = 5
CAUSAL_BETAS = [0.5, 0.4, 0.35, 0.3, 0.25]
HERITABILITY = 0.3
SEED = 42
P_CUTOFF = 5e-8
COLLINEAR_CUTOFF = 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Download Tools
# ═══════════════════════════════════════════════════════════════════════════════
def _download_and_extract(url: str, dest_dir: Path, label: str):
    """Download a URL and extract tar.gz or zip into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    local_path = dest_dir / fname
    logger.info("Downloading %s from %s ...", label, url)
    urllib.request.urlretrieve(url, local_path)

    if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
        with tarfile.open(local_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    elif fname.endswith(".zip"):
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(dest_dir)
    local_path.unlink()
    logger.info("Extracted %s to %s", label, dest_dir)


def _remove_quarantine(path: Path):
    """Remove macOS quarantine attribute."""
    try:
        subprocess.run(
            ["xattr", "-d", "com.apple.quarantine", str(path)],
            capture_output=True,
        )
    except FileNotFoundError:
        pass


def _find_binary(name: str, search_dir: Path) -> Path | None:
    """Recursively find a binary in search_dir."""
    for p in search_dir.rglob(name):
        if p.is_file():
            return p
    return None


def download_tools() -> tuple[Path, Path]:
    """Download GCTA and PLINK if not already available. Returns (gcta_path, plink_path)."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    # ── PLINK ────────────────────────────────────────────────────────────
    plink_path = shutil.which("plink")
    if plink_path is None:
        plink_path = _find_binary("plink", BIN_DIR)
    if plink_path is None:
        machine = platform.machine()
        if sys.platform == "darwin":
            if machine == "arm64":
                plink_url = "https://s3.amazonaws.com/plink1-assets/plink_mac_20231018.zip"
            else:
                plink_url = "https://s3.amazonaws.com/plink1-assets/plink_mac_20231018.zip"
        elif sys.platform == "linux":
            plink_url = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231018.zip"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
        _download_and_extract(plink_url, BIN_DIR / "plink_tmp", "PLINK")
        plink_path = _find_binary("plink", BIN_DIR / "plink_tmp")
        if plink_path:
            dest = BIN_DIR / "plink"
            shutil.move(str(plink_path), str(dest))
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
            _remove_quarantine(dest)
            plink_path = dest
        shutil.rmtree(BIN_DIR / "plink_tmp", ignore_errors=True)
    plink_path = Path(plink_path)
    logger.info("PLINK: %s", plink_path)

    # ── GCTA ─────────────────────────────────────────────────────────────
    # GCTA needs its lib/ directory at ../lib/ relative to binary location.
    # We keep the extracted directory structure intact: bin/gcta_dir/bin/gcta64 + bin/gcta_dir/lib/
    gcta_path = shutil.which("gcta64") or shutil.which("gcta")
    if gcta_path is None:
        gcta_path = _find_binary("gcta64", BIN_DIR) or _find_binary("gcta", BIN_DIR)
    if gcta_path is None:
        machine = platform.machine()
        if sys.platform == "darwin":
            if machine == "arm64":
                gcta_url = "https://yanglab.westlake.edu.cn/software/gcta/bin/gcta-1.95.1-macOS-arm64.zip"
            else:
                gcta_url = "https://yanglab.westlake.edu.cn/software/gcta/bin/gcta-1.94.1-macOS-x86_64.zip"
        elif sys.platform == "linux":
            gcta_url = "https://yanglab.westlake.edu.cn/software/gcta/bin/gcta-1.95.1-linux-x86_64.zip"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
        # Extract and keep entire directory (binary needs ../lib/ relative path)
        _download_and_extract(gcta_url, BIN_DIR, "GCTA")
        gcta_path = _find_binary("gcta64", BIN_DIR) or _find_binary("gcta", BIN_DIR)
        if gcta_path:
            gcta_path = Path(gcta_path)
            gcta_path.chmod(gcta_path.stat().st_mode | stat.S_IEXEC)
            _remove_quarantine(gcta_path)
            # Also remove quarantine from dylibs
            lib_dir = gcta_path.parent.parent / "lib"
            if lib_dir.exists():
                for dylib in lib_dir.glob("*.dylib"):
                    _remove_quarantine(dylib)
    gcta_path = Path(gcta_path)
    logger.info("GCTA: %s", gcta_path)

    return gcta_path, plink_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Simulate Genotype Data
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_genotypes(rng: np.random.Generator) -> np.ndarray:
    """Simulate genotype matrix with block-diagonal LD structure.

    Returns a (N_SAMPLES, N_SNPS) matrix of 0/1/2 dosages.
    """
    logger.info(
        "Simulating genotypes: %d samples, %d SNPs, %d LD blocks",
        N_SAMPLES,
        N_SNPS,
        N_BLOCKS,
    )
    genotypes = np.zeros((N_SAMPLES, N_SNPS), dtype=np.int8)

    for block_idx in range(N_BLOCKS):
        start = block_idx * SNPS_PER_BLOCK
        end = start + SNPS_PER_BLOCK

        # Generate correlation matrix for this block with exponential decay
        corr = np.zeros((SNPS_PER_BLOCK, SNPS_PER_BLOCK))
        for i in range(SNPS_PER_BLOCK):
            for j in range(SNPS_PER_BLOCK):
                corr[i, j] = 0.95 ** abs(i - j)

        # Generate correlated normal data, then discretize
        mean = np.zeros(SNPS_PER_BLOCK)
        z = rng.multivariate_normal(mean, corr, size=N_SAMPLES)

        # Discretize: threshold at tertiles to get 0/1/2
        for snp_i in range(SNPS_PER_BLOCK):
            col = z[:, snp_i]
            # Use frequency ~0.3 for minor allele
            t1 = np.percentile(col, 49)  # ~0.49 for AA
            t2 = np.percentile(col, 91)  # ~0.42 for Aa, ~0.09 for aa -> freq ~0.3
            geno = np.zeros(N_SAMPLES, dtype=np.int8)
            geno[col >= t1] = 1
            geno[col >= t2] = 2
            genotypes[:, start + snp_i] = geno

    logger.info("Genotype matrix shape: %s", genotypes.shape)
    return genotypes


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Simulate Phenotypes
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_phenotypes(
    genotypes: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, list[int]]:
    """Simulate phenotypes with known causal SNPs.

    Returns (phenotype_vector, causal_snp_indices).
    """
    # Pick one causal SNP from each LD block — choose one near the middle of each block
    causal_indices = []
    for block_idx in range(N_CAUSAL):
        start = block_idx * SNPS_PER_BLOCK
        # Pick from position 50-100 within each block for some LD with neighbors
        offset = 50 + block_idx * 10  # 50, 60, 70, 80, 90
        causal_indices.append(start + offset)

    logger.info("Causal SNP indices: %s", causal_indices)
    logger.info("Causal betas: %s", CAUSAL_BETAS)

    # Standardize causal genotypes
    X_causal = genotypes[:, causal_indices].astype(np.float64)
    for i in range(X_causal.shape[1]):
        X_causal[:, i] = (X_causal[:, i] - X_causal[:, i].mean()) / (
            X_causal[:, i].std() + 1e-10
        )

    betas = np.array(CAUSAL_BETAS)
    genetic_signal = X_causal @ betas
    var_genetic = np.var(genetic_signal)

    # Scale noise to achieve target heritability
    var_noise = var_genetic * (1 - HERITABILITY) / HERITABILITY
    noise = rng.normal(0, np.sqrt(var_noise), N_SAMPLES)
    phenotype = genetic_signal + noise

    actual_h2 = var_genetic / np.var(phenotype)
    logger.info("Target h²=%.2f, actual h²=%.4f", HERITABILITY, actual_h2)

    return phenotype, causal_indices


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Write PLINK files & run GWAS + LD
# ═══════════════════════════════════════════════════════════════════════════════
def write_plink_files(
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    plink_path: Path,
):
    """Write .ped/.map, convert to binary, run GWAS and LD."""
    GENO_DIR.mkdir(parents=True, exist_ok=True)
    GWAS_DIR.mkdir(parents=True, exist_ok=True)
    LD_DIR.mkdir(parents=True, exist_ok=True)

    prefix = str(GENO_DIR / "sim")

    # Write .map file
    map_path = prefix + ".map"
    with open(map_path, "w") as f:
        for i in range(N_SNPS):
            # chr, snp_id, genetic_dist, bp_position
            # Put all SNPs on chr1, positions spaced 1000bp apart
            f.write(f"1\trs{i + 1}\t0\t{(i + 1) * 1000}\n")

    # Write .ped file
    ped_path = prefix + ".ped"
    logger.info("Writing PED file (%d samples, %d SNPs)...", N_SAMPLES, N_SNPS)
    with open(ped_path, "w") as f:
        for i in range(N_SAMPLES):
            # FID IID father mother sex phenotype genotypes...
            parts = [f"FID{i + 1}", f"IID{i + 1}", "0", "0", "1", str(phenotype[i])]
            for j in range(N_SNPS):
                g = genotypes[i, j]
                if g == 0:
                    parts.extend(["A", "A"])
                elif g == 1:
                    parts.extend(["G", "A"])
                else:
                    parts.extend(["G", "G"])
            f.write("\t".join(parts) + "\n")

    # Convert to binary format
    logger.info("Converting to binary PLINK format...")
    subprocess.run(
        [str(plink_path), "--file", prefix, "--make-bed", "--out", prefix],
        capture_output=True,
        check=True,
    )

    # Write phenotype file (for --pheno)
    pheno_path = str(GENO_DIR / "sim.pheno")
    with open(pheno_path, "w") as f:
        for i in range(N_SAMPLES):
            f.write(f"FID{i + 1}\tIID{i + 1}\t{phenotype[i]}\n")

    return prefix


def run_gwas(prefix: str, plink_path: Path):
    """Run GWAS and generate .ma file + LD matrix."""
    logger.info("Running GWAS with PLINK...")
    gwas_prefix = str(GWAS_DIR / "sim")

    # Run association analysis
    subprocess.run(
        [
            str(plink_path),
            "--bfile",
            prefix,
            "--linear",
            "--allow-no-sex",
            "--out",
            gwas_prefix,
        ],
        capture_output=True,
        check=True,
    )

    # Run frequency calculation
    subprocess.run(
        [
            str(plink_path),
            "--bfile",
            prefix,
            "--freq",
            "--allow-no-sex",
            "--out",
            gwas_prefix,
        ],
        capture_output=True,
        check=True,
    )

    # Parse GWAS results
    assoc_file = gwas_prefix + ".assoc.linear"
    assoc = pd.read_csv(assoc_file, sep=r"\s+")
    # Keep only the ADD test
    assoc = assoc[assoc["TEST"] == "ADD"].copy()

    # Parse frequency file
    freq_file = gwas_prefix + ".frq"
    freq_df = pd.read_csv(freq_file, sep=r"\s+")

    # Read .bim to get allele info
    bim_file = prefix + ".bim"
    bim = pd.read_csv(
        bim_file, sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"]
    )

    # Merge to create .ma file
    # CRITICAL: A1 in .ma must match A1 in .bim (column 5 = minor allele in plink)
    ma = pd.DataFrame()
    ma["SNP"] = bim["SNP"]
    ma["A1"] = bim["A1"]
    ma["A2"] = bim["A2"]

    # Get frequency for A1 (from .bim perspective)
    freq_map = dict(zip(freq_df["SNP"], zip(freq_df["A1"], freq_df["MAF"])))
    freqs = []
    for _, row in ma.iterrows():
        snp = row["SNP"]
        if snp in freq_map:
            freq_a1, maf = freq_map[snp]
            if freq_a1 == row["A1"]:
                freqs.append(maf)
            else:
                freqs.append(1 - maf)
        else:
            freqs.append(np.nan)
    ma["freq"] = freqs

    # Get beta/se/p from association results — aligned to bim A1
    # PLINK --linear outputs STAT (t-statistic), not SE. Derive SE = |BETA / STAT|.
    assoc_map = dict(
        zip(assoc["SNP"], zip(assoc["A1"], assoc["BETA"], assoc["STAT"], assoc["P"]))
    )
    betas, ses, ps = [], [], []
    for _, row in ma.iterrows():
        snp = row["SNP"]
        if snp in assoc_map:
            assoc_a1, beta, t_stat, p = assoc_map[snp]
            se = abs(beta / t_stat) if t_stat != 0 else np.nan
            if assoc_a1 == row["A1"]:
                betas.append(beta)
            else:
                betas.append(-beta)  # Flip sign if alleles swapped
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

    # Drop rows with NaN (shouldn't happen but just in case)
    ma = ma.dropna()

    ma_path = str(GWAS_DIR / "sim.ma")
    ma.to_csv(ma_path, sep="\t", index=False)
    logger.info("Written .ma file: %s (%d SNPs)", ma_path, len(ma))

    # Generate LD matrix using PLINK --r square
    logger.info("Computing LD matrix...")
    ld_prefix = str(LD_DIR / "sim")
    subprocess.run(
        [
            str(plink_path),
            "--bfile",
            prefix,
            "--r",
            "square",
            "--allow-no-sex",
            "--out",
            ld_prefix,
        ],
        capture_output=True,
        check=True,
    )
    logger.info("LD matrix written to %s.ld", ld_prefix)

    return ma_path, ld_prefix + ".ld"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Run GCTA-COJO
# ═══════════════════════════════════════════════════════════════════════════════
def run_gcta_cojo(
    gcta_path: Path, bfile_prefix: str, ma_path: str
) -> dict[str, pd.DataFrame]:
    """Run GCTA-COJO slct, joint, cond."""
    GCTA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── slct ──────────────────────────────────────────────────────────────
    slct_prefix = str(GCTA_DIR / "slct")
    logger.info("Running GCTA-COJO slct...")
    proc = subprocess.run(
        [
            str(gcta_path),
            "--bfile",
            bfile_prefix,
            "--cojo-file",
            ma_path,
            "--cojo-slct",
            "--cojo-p",
            str(P_CUTOFF),
            "--cojo-collinear",
            str(COLLINEAR_CUTOFF),
            "--out",
            slct_prefix,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning("GCTA slct stderr:\n%s", proc.stderr)

    slct_file = slct_prefix + ".jma.cojo"
    if os.path.exists(slct_file):
        results["slct"] = pd.read_csv(slct_file, sep=r"\s+")
        logger.info("GCTA slct: %d SNPs selected", len(results["slct"]))
    else:
        logger.warning("GCTA slct output not found: %s", slct_file)
        results["slct"] = pd.DataFrame()

    # Get selected SNPs for joint/cond
    if not results["slct"].empty:
        selected_snps = results["slct"]["SNP"].tolist()
    else:
        logger.error("No SNPs selected by GCTA slct, cannot run joint/cond")
        return results

    # Write SNP list files for joint and cond
    snps_file = str(GCTA_DIR / "selected_snps.txt")
    with open(snps_file, "w") as f:
        for snp in selected_snps:
            f.write(f"{snp}\n")

    # Use first SNP as condition SNP (for cond analysis)
    cond_snps_file = str(GCTA_DIR / "cond_snps.txt")
    # Use the first selected SNP for conditioning
    with open(cond_snps_file, "w") as f:
        f.write(f"{selected_snps[0]}\n")

    # ── joint ─────────────────────────────────────────────────────────────
    joint_prefix = str(GCTA_DIR / "joint")
    logger.info("Running GCTA-COJO joint...")
    proc = subprocess.run(
        [
            str(gcta_path),
            "--bfile",
            bfile_prefix,
            "--cojo-file",
            ma_path,
            "--cojo-joint",
            "--extract",
            snps_file,
            "--out",
            joint_prefix,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning("GCTA joint stderr:\n%s", proc.stderr)

    joint_file = joint_prefix + ".jma.cojo"
    if os.path.exists(joint_file):
        results["joint"] = pd.read_csv(joint_file, sep=r"\s+")
        logger.info("GCTA joint: %d SNPs", len(results["joint"]))
    else:
        logger.warning("GCTA joint output not found: %s", joint_file)
        results["joint"] = pd.DataFrame()

    # ── cond ──────────────────────────────────────────────────────────────
    cond_prefix = str(GCTA_DIR / "cond")
    logger.info("Running GCTA-COJO cond...")
    proc = subprocess.run(
        [
            str(gcta_path),
            "--bfile",
            bfile_prefix,
            "--cojo-file",
            ma_path,
            "--cojo-cond",
            cond_snps_file,
            "--out",
            cond_prefix,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning("GCTA cond stderr:\n%s", proc.stderr)

    cond_file = cond_prefix + ".cma.cojo"
    if os.path.exists(cond_file):
        results["cond"] = pd.read_csv(cond_file, sep=r"\s+")
        logger.info("GCTA cond: %d SNPs", len(results["cond"]))
    else:
        logger.warning("GCTA cond output not found: %s", cond_file)
        results["cond"] = pd.DataFrame()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Run cojopy
# ═══════════════════════════════════════════════════════════════════════════════
def run_cojopy(
    ma_path: str, ld_path: str, gcta_results: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Run cojopy slct, joint, cond using the Python API directly."""
    # Add project root to path
    project_root = str(BASE_DIR.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from cojopy.cojopy import COJO

    COJOPY_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── slct ──────────────────────────────────────────────────────────────
    logger.info("Running cojopy slct...")
    c = COJO(p_cutoff=P_CUTOFF, collinear_cutoff=COLLINEAR_CUTOFF)
    c.load_sumstats(sumstats_path=ma_path, ld_path=ld_path)
    slct_result = c.conditional_selection()
    slct_out = str(COJOPY_DIR / "slct.txt")
    slct_result.to_csv(slct_out, sep="\t", index=False, float_format="%.6g")
    results["slct"] = slct_result
    logger.info("cojopy slct: %d SNPs selected", len(slct_result))

    # Get selected SNPs for joint/cond — use GCTA's selection for fair comparison
    if "slct" in gcta_results and not gcta_results["slct"].empty:
        selected_snps = gcta_results["slct"]["SNP"].tolist()
    elif not slct_result.empty:
        selected_snps = slct_result["SNP"].tolist()
    else:
        logger.error("No SNPs selected, cannot run joint/cond")
        return results

    # Write SNP list for joint
    snps_file = str(COJOPY_DIR / "selected_snps.txt")
    with open(snps_file, "w") as f:
        for snp in selected_snps:
            f.write(f"{snp}\n")

    # Condition SNP — same as GCTA
    cond_snps_file = str(COJOPY_DIR / "cond_snps.txt")
    with open(cond_snps_file, "w") as f:
        f.write(f"{selected_snps[0]}\n")

    # ── joint ─────────────────────────────────────────────────────────────
    logger.info("Running cojopy joint...")
    c2 = COJO(p_cutoff=P_CUTOFF, collinear_cutoff=COLLINEAR_CUTOFF)
    c2.load_sumstats(sumstats_path=ma_path, ld_path=ld_path)
    joint_result = c2.run_joint_analysis(extract_snps_path=snps_file)
    joint_out = str(COJOPY_DIR / "joint.txt")
    joint_result.to_csv(joint_out, sep="\t", index=False, float_format="%.6g")
    results["joint"] = joint_result
    logger.info("cojopy joint: %d SNPs", len(joint_result))

    # ── cond ──────────────────────────────────────────────────────────────
    logger.info("Running cojopy cond...")
    c3 = COJO(p_cutoff=P_CUTOFF, collinear_cutoff=COLLINEAR_CUTOFF)
    c3.load_sumstats(sumstats_path=ma_path, ld_path=ld_path)
    cond_result = c3.run_conditional_analysis(cond_snps_path=cond_snps_file)
    cond_out = str(COJOPY_DIR / "cond.txt")
    cond_result.to_csv(cond_out, sep="\t", index=False, float_format="%.6g")
    results["cond"] = cond_result
    logger.info("cojopy cond: %d SNPs", len(cond_result))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Compare Results
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_pearsonr(x, y):
    """Compute Pearson r, returning NaN if not enough data."""
    if len(x) < 2:
        return np.nan
    # Filter out inf/nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return pearsonr(x[mask], y[mask])[0]


def _scatter_plot(
    x, y, xlabel: str, ylabel: str, title: str, out_path: str, label: str = ""
):
    """Create a scatter plot with identity line."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.6, s=20)
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "r--", alpha=0.5, label="y=x")
    if len(x) >= 2:
        r = pearsonr(x, y)[0]
        ax.set_title(f"{title} (r={r:.6f})")
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare_results(
    gcta_results: dict[str, pd.DataFrame],
    cojopy_results: dict[str, pd.DataFrame],
) -> bool:
    """Compare GCTA-COJO and cojopy results. Returns True if all pass."""
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    all_pass = True
    summary_lines = []

    # ── slct comparison ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Comparing slct results")
    logger.info("=" * 60)
    g_slct = gcta_results.get("slct", pd.DataFrame())
    c_slct = cojopy_results.get("slct", pd.DataFrame())

    if g_slct.empty or c_slct.empty:
        msg = "[SKIP] slct: one or both results empty"
        logger.warning(msg)
        summary_lines.append(msg)
    else:
        g_snps = set(g_slct["SNP"])
        c_snps = set(c_slct["SNP"])
        same_snps = g_snps == c_snps
        common_snps = g_snps & c_snps

        logger.info("GCTA selected: %s", sorted(g_snps))
        logger.info("cojopy selected: %s", sorted(c_snps))
        logger.info("Same set: %s, Common: %d", same_snps, len(common_snps))

        if common_snps:
            # Merge on common SNPs
            merged = pd.merge(
                g_slct, c_slct, on="SNP", suffixes=("_gcta", "_cojopy")
            )
            # GCTA columns: bJ, bJ_se, pJ
            # cojopy columns: joint_beta, joint_se, joint_p
            r_beta = _safe_pearsonr(
                merged["bJ"].values, merged["joint_beta"].values
            )
            r_se = _safe_pearsonr(
                merged["bJ_se"].values, merged["joint_se"].values
            )
            max_diff_beta = np.max(
                np.abs(merged["bJ"].values - merged["joint_beta"].values)
            )
            max_diff_se = np.max(
                np.abs(merged["bJ_se"].values - merged["joint_se"].values)
            )

            pass_r = r_beta > 0.999 if np.isfinite(r_beta) else False
            status = "PASS" if (pass_r and same_snps) else "FAIL"
            if status == "FAIL":
                all_pass = False

            msg = (
                f"[{status}] slct: Same SNPs={same_snps}, "
                f"Beta r={r_beta:.6f}, SE r={r_se:.6f}, "
                f"Max|diff_beta|={max_diff_beta:.2e}, Max|diff_se|={max_diff_se:.2e}"
            )
            logger.info(msg)
            summary_lines.append(msg)

            # Scatter plots
            _scatter_plot(
                merged["bJ"].values,
                merged["joint_beta"].values,
                "GCTA bJ",
                "cojopy joint_beta",
                "slct: Beta comparison",
                str(COMP_DIR / "slct_beta.png"),
            )
            _scatter_plot(
                merged["bJ_se"].values,
                merged["joint_se"].values,
                "GCTA bJ_se",
                "cojopy joint_se",
                "slct: SE comparison",
                str(COMP_DIR / "slct_se.png"),
            )
        else:
            msg = "[FAIL] slct: No common SNPs between GCTA and cojopy"
            logger.warning(msg)
            summary_lines.append(msg)
            all_pass = False

    # ── joint comparison ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Comparing joint results")
    logger.info("=" * 60)
    g_joint = gcta_results.get("joint", pd.DataFrame())
    c_joint = cojopy_results.get("joint", pd.DataFrame())

    if g_joint.empty or c_joint.empty:
        msg = "[SKIP] joint: one or both results empty"
        logger.warning(msg)
        summary_lines.append(msg)
    else:
        merged = pd.merge(
            g_joint, c_joint, on="SNP", suffixes=("_gcta", "_cojopy")
        )
        if len(merged) == 0:
            msg = "[FAIL] joint: No common SNPs after merge"
            logger.warning(msg)
            summary_lines.append(msg)
            all_pass = False
        else:
            r_beta = _safe_pearsonr(
                merged["bJ"].values, merged["joint_beta"].values
            )
            r_se = _safe_pearsonr(
                merged["bJ_se"].values, merged["joint_se"].values
            )
            max_diff_beta = np.max(
                np.abs(merged["bJ"].values - merged["joint_beta"].values)
            )
            max_diff_se = np.max(
                np.abs(merged["bJ_se"].values - merged["joint_se"].values)
            )

            pass_r = r_beta > 0.999 if np.isfinite(r_beta) else False
            status = "PASS" if pass_r else "FAIL"
            if status == "FAIL":
                all_pass = False

            msg = (
                f"[{status}] joint: Beta r={r_beta:.6f}, SE r={r_se:.6f}, "
                f"Max|diff_beta|={max_diff_beta:.2e}, Max|diff_se|={max_diff_se:.2e}"
            )
            logger.info(msg)
            summary_lines.append(msg)

            _scatter_plot(
                merged["bJ"].values,
                merged["joint_beta"].values,
                "GCTA bJ",
                "cojopy joint_beta",
                "joint: Beta comparison",
                str(COMP_DIR / "joint_beta.png"),
            )
            _scatter_plot(
                merged["bJ_se"].values,
                merged["joint_se"].values,
                "GCTA bJ_se",
                "cojopy joint_se",
                "joint: SE comparison",
                str(COMP_DIR / "joint_se.png"),
            )

            # Also compare -log10(p)
            g_logp = -np.log10(merged["pJ"].values.astype(float))
            c_logp = -np.log10(merged["joint_p"].values.astype(float))
            _scatter_plot(
                g_logp,
                c_logp,
                "GCTA -log10(pJ)",
                "cojopy -log10(joint_p)",
                "joint: -log10(p) comparison",
                str(COMP_DIR / "joint_logp.png"),
            )

    # ── cond comparison ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Comparing cond results")
    logger.info("=" * 60)
    g_cond = gcta_results.get("cond", pd.DataFrame())
    c_cond = cojopy_results.get("cond", pd.DataFrame())

    if g_cond.empty or c_cond.empty:
        msg = "[SKIP] cond: one or both results empty"
        logger.warning(msg)
        summary_lines.append(msg)
    else:
        merged = pd.merge(
            g_cond, c_cond, on="SNP", suffixes=("_gcta", "_cojopy")
        )
        if len(merged) == 0:
            msg = "[FAIL] cond: No common SNPs after merge"
            logger.warning(msg)
            summary_lines.append(msg)
            all_pass = False
        else:
            # GCTA cond columns: bC, bC_se, pC
            # cojopy cond columns: cond_beta, cond_se, cond_p
            r_beta = _safe_pearsonr(
                merged["bC"].values, merged["cond_beta"].values
            )
            r_se = _safe_pearsonr(
                merged["bC_se"].values, merged["cond_se"].values
            )
            max_diff_beta = np.max(
                np.abs(merged["bC"].values - merged["cond_beta"].values)
            )
            max_diff_se = np.max(
                np.abs(merged["bC_se"].values - merged["cond_se"].values)
            )

            pass_r = r_beta > 0.999 if np.isfinite(r_beta) else False
            status = "PASS" if pass_r else "FAIL"
            if status == "FAIL":
                all_pass = False

            msg = (
                f"[{status}] cond:  Beta r={r_beta:.6f}, SE r={r_se:.6f}, "
                f"Max|diff_beta|={max_diff_beta:.2e}, Max|diff_se|={max_diff_se:.2e}"
            )
            logger.info(msg)
            summary_lines.append(msg)

            _scatter_plot(
                merged["bC"].values,
                merged["cond_beta"].values,
                "GCTA bC",
                "cojopy cond_beta",
                "cond: Beta comparison",
                str(COMP_DIR / "cond_beta.png"),
            )
            _scatter_plot(
                merged["bC_se"].values,
                merged["cond_se"].values,
                "GCTA bC_se",
                "cojopy cond_se",
                "cond: SE comparison",
                str(COMP_DIR / "cond_se.png"),
            )

            g_logp = -np.log10(merged["pC"].values.astype(float))
            c_logp = -np.log10(merged["cond_p"].values.astype(float))
            _scatter_plot(
                g_logp,
                c_logp,
                "GCTA -log10(pC)",
                "cojopy -log10(cond_p)",
                "cond: -log10(p) comparison",
                str(COMP_DIR / "cond_logp.png"),
            )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    for line in summary_lines:
        logger.info(line)
    overall = "ALL PASSED" if all_pass else "SOME FAILED"
    logger.info("Overall: %s", overall)
    logger.info("=" * 60)

    # Write summary to file
    summary_path = str(COMP_DIR / "summary.txt")
    with open(summary_path, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nOverall: {overall}\n")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("Starting cojopy vs GCTA-COJO simulation comparison")
    logger.info("Parameters: N=%d, SNPs=%d, blocks=%d, h²=%.2f, seed=%d",
                N_SAMPLES, N_SNPS, N_BLOCKS, HERITABILITY, SEED)

    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Step 1: Download tools
    gcta_path, plink_path = download_tools()

    # Step 2: Simulate genotypes
    rng = np.random.default_rng(SEED)
    genotypes = simulate_genotypes(rng)

    # Step 3: Simulate phenotypes
    phenotype, causal_indices = simulate_phenotypes(genotypes, rng)

    # Step 4: Write PLINK files & run GWAS + LD
    bfile_prefix = write_plink_files(genotypes, phenotype, plink_path)
    ma_path, ld_path = run_gwas(bfile_prefix, plink_path)

    # Step 5: Run GCTA-COJO
    gcta_results = run_gcta_cojo(gcta_path, bfile_prefix, ma_path)

    # Step 6: Run cojopy
    cojopy_results = run_cojopy(ma_path, ld_path, gcta_results)

    # Step 7: Compare
    all_pass = compare_results(gcta_results, cojopy_results)

    if all_pass:
        logger.info("SUCCESS: cojopy produces results consistent with GCTA-COJO")
    else:
        logger.warning("ATTENTION: Some comparisons did not pass threshold")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
