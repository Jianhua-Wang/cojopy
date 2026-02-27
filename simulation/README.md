# cojopy vs GCTA-COJO Simulation Comparison

This simulation generates synthetic genotype/phenotype data and runs both GCTA-COJO and cojopy to verify they produce equivalent results.

## Quick Start

```bash
cd simulation
python run_simulation.py
```

## What It Does

1. **Downloads tools**: Automatically fetches GCTA (1.94.1) and PLINK (1.9) to `bin/` if not in PATH
2. **Simulates genotypes**: 2000 samples, 1000 SNPs in 5 LD blocks (exponential decay correlation)
3. **Simulates phenotypes**: 5 causal SNPs (one per block), h²=0.3
4. **Runs GWAS**: PLINK `--linear` + frequency + LD matrix (`--r square`)
5. **Runs GCTA-COJO**: slct, joint, cond
6. **Runs cojopy**: slct, joint, cond (same parameters)
7. **Compares**: Pearson correlation, max absolute difference, scatter plots

## Output Structure

```
output/
├── genotype/    # .bed/.bim/.fam files
├── gwas/        # GWAS results + .ma file
├── ld/          # LD matrix
├── gcta/        # GCTA-COJO results
├── cojopy/      # cojopy results
└── comparison/  # Scatter plots + summary.txt
```

## Pass Criteria

- **slct**: Same SNPs selected, Beta/SE Pearson r > 0.999
- **joint**: Beta/SE Pearson r > 0.999
- **cond**: Beta/SE Pearson r > 0.999

## Dependencies

- numpy, pandas, scipy (from cojopy)
- matplotlib (for plots)
- Internet access (first run, to download GCTA/PLINK)

## Notes

- `bin/` and `output/` are gitignored
- The joint and cond comparisons use the **same SNP sets** from GCTA's slct to ensure a fair comparison
- Allele alignment is handled carefully: .ma A1 matches .bim A1
