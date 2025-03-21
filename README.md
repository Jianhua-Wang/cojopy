# cojopy


[![pypi](https://img.shields.io/pypi/v/cojopy.svg)](https://pypi.org/project/cojopy/)
[![python](https://img.shields.io/pypi/pyversions/cojopy.svg)](https://pypi.org/project/cojopy/)
[![Build Status](https://github.com/Jianhua-Wang/cojopy/actions/workflows/dev.yml/badge.svg)](https://github.com/Jianhua-Wang/cojopy/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/Jianhua-Wang/cojopy/branch/main/graphs/badge.svg)](https://codecov.io/github/Jianhua-Wang/cojopy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



Conditional Analysis with LD Matrix

Get the same results as [GCTA COJO](https://yanglab.westlake.edu.cn/software/gcta/#COJO), but with LD matrix.

## Installation

```bash
pip install cojopy
```

## Usage

### Input files
#### Summary statistics file

The summary statistics file should have the following columns (same as the cojo file of GCTA):

- SNP: SNP ID
- A1: Effect allele
- A2: Other allele
- b: Effect size
- se: Standard error
- p: P-value
- freq: Minor allele frequency
- N: Sample size


#### LD matrix file
- A tab-delimited LD matrix, same as the output of `plink --r square`.
> [!Note]
> The allele order of the summary statistics file should be the same as the allele order of the LD matrix file.


### Stepwise model selection of independent associated SNPs (same as `gcta --cojo-slct`)

```bash
cojo slct \
--sumstats ./exampledata/sim_gwas.ma \
--ld-matrix ./exampledata/sim_r.ld \
--output ./exampledata/slct.txt
```

### Fit all the included SNPs to estimate their joint effects without model selection (same as `gcta --cojo-joint`)

```bash
cojo joint \
--sumstats ./exampledata/sim_gwas.ma \
--ld-matrix ./exampledata/sim_r.ld \
--extract-snps ./exampledata/extract_snps.txt \
--output ./exampledata/joint.txt
```

### Perform association analysis of the included SNPs conditional on the given list of SNPs (same as `gcta --cojo-cond`)

```bash
cojo cond \
--sumstats ./exampledata/sim_gwas.ma \
--ld-matrix ./exampledata/sim_r.ld \
--cond-snps ./exampledata/cond_snps.txt \
--output ./exampledata/cond.txt
```

### Parameters

- `cond-snps`: For `cond` command, path to the file containing the SNPs to condition on, one SNP per line.
- `extract-snps`: For `joint` and `cond` commands, path to the file containing the SNPs to extract, one SNP per line.
- `ld-freq`: Path to the LD frequency file, optional, use freq in sumstats if not provided.
- `p-cutoff`: P-value cutoff, default is 5e-8.
- `collinear-cutoff`: Collinearity cutoff, default is 0.9.
- `maf-cutoff`: Minor allele frequency cutoff, default is 0.01.
- `diff-freq-cutoff`: Difference in minor allele frequency cutoff, default is 0.2, won't take effect unless `ld-freq` is provided.
