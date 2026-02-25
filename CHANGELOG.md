# Changelog

## [0.1.6] (2026-02-25)

### Fixed

* Fixed NaN (blank) P-values in joint analysis caused by negative variance from ill-conditioned matrix.
* Fixed P-value underflow to `0.0` for extremely significant SNPs (clamped to `~2.2e-308`).
* Hardened conditional stats SE check with explicit `np.isfinite` guard.
* Added warning for negative or zero effective sample sizes.

## [0.1.5] (2026-02-25)

* Boosted test coverage from 75% to 99% (cojopy.py 100% line coverage, 26 new tests).
* Fixed preview workflow: replaced tomllib with regex for Python 3.9 compatibility.
* Fixed lint and formatting issues in test files.

## [0.1.4] (2026-02-25)

* Migrated from Poetry to uv with hatchling build backend.
* Relaxed typer dependency version constraint.
* Fixed tox configuration for PEP 621 compatibility.

## [0.1.2] (2025-04-25)

* `load_sumstats` supports loading ld_freq from a pandas DataFrame.

## [0.1.1] (2025-04-25)

* `load_sumstats` supports loading sumstats, ld_matrix, and ld_freq from files or from pandas DataFrames.

## [0.1.0] (2025-03-21)

* Added `extract-snps` option for `joint` and `cond` commands.
* Added `ld-freq` option for `joint` and `cond` commands.
* Added `p-cutoff` option for `joint` and `cond` commands.
* Added `collinear-cutoff` option for `joint` and `cond` commands.
* Added `maf-cutoff` option for `joint` and `cond` commands.
* Added `diff-freq-cutoff` option for `joint` and `cond` commands.
* Wrote README.md.

## [0.0.4] (2025-03-20)

* Added tests for conditional selection and joint analysis.
* Fixed bug in joint statistics calculation.
* Added tests for conditional selection and joint analysis.

## [0.0.3] (2025-03-20)

* Fixed bug in joint statistics calculation.
* Added tests for conditional selection and joint analysis.

## [0.0.2] (2025-03-09)

* Conditional selection with LD matrix.

## [0.0.1] (2025-03-06)

* First release on PyPI.
