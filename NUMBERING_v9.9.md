# Supplementary numbering in this repository (v9.9 sync)

The manuscript (NAR submission v9.9, 2026-11-23) numbers its
supplementary tables **S1-S22** and supplementary figures **S1-S25**.
This repository was last updated at the v9.0 stage; its scripts, data
filenames and some shorthand labels still carry the v9.0 numbering.

## What was synced

- Anchored prose references ("Table S23", "Supplementary Tables S5, S6
  and S8", "Fig. S6", ...) in READMEs, docstrings and comments have been
  rewritten to the v9.9 numbering using the mapping tables below.
- **Not renamed**: file names, directory names, output artifact names,
  variable names and Excel sheet names. They are part of the executable
  reproduction chain and renaming them would break it. Examples:
  `scripts/table_S4/` (v9.9 Table S2), `scripts/rate_window_S23.py`
  (v9.9 Table S9), `classifier_data/ablation_orphan_calls/S13_unified.tsv`
  (v9.9 Table S5), `verify_vs_S12.tsv` (v9.9 Table S11).
- The published Zenodo data bundle 10.5281/zenodo.24130664 keeps its
  original title "... Table S4 reproduction data bundle ...": published
  records cannot be edited. In v9.9 that table is Supplementary Table S2.

## v9.0 -> v9.9 : supplementary TABLES

| v9.0 | v9.9 | v9.0 | v9.9 | v9.0 | v9.9 |
|---|---|---|---|---|---|
| S1 | S6 | S2 | S8 | S3 | S7 |
| S4 | S2 | S5 | S3 | S6 | S3 |
| S7 | S5 | S8 | S10 | S9 | S13 |
| S10 | S14 | S11 | S4 | S12 | S11 |
| S13 | S5 | S14 | S16 | S15a | S17 |
| S15b | S17 | S16a | S18 | S16b | S18 |
| S17 | S19 | S18 | S20 | S19 | S20 |
| S20 | S21 | S21 | S22 | S22 | S22 |
| S23 | S9 | S24 | S1 | S25 | S1 |
| S26 | S1 | S27 | S12 |  |  |

## v9.0 -> v9.9 : supplementary FIGURES

| v9.0 | v9.9 | v9.0 | v9.9 | v9.0 | v9.9 |
|---|---|---|---|---|---|
| S1 | S2 | S2 | S3 | S3 | S4 |
| S4 | S5 | S5 | S6 | S6 | S7 |
| S7 | S8 | S8 | S9 | S9 | S10 |
| S10 | S11 | S11 | S12 | S12 | S13 |
| S13 | S14 | S14 | S15 | S15 | S16 |
| S16 | S17 | S17 | S18 | S18 | S19 |
| S19 | S20 | S20 | S1 | S21 | S21 |
| S22 | S22 | S23 | S23 | S24 | S24 |
| S25 | S25 |  |  |  |  |

Generated programmatically from `renumber_map_final.json` (v9.6
renumbering) and `v99_plan.json` (v9.9 merge + renumbering).
