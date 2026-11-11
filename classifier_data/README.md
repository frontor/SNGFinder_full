# Classifier input data

`ModelTrain_v3.py` expects three tab-separated files with a header row and the
gene identifier in the first column. This directory contains the exact files
used for the reported classifier runs, so that
`Supplementary Tables S5, S6 and S8` can be regenerated from the deposited
inputs without rebuilding the feature matrix from raw sequencing data.

| File | Rows | Published counterpart |
|---|---|---|
| `Training_Dataset.csv` | 4,067 (3,849 old + 216 prior de novo, 2 rows removed later for missing gtAI) | Supplementary Table S5 |
| `Test_Dataset.csv` | 2,034 (1,926 old + 108 prior de novo) | Supplementary Table S6 |
| `Orphan_Dataset.csv` | 6,149 orphan genes (no `Categories` column) | Supplementary Table S8 |

## Column names

Thirty of the 61 columns carry pipeline names that differ from the names used
in the published supplementary tables (the three `Average*Length` columns and
the 27 chromatin columns). `COLUMN_MAP.tsv` gives the one-to-one mapping;
renaming with it reproduces the published column headers exactly, and the
values are identical (verified on a 300-gene sample across all 59 features and
the class label).

## Running the classifier

```bash
python ModelTrain_v3.py XGBoost classifier_data/Training_Dataset.csv \
    classifier_data/Test_Dataset.csv classifier_data/Orphan_Dataset.csv \
    A_ppv0.95_npv0.995_30_200 0.10 A 0.95 0.995 30 200 out_std
```

The ablation table (Supplementary Table S12) is produced by the same script in
the same run (`out_std_ablation_summary.tsv`); the per-model orphan calls and
the unified ablation table (Supplementary Table S13) are produced by
`scripts/classifier_ablation/ablation_orphan_unify.py`. The exact outputs of the reported run (nine `<Model>_orphan_predictions.tsv` files with the per-gene binary call behind each count in Table S13 column G, and the summary `S13_unified.tsv`) are deposited under `classifier_data/ablation_orphan_calls/`; the n column of Table S13 can be re-derived from them by counting `BinaryPrediction == 1`.

## Notes on the deposited code

`ModelTrain_v3.py` and the scripts under `scripts/` are parameterized
clean-ups of the original research scripts: the analysis logic is unchanged,
but file paths and run parameters were moved to command-line arguments so the
reported runs can be reproduced from this repository alone.

The bit-exact verification of the unified ablation run against Supplementary
Table S12 (`ablation_orphan_calls/verify_vs_S12.tsv`, max_abs_diff <= 1.2e-16
on every metric) covers eight of the nine ablation models. The unified run
emits the per-gene orphan calls of `Gene_architecture_only` but not its full
metric row, so that row of Table S12 was recomputed separately with
`scripts/classifier_ablation/ablation_row_full.py`; the result is deposited as
`ablation_orphan_calls/Gene_architecture_only_full_row.tsv`.
