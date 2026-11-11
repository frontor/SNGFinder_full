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
