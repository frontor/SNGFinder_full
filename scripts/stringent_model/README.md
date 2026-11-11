# Stringent-positive-set retraining (175-gene / 167-gene subset)

Reproduces the stringent-positive-set classifier analysis reported in the
manuscript: retraining the XGBoost classifier on the "set 1" prior de novo
genes (absence in ≥ 2 outgroup species) instead of the full 324-gene prior
set, and re-applying it to the 6,149 orphan genes.

## Contents

| Item | Description |
|---|---|
| `denovo_175.txt` | the 175 "set 1" prior de novo genes |
| `deal.py` | filters Training/Test datasets to this positive set (negatives unchanged) |

## How to run

The 175-gene list intersects the 324 published positives (Supplementary
Tables S5/S6) in 167 genes, which become the stringent training/test
positives (114 + 53).

```bash
# 1. rebuild the stringent datasets from the published S5/S6 content
#    (Training_Dataset.csv / Test_Dataset.csv; last column = Categories)
python deal.py denovo_175.txt Training_Dataset.csv Training_Dataset_denovo175.csv
python deal.py denovo_175.txt Test_Dataset.csv Test_Dataset_denovo175.csv

# 2. retrain (repository root)
python ModelTrain_v3.py XGBoost Training_Dataset_denovo175.csv \
    Test_Dataset_denovo175.csv Orphan_Dataset.csv \
    A_ppv0.95_npv0.995_30_200 0.2 A 0.95 0.995 30 200 revision_outputs_stringent
```

`Orphan_Dataset.csv` is reconstructed from Supplementary Table S8.

## Verified reproduction (2026-11-11)

- 10-fold CV AP = 0.8973 ± 0.0463; independent test AP = 0.9229
- abstention thresholds t_low = 0.3376, t_high = 0.9769
- applied to the 6,149 orphans: 2,877 predicted de novo among 4,241
  high-confidence calls = 67.84%
- 20,000-resample bootstrap CI of the de novo fraction: 66.4% – 69.2%
  (seed-dependent at the second decimal; manuscript reports 66.45–69.23)
