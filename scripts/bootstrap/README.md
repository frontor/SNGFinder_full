# Bootstrap CI for the de novo fraction (20,000 resamples)

`denovo_fraction_bootstrap.py` reproduces the 20,000-resample bootstrap
confidence intervals for the proportion of predicted de novo genes among
high-confidence classified orphan genes.

## How to run

Prepare a two-column table (Gene, Probabilities) of the high-confidence
genes (predicted de novo and predicted old) from Supplementary Table S10,
then:

```bash
python denovo_fraction_bootstrap.py high_confidence.tsv --n-boot 20000
```

Alternatively feed the full S10 prediction table: with the `Class` column
(1 / 0) present, unclassified rows are dropped automatically.

## Verified reproduction (2026-11-11)

From the published Table S10 (standard model, 5,026 high-confidence genes,
4,018 predicted de novo = 79.94%):

```
95% CI (20000 bootstrap resamples): 78.83% - 81.02%   (seed 0/42)
                                      78.81% - 81.04%   (seed 0, 2026)
```

i.e. the reported 95% CI of 78.83–81.04% is reproduced within Monte-Carlo
noise of the resampling seed.
