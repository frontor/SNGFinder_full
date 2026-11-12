#!/usr/bin/env python3
"""Regenerate Supplementary Table S23 (origination-rate window sensitivity).

    python scripts/rate_window_S23.py

Every numerator below is a published supplementary-table value and every
denominator is a published divergence-time estimate, so the whole table is
recomputed from first principles and checked against the deposited values.

Numerators
----------
9068  Candidate_new_genes, O. sativa japonica            (Table S3)
6613  refined candidate-new-gene set = 4018 + 2244 duplicated + 351 CNV genes
5901  Candidate_new_genes, O. glaberrima (lowest of the six)  (Table S3)
4018  machine-learning-predicted de novo genes          (Table S10)
1247  Candidate_new_genes, D. melanogaster              (Table S3)

Denominators (Myr)
------------------
5.5   midpoint of the 5-6 Ma interval for the A-/B-/C-genome diversification
      (Zou et al. 2013, New Phytol 198:1155-1164)
6.76  AA-BB divergence (Stein et al. 2018, Nat Genet 50:285-296; PATHd8,
      conditional on a fixed 15-Myr Oryza crown-group calibration)
43    D. grimshawi-D. melanogaster split, the longest window consistent with
      the D. melanogaster orphan definition
"""
from math import floor

W_PRIMARY = 5.5      # Myr, primary window (Zou et al. 2013)
W_CONSERV = 6.76     # Myr, conservative window (Stein et al. 2018)
W_DROS = 43.0        # Myr, D. melanogaster comparison window

JAP = 9068           # Table S3, O. sativa japonica Candidate_new_genes
GLA = 5901           # Table S3, O. glaberrima Candidate_new_genes (lowest)
DENOVO = 4018        # Table S10, predicted de novo
DUP = 2244           # Table S3, duplicated component of the refined set
CNV = 351            # Table S3, CNV component of the refined set
REFINED = DENOVO + DUP + CNV      # 6613
DMEL = 1247          # Table S3, D. melanogaster Candidate_new_genes
PRIOR_51_5 = 51.5    # Zhang et al. 2019 previous estimate, genes per Myr


def rnd(x):
    """Round half up to the nearest integer, as in Table S23."""
    return int(floor(x + 0.5))


def rate(n, w):
    return n / w


rows = []
rows.append(('Candidate new genes, O. sativa japonica', JAP, 'genes'))
rows.append(('Candidate new genes after machine-learning refinement', REFINED, 'genes'))
rows.append(('Lowest total new-gene count among the six Oryza species (O. glaberrima)', GLA, 'genes'))
rows.append(('Origination rate, 5.5-Myr window (as reported)', rnd(rate(JAP, W_PRIMARY)), 'genes per Myr'))
rows.append(('Origination rate, 6.76-Myr window (longest window consistent with the orphan definition)',
             rnd(rate(JAP, W_CONSERV)), 'genes per Myr'))
rows.append(('Refined rate, 5.5-Myr window (as reported)', rnd(rate(REFINED, W_PRIMARY)), 'genes per Myr'))
rows.append(('Refined rate, 6.76-Myr window', rnd(rate(REFINED, W_CONSERV)), 'genes per Myr'))
rows.append(('Six-species rate range, 6.76-Myr window',
             '%d-%d' % (rnd(rate(GLA, W_CONSERV)), rnd(rate(JAP, W_CONSERV))), 'genes per Myr'))
rows.append(('Six-species rate range, 5.5-Myr window',
             '%d-%d' % (rnd(rate(GLA, W_PRIMARY)), rnd(rate(JAP, W_PRIMARY))), 'genes per Myr'))
rows.append(('Predicted de novo genes, O. sativa japonica', DENOVO, 'genes'))
rows.append(('De novo rate, 5.5-Myr window', rnd(rate(DENOVO, W_PRIMARY)), 'genes per Myr'))
rows.append(('De novo rate, 6.76-Myr window', rnd(rate(DENOVO, W_CONSERV)), 'genes per Myr'))
rows.append(('D. melanogaster candidate new genes (Table S3)', DMEL, 'genes'))
rows.append(('D. melanogaster comparison window (D. grimshawi outgroup)', W_DROS, 'Myr'))
rows.append(('D. melanogaster candidate new-gene rate (framework-consistent)', rnd(rate(DMEL, W_DROS)),
             'genes per Myr'))

print('%-72s %10s  %s' % ('Quantity', 'Value', 'Unit'))
for a, b, c in rows:
    val = format(b, ',') if isinstance(b, int) else str(b)
    print('%-72s %10s  %s' % (a, val, c))

# --- checks against the deposited table and the main text -------------------
assert REFINED == 6613
assert rnd(rate(JAP, W_PRIMARY)) == 1649          # main text: 1,073-1,649
assert rnd(rate(GLA, W_PRIMARY)) == 1073
assert rnd(rate(JAP, W_CONSERV)) == 1341          # main text: 873-1,341
assert rnd(rate(GLA, W_CONSERV)) == 873
assert rnd(rate(REFINED, W_PRIMARY)) == 1202
assert rnd(rate(REFINED, W_CONSERV)) == 978       # main text: ~978
assert rnd(rate(DENOVO, W_PRIMARY)) == 731        # main text: ~731
assert rnd(rate(DENOVO, W_CONSERV)) == 594
assert rnd(rate(DMEL, W_DROS)) == 29              # main text: 29 per Myr

# fold changes relative to the previous single-species estimate (51.5 per Myr)
lo, hi = rate(GLA, W_CONSERV), rate(JAP, W_PRIMARY)
print('\nFolds vs %.1f per Myr (Zhang et al. 2019): conservative %.1f-%.1f, primary %.1f-%.1f'
      % (PRIOR_51_5, rate(GLA, W_CONSERV) / PRIOR_51_5, rate(JAP, W_CONSERV) / PRIOR_51_5,
         rate(GLA, W_PRIMARY) / PRIOR_51_5, rate(JAP, W_PRIMARY) / PRIOR_51_5))
print('Folds vs D. melanogaster (29 per Myr): %.1f-%.1f' % (rate(GLA, W_PRIMARY) / 29, rate(JAP, W_PRIMARY) / 29))
print('Folds vs human (17 per Myr):             %.1f-%.1f' % (rate(GLA, W_PRIMARY) / 17, rate(JAP, W_PRIMARY) / 17))
print('\nAll Table S23 values reproduced.')
