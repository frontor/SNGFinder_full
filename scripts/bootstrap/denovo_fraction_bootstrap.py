#!/usr/bin/env python
"""Bootstrap confidence intervals for the de novo fraction among high-confidence
orphan genes, as reported in the manuscript (20,000 resamples).

Usage:
    python denovo_fraction_bootstrap.py <prediction_table.tsv> [--n-boot 20000]

Input: a two-column (Gene, Probabilities) or three-column
(Gene, Probabilities, Class) table with per-gene probabilities of the
high-confidence classified orphan genes, or a full prediction table with a
`Class` column where `1` = predicted de novo and `0` = predicted old
(high-confidence genes only; unclassified rows must be excluded or will be
dropped automatically).

Example (reproduces the standard-set CI from the manuscript's Table S10):
    python denovo_fraction_bootstrap.py table_S10_high_confidence.tsv

Output: point estimate and percentile 95% CI of the de novo fraction.
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", help="prediction table (Gene<TAB>Probabilities[<TAB>Class])")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=888)
    ap.add_argument("--low", type=float, default=0.3758, help="t_low (standard model)")
    ap.add_argument("--high", type=float, default=0.9214, help="t_high (standard model)")
    args = ap.parse_args()

    df = pd.read_csv(args.table, sep="\t")
    if "Class" in df.columns:
        df = df[df["Class"].isin([0, 1])]
        labels = df["Class"].astype(float).values
    else:
        p = df["Probabilities"].astype(float).values
        labels = np.where(p >= args.high, 1.0, np.where(p <= args.low, 0.0, np.nan))
        labels = labels[~np.isnan(labels)]

    n = len(labels)
    k = int(labels.sum())
    print("high-confidence n = %d, predicted de novo = %d, fraction = %.4f" % (n, k, k / n))

    rng = np.random.default_rng(args.seed)
    fractions = np.empty(args.n_boot)
    for i in range(args.n_boot):
        draw = labels[rng.integers(0, n, size=n)]
        fractions[i] = draw.mean()

    lo, hi = np.percentile(fractions, [2.5, 97.5])
    print("95%% CI (%d bootstrap resamples): %.2f%% - %.2f%%"
          % (args.n_boot, 100 * lo, 100 * hi))


if __name__ == "__main__":
    main()
