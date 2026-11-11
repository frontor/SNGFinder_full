# -*- coding: utf-8 -*-
"""Reproduce the classifier ablation table (Supplementary Table S13) and the
per-model orphan calls behind it.

The main classifier script (ModelTrain_v3.py) writes the performance summary of
the ablation models (Supplementary Table S12) but stops there: it never applies
the ablated models to the orphan pool. This script fills that gap. It re-runs
every feature family through exactly the same code path
(ModelTrain_v3.run_original_style_subset_model: grid search, F1-optimal
threshold via TunedThresholdClassifierCV, 10-fold out-of-fold predictions),
refits each model on the full training set, predicts the orphan pool, and writes

  <outdir>/<Model>_orphan_predictions.tsv   GeneID, Probability, BinaryPrediction
  <outdir>/S13_unified.tsv                  one row per model (Table S13 columns)
  <outdir>/verify_vs_S12.tsv                per-model diff against the S12 artifact

Usage
-----
python ablation_orphan_unify.py \
    --train classifier_data/Training_Dataset.csv \
    --test  classifier_data/Test_Dataset.csv \
    --orphan classifier_data/Orphan_Dataset.csv \
    --s12   out_std/out_std_ablation_summary.tsv \
    --outdir ablation_orphan

Gene_architecture_only is the six-feature gene-architecture family
(GeneLength, IsoformNumber, ExonNumber, AverageExonLength, AverageIntronLength,
AverageCDSLength), i.e. `detect_feature_groups(...)["families_detailed"]
["Gene_architecture"]`; it is not part of the S12 run because S12 summarises the
seven feature-family models plus the matched and stringent-positive-set refits.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ModelTrain_v3 as M  # noqa: E402

UCMethod, PPV, NPV, MinPos, MinNeg, RQ = "A", 0.95, 0.995, 30, 200, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--orphan", required=True)
    ap.add_argument("--s12", default="", help="out_std_ablation_summary.tsv, for verification")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.train, sep="\t", header=0, index_col=0).dropna()
    X, y = df.drop("Categories", axis=1), df["Categories"]
    dt = pd.read_csv(args.test, sep="\t", header=0, index_col=0).dropna()
    X_test, y_test = dt.drop("Categories", axis=1), dt["Categories"]
    orp = pd.read_csv(args.orphan, sep="\t", header=0, index_col=0).dropna()

    fg = M.detect_feature_groups(X.columns)
    gene_arch = fg["families_detailed"]["Gene_architecture"]
    assert len(gene_arch) == 6, gene_arch

    feature_sets = OrderedDict()
    feature_sets["Full"] = list(X.columns)
    feature_sets["Simple_only"] = fg["simple"]
    feature_sets["Core_length_GC"] = [c for c in ["GeneLength", "ExonNumber", "AverageCDSLength",
                                                  "GCContent", "PurineContent"] if c in X.columns]
    feature_sets["Gene_architecture_only"] = list(gene_arch)
    feature_sets["Expression_only"] = fg["expression"]
    feature_sets["Chromatin_only"] = fg["chromatin"]
    feature_sets["Full_minus_expression"] = [c for c in X.columns if c not in set(fg["expression"])]
    feature_sets["Full_minus_chromatin"] = [c for c in X.columns if c not in set(fg["chromatin"])]
    feature_sets["Full_minus_simple"] = [c for c in X.columns if c not in set(fg["simple"])]

    ref = None
    if args.s12 and os.path.exists(args.s12):
        ref = pd.read_csv(args.s12, sep="\t").set_index("Model")

    rows, verify = [], []
    for name, cols in feature_sets.items():
        print("=" * 20, name, len(cols), flush=True)
        row = M.run_original_style_subset_model(
            "XGBoost", X[cols], y, X_test[cols], y_test,
            RQ, UCMethod, PPV, NPV, MinPos, MinNeg, n_jobs=args.jobs)

        if ref is not None and name in ref.index:
            r = ref.loc[name]
            diffs = {k: abs(float(row[k]) - float(r[k])) for k in
                     ["Best_CV_AP_grid", "Threshold", "AP", "AUC", "F1", "Recall", "Precision"]}
            verify.append({"Model": name, **{k: f"{v:.3e}" for k, v in diffs.items()},
                           "max_abs_diff": max(diffs.values())})
            print("verify", name, "max_abs_diff", max(diffs.values()), flush=True)

        best_param = json.loads(row["best_param_json"])
        ratio = y[y == 0].shape[0] / y[y == 1].shape[0]
        clf = M.InitializeClassifier("XGBoost", best_param, ratio)
        clf.fit(X[cols], y)
        prob = clf.predict_proba(orp[cols].values)[:, 1]
        thr = float(row["Threshold"])
        pred = (prob > thr).astype(int)
        pd.DataFrame({"GeneID": orp.index, "Probability": prob, "BinaryPrediction": pred}).to_csv(
            os.path.join(args.outdir, f"{name}_orphan_predictions.tsv"), sep="\t", index=False)

        n_call = int(pred.sum())
        rows.append(OrderedDict([
            ("Model", name), ("Features", len(cols)),
            ("CV AP", round(float(row["Best_CV_AP_grid"]), 4)),
            ("Test AP", round(float(row["AP"]), 4)),
            ("Test AUC", round(float(row["AUC"]), 4)),
            ("Threshold", round(thr, 4)),
            ("Orphans called de novo", round(100.0 * n_call / len(orp), 2)),
            ("n", n_call), ("Denominator", len(orp)),
            ("uc_method_used", row["uc_method_used"]),
        ]))
        print(name, "orphans:", n_call, f"({100.0 * n_call / len(orp):.2f}%)", flush=True)

    s13 = pd.DataFrame(rows)
    s13.to_csv(os.path.join(args.outdir, "S13_unified.tsv"), sep="\t", index=False)
    if verify:
        pd.DataFrame(verify).to_csv(os.path.join(args.outdir, "verify_vs_S12.tsv"), sep="\t", index=False)
    print("\n=== S13_unified ===")
    print(s13.to_string(index=False))


if __name__ == "__main__":
    main()
