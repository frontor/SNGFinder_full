# -*- coding: utf-8 -*-
"""Compute the FULL metric row (all Supplementary Table S12 columns) for one
ablation feature set, so that Table S12 can carry the same row set as Table S13.

The unified run (ablation_orphan_unify.py) emits the per-gene orphan calls for
all nine models but only the Table S13 summary columns; this script recomputes
the complete S12 metric row for a single model. It was used for the
Gene_architecture_only row of Supplementary Table S12.

Usage (from the repository root):
    python scripts/classifier_ablation/ablation_row_full.py Gene_architecture_only [outdir]

Reads the deposited inputs from classifier_data/ and writes
<outdir>/<model>_full_row.tsv (default outdir: ablation_row_out/).
"""
import sys, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import ModelTrain_v3 as M

DATA = os.path.join(ROOT, "classifier_data")
model = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "ablation_row_out"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "Training_Dataset.csv"), sep="\t", header=0, index_col=0).dropna()
X, y = df.drop("Categories", axis=1), df["Categories"]
dt = pd.read_csv(os.path.join(DATA, "Test_Dataset.csv"), sep="\t", header=0, index_col=0).dropna()
X_test, y_test = dt.drop("Categories", axis=1), dt["Categories"]

fg = M.detect_feature_groups(X.columns)
cols = fg["families_detailed"]["Gene_architecture"] if model == "Gene_architecture_only" else None
assert cols is not None and len(cols) == 6, cols

# same parameters as the reported run (UCMethod A, PPV 0.95, NPV 0.995, 30/200)
row = M.run_original_style_subset_model(
    "XGBoost", X[cols], y, X_test[cols], y_test,
    0.10, "A", 0.95, 0.995, 30, 200, n_jobs=10)
row["Model"] = model

# S12 needs "Selective Accuracy", which the helper does not emit
tn, fp = float(row["Selective_TN"]), float(row["Selective_FP"])
fn, tp = float(row["Selective_FN"]), float(row["Selective_TP"])
row["Selective_Accuracy"] = (tp + tn) / (tp + tn + fp + fn)

out = os.path.join(OUT, f"{model}_full_row.tsv")
pd.DataFrame([row]).to_csv(out, sep="\t", index=False)
print(pd.DataFrame([row]).T.to_string())
