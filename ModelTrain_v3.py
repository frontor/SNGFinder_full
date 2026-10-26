#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModelTrain_v3.py

Goal:
- maximize practical agreement with the historical full-model results;
- avoid re-implementing the baseline in a way that silently changes it;
- add reviewer-driven supplementary tables/figures without changing the original
  best_param / best_threshold / t_low / t_high / final orphan predictions path.

Examples:
# headline XGBoost model
python ModelTrain_v3.py \
    XGBoost Training_Dataset.csv Test_Dataset.csv Orphan_Dataset.csv \
    A_ppv0.95_npv0.995_30_200 0.2 A 0.95 0.995 30 200 revision_outputs

# AdaCost comparison model (requires adacost.py, vendored in this directory)
python ModelTrain_v3.py \
    AdaCost Training_Dataset.csv Test_Dataset.csv Orphan_Dataset.csv \
    A_ppv0.95_npv0.995_30_200 0.2 A 0.95 0.995 30 200 revision_outputs
"""
import json
import math
import os
import pickle
import platform
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score, accuracy_score,
    matthews_corrcoef, precision_recall_curve, auc,
    average_precision_score, precision_score, recall_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, TunedThresholdClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFE
from scipy.special import logit
from xgboost import XGBClassifier
import xgboost

# optional import; original script imported this at top-level, which breaks XGBoost-only runs
try:
    from adacost import AdaCostClassifier  # noqa
except Exception:
    AdaCostClassifier = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

UNCLASSIFIED_CODE = -1
UNCLASSIFIED_LABEL = "unclassified"
RANDOM_STATE = 888


# =========================================================
# Original helper functions (kept as close as possible)
# =========================================================
def InitializeClassifier(Classifier, BestParam, ScalePosWeight):
    if Classifier == "XGBoost":
        clf = XGBClassifier(
            scale_pos_weight=ScalePosWeight,
            learning_rate=BestParam['learning_rate'],
            n_estimators=BestParam['n_estimators'],
            max_depth=BestParam['max_depth'],
            min_child_weight=BestParam['min_child_weight'],
            importance_type='gain'
        )
    elif Classifier == "AdaCost":
        if AdaCostClassifier is None:
            raise RuntimeError("AdaCostClassifier is unavailable in this environment.")
        clf = AdaCostClassifier(
            algorithm='SAMME',
            FNcost=BestParam['FNcost'],
            FPcost=BestParam["FPcost"],
            learning_rate=BestParam["learning_rate"],
            n_estimators=BestParam["n_estimators"]
        )
    else:
        print("Classifier error")
        sys.exit()
    return clf


def ApplyUnclassifiedThreshold(proba_list, t_low, t_high, unclassified_code=UNCLASSIFIED_CODE):
    out = []
    for p in proba_list:
        if p >= t_high:
            out.append(1)
        elif p <= t_low:
            out.append(0)
        else:
            out.append(unclassified_code)
    return out


def UCCodeToLabel(code, unclassified_label=UNCLASSIFIED_LABEL, unclassified_code=UNCLASSIFIED_CODE):
    try:
        if pd.isna(code):
            return np.nan
    except Exception:
        pass
    if int(code) == int(unclassified_code):
        return unclassified_label
    return int(code)


def pick_t_high_for_ppv(y, p, ppv_target=0.90, min_pred_pos=20):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    thresholds = np.unique(p)
    thresholds.sort()
    best = None
    for t in thresholds[::-1]:
        pred_pos = p >= t
        n_pos = int(pred_pos.sum())
        if n_pos < int(min_pred_pos):
            continue
        tp = int(((y == 1) & pred_pos).sum())
        ppv = tp / n_pos if n_pos > 0 else 0.0
        if ppv >= float(ppv_target):
            best = float(t)
    return best


def pick_t_low_for_npv(y, p, npv_target=0.99, min_pred_neg=50):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    thresholds = np.unique(p)
    thresholds.sort()
    best = None
    for t in thresholds:
        pred_neg = p <= t
        n_neg = int(pred_neg.sum())
        if n_neg < int(min_pred_neg):
            continue
        tn = int(((y == 0) & pred_neg).sum())
        npv = tn / n_neg if n_neg > 0 else 0.0
        if npv >= float(npv_target):
            best = float(t)
    return best


# =========================================================
# New supplementary helper functions
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_binary_metrics(y_true, probas, threshold):
    y_true = np.asarray(y_true).astype(int)
    probas = np.asarray(probas).astype(float)
    y_pred = (probas > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = OrderedDict()
    out["Threshold"] = float(threshold)
    out["TN"] = int(tn)
    out["FP"] = int(fp)
    out["FN"] = int(fn)
    out["TP"] = int(tp)
    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["Precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["F1"] = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    out["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["NPV"] = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    out["AUC"] = float(roc_auc_score(y_true, probas))
    out["AP"] = float(average_precision_score(y_true, probas, pos_label=1))
    precision_point, recall_point, _ = precision_recall_curve(y_true, probas, pos_label=1)
    out["AUPRC_by_auc"] = float(auc(recall_point, precision_point))
    out["Brier"] = float(brier_score_loss(y_true, probas))
    return out


def compute_selective_metrics(y_true, probas, t_low, t_high):
    y_true = np.asarray(y_true).astype(int)
    probas = np.asarray(probas).astype(float)
    y_sel = np.asarray(ApplyUnclassifiedThreshold(probas, t_low, t_high), dtype=int)
    mask = y_sel != UNCLASSIFIED_CODE
    abst_mask = ~mask
    out = OrderedDict()
    out["t_low"] = float(t_low)
    out["t_high"] = float(t_high)
    out["Coverage"] = float(mask.mean()) if len(mask) else np.nan
    out["AbstentionRate"] = float(1.0 - out["Coverage"]) if len(mask) else np.nan
    out["AbstainedPositives"] = int(np.sum(y_true[abst_mask] == 1))
    out["AbstainedNegatives"] = int(np.sum(y_true[abst_mask] == 0))
    if np.sum(mask) == 0:
        for k in [
            "Selective_TN", "Selective_FP", "Selective_FN", "Selective_TP",
            "Selective_Precision", "Selective_Recall", "Selective_Specificity",
            "Selective_NPV", "Selective_F1", "Selective_MCC",
            "OverallRecallTreatAbstentionAsMiss","Selective_AUC"
        ]:
            out[k] = np.nan
        return out, y_sel
    tn, fp, fn, tp = confusion_matrix(y_true[mask], y_sel[mask], labels=[0, 1]).ravel()
    out["Selective_TN"] = int(tn)
    out["Selective_FP"] = int(fp)
    out["Selective_FN"] = int(fn)
    out["Selective_TP"] = int(tp)
    out["Selective_Precision"] = float(precision_score(y_true[mask], y_sel[mask], pos_label=1, zero_division=0))
    out["Selective_Recall"] = float(recall_score(y_true[mask], y_sel[mask], pos_label=1, zero_division=0))
    out["Selective_Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["Selective_NPV"] = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    out["Selective_F1"] = float(f1_score(y_true[mask], y_sel[mask], average="binary", pos_label=1, zero_division=0))
    out["Selective_MCC"] = float(matthews_corrcoef(y_true[mask], y_sel[mask]))
    out["OverallRecallTreatAbstentionAsMiss"] = float(tp / (tp + fn + out["AbstainedPositives"])) if (tp + fn + out["AbstainedPositives"]) > 0 else np.nan
    if len(np.unique(y_true[mask])) == 2:
        out["Selective_AUC"] = float(roc_auc_score(y_true[mask], probas[mask]))
    else:
        out["Selective_AUC"] = np.nan
    return out, y_sel


def compute_negative_only_metrics_detailed(probas, binary_threshold, t_low, t_high):
    """
    Detailed safety-oriented metrics for a negatives-only set.

    Parameters
    ----------
    probas : array-like
        Predicted probabilities for samples whose true label is all 0.
    binary_threshold : float
        Threshold for the original binary classifier.
    t_low, t_high : float
        Reject-option thresholds.

    Returns
    -------
    OrderedDict
        Detailed metrics for negatives only.
    """
    probas = np.asarray(probas).astype(float)
    n = len(probas)

    binary_pred = (probas > binary_threshold).astype(int)
    selective_pred = np.asarray(ApplyUnclassifiedThreshold(probas.tolist(), t_low, t_high), dtype=int)

    # Binary counts/rates
    binary_fp = int(np.sum(binary_pred == 1))
    binary_tn = int(np.sum(binary_pred == 0))

    # Selective counts/rates
    selective_fp = int(np.sum(selective_pred == 1))
    selective_tn = int(np.sum(selective_pred == 0))
    selective_uc = int(np.sum(selective_pred == UNCLASSIFIED_CODE))
    selective_classified = int(np.sum(selective_pred != UNCLASSIFIED_CODE))

    out = OrderedDict()
    out["n_negatives"] = int(n)

    out["binary_false_positive_count"] = binary_fp
    out["binary_false_positive_rate"] = float(binary_fp / n) if n > 0 else np.nan
    out["binary_true_negative_count"] = binary_tn
    out["binary_true_negative_rate"] = float(binary_tn / n) if n > 0 else np.nan

    out["selective_false_positive_count"] = selective_fp
    out["selective_false_positive_rate"] = float(selective_fp / n) if n > 0 else np.nan
    out["selective_true_negative_count"] = selective_tn
    out["selective_true_negative_rate"] = float(selective_tn / n) if n > 0 else np.nan

    out["selective_unclassified_count"] = selective_uc
    out["selective_unclassified_rate"] = float(selective_uc / n) if n > 0 else np.nan

    out["selective_classified_count"] = selective_classified
    out["selective_coverage"] = float(selective_classified / n) if n > 0 else np.nan

    # Among only the classified negatives, what fraction are safely called negative?
    out["selective_specificity_on_classified"] = (
        float(selective_tn / selective_classified) if selective_classified > 0 else np.nan
    )

    # Among all negatives, what fraction are safely retained as high-confidence negatives?
    out["safe_negative_yield"] = float(selective_tn / n) if n > 0 else np.nan

    return out


def save_prcurve_npz(y_true, probas, out_npz):
    precision_point, recall_point, thresholds = precision_recall_curve(y_true, probas, pos_label=1)
    np.savez_compressed(out_npz, Precision=precision_point, Recall=recall_point, Threshold=thresholds)
    return precision_point, recall_point, thresholds


def plot_cv_test_prcurve(cv_recall, cv_precision, test_recall, test_precision, cv_ap, test_ap, out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=300)
    ax.plot(cv_recall, cv_precision, label=f"CV OOF (AP={cv_ap:.3f})")
    ax.plot(test_recall, test_precision, label=f"Test (AP={test_ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_prediction_probability_density(probabilities, t_low, t_high, out_png, out_pdf):
    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.hist(probabilities, bins=bins, density=True, alpha=0.8)
    ax.axvspan(t_low, t_high, alpha=0.20)
    ax.axvline(t_low, linestyle=":")
    ax.axvline(t_high, linestyle=":")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probability density in the final prediction set")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_prediction_pie(pred_labels, out_png, out_pdf):
    label_map = {0: "old gene", 1: "de novo", UNCLASSIFIED_LABEL: "unclassified"}
    ser = pd.Series(pred_labels).map(lambda x: label_map.get(x, str(x)))
    counts = ser.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.pie(counts.values, labels=counts.index.tolist(), autopct="%1.1f%%", startangle=90)
    ax.set_title("Classification of the final prediction set")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return counts


def plot_calibration(y_true, probas, out_tsv, out_png, out_pdf):
    frac_pos, mean_pred = calibration_curve(y_true, probas, n_bins=10, strategy="quantile")
    df = pd.DataFrame({"MeanPredicted": mean_pred, "FractionPositive": frac_pos})
    df.to_csv(out_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration curve")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return df


def detect_feature_groups(columns):
    columns = list(columns)
    protein_property = [c for c in columns if c in {"SignalP", "DISO", "AccExposed", "TM"}]
    gene_architecture = [c for c in columns if c in {
        "GeneLength", "IsoformNumber", "IsoformN", "ExonNumber", "ExonN",
        "AverageExonLength", "MeanExonLength", "ExonL",
        "AverageIntronLength", "MeanIntronLength", "IntronL",
        "AverageCDSLength", "MeanCDSLength", "CDSL"
    }]
    nucleotide_composition = [c for c in columns if c in {"GCContent", "PurineContent"}]
    codon_translation_adaptation = [c for c in columns if c in {"DCBS", "gtAI"}]
    expression = [c for c in columns if c in {
        "Cold6", "Salt6", "SaltMock6", "Cold3", "Salt3", "SaltMock3",
        "ABA", "ABAMock", "Flagleaf", "Root", "Panicle", "Youngleaf",
        "Pollen", "Sperm", "Egg", "Rehydration", "Deyhydration", "DryCtrl",
    }]
    r_loop = [c for c in columns if ("sense" in c.lower()) or ("antisense" in c.lower())]
    accessibility = [c for c in columns if ("faire" in c.lower()) or ("atac" in c.lower()) or ("dnase" in c.lower())]
    active_histone = [c for c in columns if any(mark in c.lower() for mark in ["h3k27ac", "h3k4me1", "h3k4me3", "h3k36me3"])]
    repressive_histone = [c for c in columns if any(mark in c.lower() for mark in ["h3k27me3", "h3k9me1", "h3k9me2", "h3k9me3"])]
    rnapii = [c for c in columns if ("rnapii" in c.lower()) or ("polii" in c.lower()) or ("rna_pol_ii" in c.lower())]

    assigned = set(
        protein_property + gene_architecture + nucleotide_composition +
        codon_translation_adaptation + expression + r_loop + accessibility +
        active_histone + repressive_histone + rnapii
    )
    other_reg = [c for c in columns if c not in assigned]
    simple = protein_property + gene_architecture + nucleotide_composition + codon_translation_adaptation
    chromatin = r_loop + accessibility + active_histone + repressive_histone + rnapii + other_reg
    families = OrderedDict([
        ("Protein_property", protein_property),
        ("Gene_architecture", gene_architecture),
        ("Nucleotide_composition", nucleotide_composition),
        ("Codon_translation_adaptation", codon_translation_adaptation),
        ("Expression", expression),
        ("R_loop", r_loop),
        ("Chromatin_accessibility", accessibility),
        ("Active_histone_marks", active_histone),
        ("Repressive_histone_marks", repressive_histone),
        ("RNAPII_occupancy", rnapii),
        ("Other_regulatory", other_reg),
    ])
    return {"simple": simple, "expression": expression, "chromatin": chromatin, "families_detailed": families}


def save_feature_membership(feature_groups, columns, out_tsv):
    rows = []
    for feat in columns:
        family = None
        for fam, members in feature_groups["families_detailed"].items():
            if feat in members:
                family = fam
                break
        rows.append({
            "Feature": feat,
            "Simple_group": int(feat in feature_groups["simple"]),
            "Expression_group": int(feat in feature_groups["expression"]),
            "Chromatin_group": int(feat in feature_groups["chromatin"]),
            "Detailed_family": family,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep="\t", index=False)
    return df


def plot_gain_importance(full_model, topk, out_tsv, out_png, out_pdf):
    booster = full_model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    rows = [{"Feature": feat, "Gain": float(gain)} for feat, gain in gain_dict.items()]
    df = pd.DataFrame(rows).sort_values("Gain", ascending=False)
    df.to_csv(out_tsv, sep="\t", index=False)
    top = df.head(topk).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.barh(top["Feature"], top["Gain"])
    ax.set_xlabel("Gain")
    ax.set_ylabel("")
    ax.set_title(f"Top {topk} XGBoost gain features")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return df


def permutation_importance_table(model, X_test, y_test, repeats, out_tsv, out_png, out_pdf, topk=20):
    result = permutation_importance(
        model, X_test, y_test,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_mean_delta_AP": result.importances_mean,
        "Importance_std": result.importances_std,
    }).sort_values("Importance_mean_delta_AP", ascending=False)
    df.to_csv(out_tsv, sep="\t", index=False)
    top = df.head(topk).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.barh(top["Feature"], top["Importance_mean_delta_AP"], xerr=top["Importance_std"])
    ax.set_xlabel("Permutation importance (delta AP)")
    ax.set_ylabel("")
    ax.set_title(f"Top {topk} permutation importance features")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return df


def grouped_permutation_importance(model, X_test, y_test, groups, repeats, out_tsv, out_png, out_pdf):
    rng = np.random.default_rng(RANDOM_STATE)
    baseline = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    rows = []
    for group_name, cols in groups.items():
        if len(cols) == 0:
            continue
        drops = []
        for _ in range(repeats):
            Xp = X_test.copy()
            perm_idx = rng.permutation(len(Xp))
            Xp.loc[:, cols] = Xp.iloc[perm_idx][cols].to_numpy()
            score = average_precision_score(y_test, model.predict_proba(Xp)[:, 1])
            drops.append(baseline - score)
        rows.append({
            "Group": group_name,
            "n_features": len(cols),
            "Importance_mean_delta_AP": float(np.mean(drops)),
            "Importance_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("Importance_mean_delta_AP", ascending=False)
    df.to_csv(out_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    ax.bar(df["Group"], df["Importance_mean_delta_AP"], yerr=df["Importance_std"])
    ax.set_ylabel("Grouped permutation importance (delta AP)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Importance of feature families")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return df


def extract_shap_matrix(shap_values):
    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        if len(values) == 0:
            raise ValueError("Empty SHAP values list.")
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] == 1:
            values = values[:, :, 0]
        else:
            values = values[:, :, -1]
    if values.ndim != 2:
        raise ValueError(f"Unsupported SHAP values shape: {values.shape}")
    return values


def grouped_shap_importance(shap_matrix, X_test, groups, out_tsv, out_png, out_pdf):
    rows = []
    col_to_idx = {c: i for i, c in enumerate(X_test.columns)}
    for group_name, cols in groups.items():
        cols = [c for c in cols if c in col_to_idx]
        if len(cols) == 0:
            continue
        idx = [col_to_idx[c] for c in cols]
        group_abs_shap = np.abs(shap_matrix[:, idx]).sum(axis=1)
        rows.append({
            "Group": group_name,
            "n_features": len(cols),
            "Importance_mean_abs_SHAP": float(np.mean(group_abs_shap)),
            "Importance_std": float(np.std(group_abs_shap, ddof=1)) if len(group_abs_shap) > 1 else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("Importance_mean_abs_SHAP", ascending=False)
    df.to_csv(out_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    ax.bar(df["Group"], df["Importance_mean_abs_SHAP"], yerr=df["Importance_std"])
    ax.set_ylabel("Grouped SHAP importance (mean sum abs SHAP)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Importance of feature families")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return df


def run_shap(model, X_test, feature_membership, feature_groups, outdir, prefix, topk=20):
    try:
        import shap
    except Exception as e:
        with open(os.path.join(outdir, f"{prefix}_shap_skipped.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"SHAP could not be imported: {e}\n")
        return None
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_matrix = extract_shap_matrix(shap_values)
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "MeanAbsSHAP": np.abs(shap_matrix).mean(axis=0),
    }).merge(feature_membership[["Feature", "Detailed_family"]], on="Feature", how="left")
    importance = importance.sort_values("MeanAbsSHAP", ascending=False)
    importance.to_csv(os.path.join(outdir, f"{prefix}_shap_importance.tsv"), sep="\t", index=False)

    grouped_shap_importance(
        shap_matrix,
        X_test,
        feature_groups,
        os.path.join(outdir, f"{prefix}_grouped_shap_importance.tsv"),
        os.path.join(outdir, f"{prefix}_grouped_shap_importance.png"),
        os.path.join(outdir, f"{prefix}_grouped_shap_importance.pdf"),
    )

    try:
        plt.figure(figsize=(8, 10), dpi=300)
        shap.plots.beeswarm(shap_values, max_display=topk, show=False, plot_size=(13, 10))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_shap_beeswarm_top{topk}.png"), bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(outdir, f"{prefix}_shap_beeswarm_top{topk}.pdf"), bbox_inches="tight")
        plt.close()
    except Exception:
        plt.figure(figsize=(8, 10), dpi=300)
        shap.summary_plot(shap_matrix, X_test, max_display=topk, show=False, plot_size=(13, 10))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_shap_beeswarm_top{topk}.png"), bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(outdir, f"{prefix}_shap_beeswarm_top{topk}.pdf"), bbox_inches="tight")
        plt.close()
    return importance
def standardized_mean_difference(df, label_col, covariates):
    rows = []
    g1 = df[df[label_col] == 1]
    g0 = df[df[label_col] == 0]
    for cov in covariates:
        m1 = g1[cov].mean()
        m0 = g0[cov].mean()
        v1 = g1[cov].var(ddof=1)
        v0 = g0[cov].var(ddof=1)
        pooled = np.sqrt((v1 + v0) / 2.0)
        smd = (m1 - m0) / pooled if pooled > 0 else 0.0
        rows.append({"Covariate": cov, "Mean_positive": m1, "Mean_negative": m0, "SMD": smd})
    return pd.DataFrame(rows)


def propensity_score_match(df, match_vars, label_col="TrueLabel", seed=RANDOM_STATE):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[match_vars])
    ps_model = LogisticRegression(max_iter=5000, random_state=seed)
    ps_model.fit(Xs, df[label_col].astype(int))
    ps = ps_model.predict_proba(Xs)[:, 1]
    tmp = df.copy()
    tmp["PropensityScore"] = ps
    tmp["LogitPS"] = logit(np.clip(ps, 1e-6, 1 - 1e-6))
    caliper = 0.2 * tmp["LogitPS"].std(ddof=1)

    pos = tmp[tmp[label_col] == 1].copy().sort_values("PropensityScore")
    neg = tmp[tmp[label_col] == 0].copy()
    available_neg = neg.index.tolist()
    pairs = []
    for idx, row in pos.iterrows():
        if not available_neg:
            break
        cand = neg.loc[available_neg].copy()
        d = np.abs(cand["LogitPS"].values - row["LogitPS"])
        best_j = int(np.argmin(d))
        if d[best_j] <= caliper:
            neg_idx = cand.index[best_j]
            pairs.append({"PositiveID": idx, "NegativeID": neg_idx, "Distance_logitPS": float(d[best_j])})
            available_neg.remove(neg_idx)
    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        raise RuntimeError("No matched pairs were found under the default caliper.")
    matched_pos = tmp.loc[pair_df["PositiveID"]].copy()
    matched_neg = tmp.loc[pair_df["NegativeID"]].copy()
    matched_df = pd.concat([matched_pos, matched_neg], axis=0).copy()
    return matched_df, pair_df


def plot_love(pre_df, post_df, out_tsv, out_png, out_pdf):
    merged = pre_df[["Covariate", "SMD"]].rename(columns={"SMD": "Before_matching"}).merge(
        post_df[["Covariate", "SMD"]].rename(columns={"SMD": "After_matching"}),
        on="Covariate",
        how="outer",
    )
    merged.to_csv(out_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.scatter(merged["Before_matching"], merged["Covariate"], label="Before")
    ax.scatter(merged["After_matching"], merged["Covariate"], label="After")
    ax.axvline(0.1, linestyle="--")
    ax.axvline(-0.1, linestyle="--")
    ax.set_xlabel("Standardized mean difference")
    ax.set_ylabel("")
    ax.set_title("Covariate balance for matched negatives")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return merged


def plot_confusion(cm, labels, out_png, out_pdf, title):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def run_original_style_subset_model(Classifier, X_train, y_train, X_test, y_test, RejectQuantile, UCMethod, PPV_target, NPV_target, MinPredPos, MinPredNeg, n_jobs=10):
    ratio = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]
    scale_pos_weight = ratio
    if Classifier == "XGBoost":
        clf = XGBClassifier(scale_pos_weight=scale_pos_weight)
        param = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0], 'n_estimators': [50, 100, 200, 500, 800], 'max_depth': [3, 6, 9], 'min_child_weight': [1, 3, 5, 7]}
        scoring = 'average_precision'
    else:
        raise RuntimeError("Only XGBoost is supported in the revision patch subset models.")

    grid = GridSearchCV(clf, param_grid=param, cv=10, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    best_param = grid.best_params_

    clf = InitializeClassifier(Classifier, best_param, scale_pos_weight)
    th = TunedThresholdClassifierCV(clf, scoring="f1", cv=10, n_jobs=n_jobs)
    th.fit(X_train, y_train)
    best_threshold = th.best_threshold_

    yal = []
    sal = []
    kfold = StratifiedKFold(10, shuffle=True, random_state=888)
    for train, test in kfold.split(X_train.values, y_train.values):
        clf = InitializeClassifier(Classifier, best_param, scale_pos_weight)
        clf.fit(X_train.values[train], y_train.values[train])
        s = clf.predict_proba(X_train.values[test])[:, 1]
        yal += list(y_train.values[test])
        sal += list(s)

    sal_arr = np.array(sal, dtype=float)
    yal_arr = np.array(yal, dtype=int)

    uc_method_requested = UCMethod
    uc_method_used = None
    uc_fallback_reason = ""

    if UCMethod == "C":
        reject_margin = float(np.quantile(np.abs(sal_arr - best_threshold), RejectQuantile)) if sal_arr.size > 0 else 0.0
        t_low = max(0.0, float(best_threshold) - reject_margin)
        t_high = min(1.0, float(best_threshold) + reject_margin)
        uc_method_used = "C"
    else:
        t_high = pick_t_high_for_ppv(yal_arr, sal_arr, ppv_target=PPV_target, min_pred_pos=MinPredPos)
        t_low = pick_t_low_for_npv(yal_arr, sal_arr, npv_target=NPV_target, min_pred_neg=MinPredNeg)
        if (t_high is None) or (t_low is None) or (float(t_low) >= float(t_high)):
            reject_margin = float(np.quantile(np.abs(sal_arr - best_threshold), RejectQuantile)) if sal_arr.size > 0 else 0.0
            t_low = max(0.0, float(best_threshold) - reject_margin)
            t_high = min(1.0, float(best_threshold) + reject_margin)
            uc_method_used = "C_fallback"
            uc_fallback_reason = "A-method thresholds infeasible for this subset under current PPV/NPV/min-count constraints"
        else:
            uc_method_used = "A"

    clf = InitializeClassifier(Classifier, best_param, scale_pos_weight)
    clf.fit(X_train, y_train)
    test_prob = clf.predict_proba(X_test.values)[:, 1]
    binary = compute_binary_metrics(y_test.values, test_prob, best_threshold)
    selective, _ = compute_selective_metrics(y_test.values, test_prob, t_low, t_high)

    row = OrderedDict()
    row["n_features"] = int(X_train.shape[1])
    row["Best_CV_AP_grid"] = float(grid.best_score_)
    row.update(binary)
    row.update(selective)
    row["uc_method_requested"] = uc_method_requested
    row["uc_method_used"] = uc_method_used
    row["uc_fallback_reason"] = uc_fallback_reason
    row["best_param_json"] = json.dumps(best_param, sort_keys=True)
    return row


def bootstrap_ci(y_true, probas, threshold, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    probas = np.asarray(probas).astype(float)
    n = len(y_true)
    base = compute_binary_metrics(y_true, probas, threshold)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = probas[idx]
        if len(np.unique(yb)) < 2:
            continue
        rows.append(compute_binary_metrics(yb, pb, threshold))
    if not rows:
        return pd.DataFrame(columns=["Metric", "Estimate", "CI_lower", "CI_upper"])
    df = pd.DataFrame(rows)
    metrics = ["AUC", "AP", "AUPRC_by_auc", "Accuracy", "Precision", "Recall", "Specificity", "NPV", "F1", "MCC", "Brier"]
    out = []
    for m in metrics:
        out.append({"Metric": m, "Estimate": base[m], "CI_lower": float(df[m].quantile(0.025)), "CI_upper": float(df[m].quantile(0.975))})
    return pd.DataFrame(out)


def write_summary_text(out_path, summary_dict):
    txt = []
    txt.append("# Revision-patch note\n")
    txt.append("This run used the ORIGINAL full-model path from ModelTrain_v3_unclassified_AC.py as the baseline.\n")
    txt.append("Supplementary analyses were appended only after the baseline model, threshold, and orphan predictions were computed.\n\n")
    txt.append("## Key full-model values\n")
    txt.append(f"- best_param: {summary_dict['best_param_json']}\n")
    txt.append(f"- best_threshold: {summary_dict['best_threshold']:.6f}\n")
    txt.append(f"- t_low: {summary_dict['t_low']:.6f}\n")
    txt.append(f"- t_high: {summary_dict['t_high']:.6f}\n")
    txt.append(f"- CV AP(all OOF): {summary_dict['cv_ap_all']:.6f}\n")
    txt.append(f"- Test AP: {summary_dict['test_ap']:.6f}\n")
    txt.append(f"- Test F1: {summary_dict['test_f1']:.6f}\n")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(txt))


if __name__ == '__main__':
    # =========================================================
    # Original command-line interface
    # =========================================================
    Classifier = sys.argv[1]
    TrainDataset = sys.argv[2]
    TestDataset = sys.argv[3]
    PredictDataset = sys.argv[4]
    Tag = sys.argv[5]

    RejectQuantile = 0.10
    if len(sys.argv) > 6:
        RejectQuantile = float(sys.argv[6])
    if (RejectQuantile <= 0.0) or (RejectQuantile >= 0.5):
        print("RejectQuantile out of range (0,0.5). Reset to default 0.10")
        RejectQuantile = 0.10

    UCMethod = "C"
    PPV_target = 0.90
    NPV_target = 0.99
    MinPredPos = 20
    MinPredNeg = 50
    if len(sys.argv) > 7:
        UCMethod = str(sys.argv[7]).upper()
    if len(sys.argv) > 8:
        PPV_target = float(sys.argv[8])
    if len(sys.argv) > 9:
        NPV_target = float(sys.argv[9])
    if len(sys.argv) > 10:
        MinPredPos = int(sys.argv[10])
    if len(sys.argv) > 11:
        MinPredNeg = int(sys.argv[11])

    OutDir = None
    if len(sys.argv) > 12:
        OutDir = sys.argv[12]
    else:
        OutDir = f"revision_outputs_{Tag}"

    ensure_dir(OutDir)

    # =========================================================
    # ORIGINAL FULL-MODEL PATH (copied as closely as possible)
    # =========================================================
    df = pd.read_csv(TrainDataset, sep="\t", header=0, index_col=0)
    train_before = df.shape[0]
    print(df.shape)
    df.dropna(inplace=True)
    print(df.shape)
    train_removed = train_before - df.shape[0]
    print("Old singleton genens: ", df[df['Categories'] == 0].shape[0])
    print("De novo genes: ", df[df['Categories'] == 1].shape[0])

    X_or = df.drop('Categories', axis=1)
    y = df['Categories']
    print("X shape : ", X_or.shape)
    print("y shape : ", y.shape)
    X = X_or.copy()

    ratio = y[y == 0].shape[0] / y[y == 1].shape[0]
    print("ratio", ratio)
    scale_pos_weight = ratio

    if Classifier == "XGBoost":
        clf = XGBClassifier(scale_pos_weight=scale_pos_weight)
        param = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0], 'n_estimators': [50, 100, 200, 500, 800], 'max_depth': [3, 6, 9], 'min_child_weight': [1, 3, 5, 7]}
        scoring = 'average_precision'
    elif Classifier == "AdaCost":
        if AdaCostClassifier is None:
            raise RuntimeError("AdaCostClassifier is unavailable in this environment.")
        clf = AdaCostClassifier(algorithm='SAMME')
        param = {'FNcost': np.linspace(1.1, ratio, 5), 'FPcost': np.linspace(0.5, 1, 5), 'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0], 'n_estimators': [50, 100, 200, 500, 800]}
        scoring = 'average_precision'
    else:
        print("Classifier error")
        sys.exit()
    print("param", param)

    df_test = pd.read_csv(TestDataset, sep="\t", header=0, index_col=0)
    test_before = df_test.shape[0]
    df_test.dropna(inplace=True)
    test_removed = test_before - df_test.shape[0]
    X_test_or = df_test.drop('Categories', axis=1)
    y_test = df_test['Categories']
    X_test = X_test_or.copy()
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    grid = GridSearchCV(clf, param_grid=param, cv=10, scoring=scoring, n_jobs=10)
    grid.fit(X, y)
    best_param = grid.best_params_
    print("best param", best_param)

    AUPRCl = []
    G_meanl = []
    MCCl = []
    F1_scorel = []
    APl = []
    Recalll = []
    Precisionl = []
    yal = []
    sal = []
    oof_ids = []

    clf = InitializeClassifier(Classifier, best_param, scale_pos_weight)
    th = TunedThresholdClassifierCV(clf, scoring="f1", cv=10, n_jobs=10)
    th.fit(X, y)
    best_threshold = th.best_threshold_
    print("best threshold", best_threshold)

    kfold = StratifiedKFold(10, shuffle=True, random_state=888)
    for fold, (train, test) in enumerate(kfold.split(X.values, y.values)):
        clf = InitializeClassifier(Classifier, best_param, scale_pos_weight)
        clf.fit(X.values[train], y.values[train])
        r = clf.predict_proba(X.values[test])
        s = [ri[1] for ri in r]
        c = [1 if si > best_threshold else 0 for si in s]
        recall = recall_score(y.values[test], c, pos_label=1)
        precision = precision_score(y.values[test], c, pos_label=1)
        AUC_ROC = roc_auc_score(y.values[test], s)
        F1_score = f1_score(y.values[test], c, average="binary", pos_label=1, zero_division=0)
        MCC = matthews_corrcoef(y.values[test], c)
        G_mean = pow(recall * precision, 0.5)
        yal += list(y.values[test])
        sal += s
        oof_ids += list(X.index[test])
        AP = average_precision_score(y.values[test], s, pos_label=1)
        print(recall, precision, AUC_ROC, F1_score, MCC, G_mean, AP)
        G_meanl.append(G_mean)
        MCCl.append(MCC)
        F1_scorel.append(F1_score)
        APl.append(AP)
        Recalll.append(recall)
        Precisionl.append(precision)

    sal_arr = np.array(sal, dtype=float)
    yal_arr = np.array(yal, dtype=int)

    if UCMethod == "C":
        if sal_arr.size > 0:
            reject_margin = float(np.quantile(np.abs(sal_arr - best_threshold), RejectQuantile))
        else:
            reject_margin = 0.0
        t_low = max(0.0, float(best_threshold) - reject_margin)
        t_high = min(1.0, float(best_threshold) + reject_margin)
    elif UCMethod == "A":
        reject_margin = np.nan
        t_high = pick_t_high_for_ppv(yal_arr, sal_arr, ppv_target=PPV_target, min_pred_pos=MinPredPos)
        t_low = pick_t_low_for_npv(yal_arr, sal_arr, npv_target=NPV_target, min_pred_neg=MinPredNeg)
        if (t_high is None) or (t_low is None):
            print("ERROR: Could not find t_low/t_high that satisfy targets. Try relaxing PPV_target/NPV_target or MinPredPos/MinPredNeg.")
            sys.exit()
        if float(t_low) >= float(t_high):
            print("ERROR: t_low >= t_high. Targets are likely too strict; try relaxing PPV_target/NPV_target.")
            sys.exit()
    else:
        print("ERROR: UCMethod must be 'C' or 'A'.")
        sys.exit()

    print("-------------------------------")
    print("Unclassified (reject option)")
    print("UCMethod", UCMethod)
    if UCMethod == "A":
        print("PPV_target", PPV_target, "NPV_target", NPV_target, "MinPredPos", MinPredPos, "MinPredNeg", MinPredNeg)
    if UCMethod == "C":
        print("RejectQuantile", RejectQuantile)
        print("reject_margin", reject_margin)
    print("t_low", t_low, "t_high", t_high)

    if sal_arr.size > 0:
        cv_uc = np.array(ApplyUnclassifiedThreshold(sal_arr.tolist(), t_low, t_high), dtype=int)
        cv_mask = cv_uc != UNCLASSIFIED_CODE
        cv_coverage = float(np.mean(cv_mask)) if cv_mask.size > 0 else 0.0
        cv_abstain = 1.0 - cv_coverage
        print("CV coverage", cv_coverage, "CV abstain_rate", cv_abstain)
        if np.sum(cv_mask) > 0:
            cv_recall = recall_score(yal_arr[cv_mask], cv_uc[cv_mask], pos_label=1, zero_division=0)
            cv_precision = precision_score(yal_arr[cv_mask], cv_uc[cv_mask], pos_label=1, zero_division=0)
            cv_f1 = f1_score(yal_arr[cv_mask], cv_uc[cv_mask], average="binary", pos_label=1, zero_division=0)
            cv_mcc = matthews_corrcoef(yal_arr[cv_mask], cv_uc[cv_mask])
            cv_gmean = pow(cv_recall * cv_precision, 0.5)
            print("CV (classified only) Recall", cv_recall)
            print("CV (classified only) Precision", cv_precision)
            print("CV (classified only) F1-score", cv_f1)
            print("CV (classified only) MCC", cv_mcc)
            print("CV (classified only) G-mean", cv_gmean)
        print("-------------------------------")

        AP_all = average_precision_score(yal, sal, pos_label=1)
        precision_point, recall_point, pr_th = precision_recall_curve(yal, sal, pos_label=1)
        if Tag:
            np.savez_compressed(f"cv_{Classifier}_prcurve_{Tag}.npz", {"Precision": precision_point, "Recall": recall_point, "Threshold": pr_th})
        else:
            np.savez_compressed(f"cv_{Classifier}_prcurve.npz", {"Precision": precision_point, "Recall": recall_point, "Threshold": pr_th})
        G_mean = str(str(np.round(np.mean(np.array(G_meanl)), 4)) + "+-" + str(np.round(np.std(np.array(G_meanl)), 4)))
        MCC = str(str(np.round(np.mean(np.array(MCCl)), 4)) + "+-" + str(np.round(np.std(np.array(MCCl)), 4)))
        AP = str(str(np.round(np.mean(np.array(APl)), 4)) + "+-" + str(np.round(np.std(np.array(APl)), 4)))
        F1_score = str(str(np.round(np.mean(np.array(F1_scorel)), 4)) + "+-" + str(np.round(np.std(np.array(F1_scorel)), 4)))
        Recall = str(str(np.round(np.mean(np.array(Recalll)), 4)) + "+-" + str(np.round(np.std(np.array(Recalll)), 4)))
        Precision = str(str(np.round(np.mean(np.array(Precisionl)), 4)) + "+-" + str(np.round(np.std(np.array(Precisionl)), 4)))
        print("-------------------------------")
        print("KFold")
        print("G-mean", G_mean)
        print("MCC", MCC)
        print("AP", AP)
        print("F1-score", F1_score)
        print("Recall", Recall)
        print("Precision", Precision)
        print("AP all", AP_all)
        print("-------------------------------")

        clf = InitializeClassifier(Classifier, best_param, scale_pos_weight)
        clf.fit(X, y)
        rl = clf.predict_proba(X_test.values)
        sl = [ri[1] for ri in rl]
        cl = [1 if si > best_threshold else 0 for si in sl]
        recall = recall_score(y_test.values, cl, pos_label=1)
        precision = precision_score(y_test.values, cl, pos_label=1)
        AUC_ROC = roc_auc_score(y_test.values, sl)
        F1_score = f1_score(y_test.values, cl, average="binary", pos_label=1, zero_division=0)
        MCC = matthews_corrcoef(y_test.values, cl)
        G_mean = pow(recall * precision, 0.5)
        precision_point, recall_point, _thresholds = precision_recall_curve(y_test.values, sl, pos_label=1)
        if Tag:
            np.savez_compressed(f"test_{Classifier}_prcurve_{Tag}.npz", {"Precision": precision_point, "Recall": recall_point, "Threshold": _thresholds})
        else:
            np.savez_compressed(f"test_{Classifier}_prcurve.npz", {"Precision": precision_point, "Recall": recall_point, "Threshold": _thresholds})
        AUCPRC = auc(recall_point, precision_point)
        AP = average_precision_score(y_test.values, sl, pos_label=1)
        cm = confusion_matrix(y_test.values, cl)
        tn, fp, fn, tp = cm.ravel()
        print(recall, precision, AUC_ROC, F1_score, MCC, G_mean, AUCPRC, AP)
        print("-------------------------------")
        print("Test data")
        print(f"TN: {tn}; FP: {fp}; FN: {fn}; TP: {tp}")
        print("AUCPRC", AUCPRC)
        print("G-mean", G_mean)
        print("MCC", MCC)
        print("AP", AP)
        print("F1-score", F1_score)
        print("Recall", recall)
        print("Precision", precision)

        test_uc = np.array(ApplyUnclassifiedThreshold(sl, t_low, t_high), dtype=int)
        test_mask = test_uc != UNCLASSIFIED_CODE
        test_coverage = float(np.mean(test_mask)) if test_mask.size > 0 else 0.0
        test_abstain = 1.0 - test_coverage

        abst_mask = ~test_mask
        abst_pos = int((y_test.values[abst_mask] == 1).sum())
        abst_neg = int((y_test.values[abst_mask] == 0).sum())
        print("-------------------------------")
        print("Test data (with unclassified)")
        print("coverage", test_coverage, "abstain_rate", test_abstain)
        print("abstained positives:", abst_pos, "abstained negatives:", abst_neg)

        if np.sum(test_mask) > 0:
            tn, fp, fn, tp = confusion_matrix(y_test.values[test_mask], test_uc[test_mask], labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
            acc_cls = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")
            overall_recall_with_abstain = tp / (tp + fn + abst_pos) if (tp + fn + abst_pos) > 0 else float("nan")
            print("classified-only confusion: TN,FP,FN,TP =", tn, fp, fn, tp)
            print("Specificity(TNR)", specificity, "NPV", npv, "Accuracy(classified-only)", acc_cls)
            print("Recall (treat abstain as miss)", overall_recall_with_abstain)

            recall_uc = recall_score(y_test.values[test_mask], test_uc[test_mask], pos_label=1, zero_division=0)
            precision_uc = precision_score(y_test.values[test_mask], test_uc[test_mask], pos_label=1, zero_division=0)
            f1_uc = f1_score(y_test.values[test_mask], test_uc[test_mask], average="binary", pos_label=1, zero_division=0)
            mcc_uc = matthews_corrcoef(y_test.values[test_mask], test_uc[test_mask])
            gmean_uc = pow(recall_uc * precision_uc, 0.5)
            print("Recall (classified only)", recall_uc)
            print("Precision (classified only)", precision_uc)
            print("F1-score (classified only)", f1_uc)
            print("MCC (classified only)", mcc_uc)
            print("G-mean (classified only)", gmean_uc)
        else:
            print("All samples are unclassified; classified-only metrics are unavailable.")
            overall_recall_with_abstain = 0.0 if abst_pos > 0 else float("nan")
            print("Recall (treat abstain as miss)", overall_recall_with_abstain)

        print("-------------------------------")

        if Tag:
            model_file = f"{Classifier}Model_{Tag}.pkl"
            with open(model_file, "wb") as fm:
                pickle.dump(clf, fm)
        else:
            model_file = f"{Classifier}Model.pkl"
            with open(model_file, "wb") as fm:
                pickle.dump(clf, fm)

        df_orphan = pd.read_csv(PredictDataset, sep="\t", header=0, index_col=0)
        orphan_before = df_orphan.shape[0]
        print("df_orphan", df_orphan.shape)
        df_orphan_nan = df_orphan[df_orphan.isna().any(axis=1)].copy()
        print("df_orphan_nan", df_orphan_nan.shape)
        df_orphan.dropna(inplace=True)
        orphan_removed = orphan_before - df_orphan.shape[0]
        print("df_orphan_dropna", df_orphan.shape)
        X_orphan = df_orphan.copy()
        rl = clf.predict_proba(X_orphan.values)
        sl_orphan = [ri[1] for ri in rl]
        cl_orphan_raw = [1 if si > best_threshold else 0 for si in sl_orphan]
        df_orphan["Probabilities"] = sl_orphan
        df_orphan["Categories_raw"] = cl_orphan_raw
        cl_uc = ApplyUnclassifiedThreshold(sl_orphan, t_low, t_high)
        df_orphan["Categories"] = [UCCodeToLabel(ci) for ci in cl_uc]
        if df_orphan_nan.shape[0] > 0:
            df_orphan_nan["Probabilities"] = np.nan
            df_orphan_nan["Categories_raw"] = np.nan
            df_orphan_nan["Categories"] = np.nan
            df_orphan = pd.concat([df_orphan, df_orphan_nan])
        if Tag:
            orphan_outfile = f"Orphan_{Classifier}_predict_{Tag}.csv"
            df_orphan.loc[:, ["Probabilities", "Categories", "Categories_raw"]].to_csv(orphan_outfile, sep="\t", na_rep="NaN")
        else:
            orphan_outfile = f"Orphan_{Classifier}_predict.csv"
            df_orphan.loc[:, ["Probabilities", "Categories", "Categories_raw"]].to_csv(orphan_outfile, sep="\t", na_rep="NaN")

    # =========================================================
    # Supplementary analyses (run AFTER the original baseline)
    # =========================================================
    prefix = os.path.basename(os.path.normpath(OutDir))
    feature_groups = detect_feature_groups(X.columns)
    feature_membership = save_feature_membership(feature_groups, X.columns, os.path.join(OutDir, f"{prefix}_feature_groups.tsv"))

    # dataset details
    pd.DataFrame({
        "Dataset": ["Train", "Test", "Predict"],
        "Rows_before_dropna": [train_before, test_before, orphan_before],
        "Rows_after_dropna": [df.shape[0], df_test.shape[0], orphan_before - orphan_removed],
        "Rows_removed_by_dropna": [train_removed, test_removed, orphan_removed],
        "Negative_n": [int((y == 0).sum()), int((y_test == 0).sum()), np.nan],
        "Positive_n": [int((y == 1).sum()), int((y_test == 1).sum()), np.nan],
        "Features_n": [int(X.shape[1]), int(X_test.shape[1]), int(X_orphan.shape[1])],
    }).to_csv(os.path.join(OutDir, f"{prefix}_dataset_details.tsv"), sep="\t", index=False)

    # baseline config
    config = OrderedDict([
        ("TrainDataset", TrainDataset),
        ("TestDataset", TestDataset),
        ("PredictDataset", PredictDataset),
        ("Classifier", Classifier),
        ("Tag", Tag),
        ("RejectQuantile", float(RejectQuantile)),
        ("UCMethod", UCMethod),
        ("PPV_target", float(PPV_target)),
        ("NPV_target", float(NPV_target)),
        ("MinPredPos", int(MinPredPos)),
        ("MinPredNeg", int(MinPredNeg)),
        ("best_param", best_param),
        ("best_threshold", float(best_threshold)),
        ("t_low", float(t_low)),
        ("t_high", float(t_high)),
        ("python_version", platform.python_version()),
        ("sklearn_version", sklearn.__version__),
        ("xgboost_version", xgboost.__version__),
        ("platform", platform.platform()),
        ("note", "Full model path copied from original ModelTrain_v3_unclassified_AC.py; supplementary analyses appended afterwards."),
    ])
    with open(os.path.join(OutDir, f"{prefix}_analysis_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # prediction tables
    pd.DataFrame({
        "GeneID": oof_ids,
        "TrueLabel": yal_arr,
        "OOF_probability": sal_arr,
        "OOF_binary_prediction": (sal_arr > best_threshold).astype(int),
        "OOF_selective_prediction": np.asarray(ApplyUnclassifiedThreshold(sal_arr.tolist(), t_low, t_high), dtype=int),
    }).to_csv(os.path.join(OutDir, f"{prefix}_full_oof_predictions.tsv"), sep="\t", index=False)

    test_prob_arr = np.asarray(sl, dtype=float)
    test_binary_arr = (test_prob_arr > best_threshold).astype(int)
    test_selective_arr = np.asarray(ApplyUnclassifiedThreshold(test_prob_arr.tolist(), t_low, t_high), dtype=int)
    pd.DataFrame({
        "GeneID": X_test.index,
        "TrueLabel": y_test.values,
        "Probability": test_prob_arr,
        "BinaryPrediction": test_binary_arr,
        "SelectivePrediction": test_selective_arr,
        "SelectiveLabel": [UCCodeToLabel(x) for x in test_selective_arr],
    }).to_csv(os.path.join(OutDir, f"{prefix}_test_predictions.tsv"), sep="\t", index=False)

    pd.DataFrame({
        "GeneID": X_orphan.index,
        "Probability": np.asarray(sl_orphan, dtype=float),
        "BinaryPrediction": np.asarray(cl_orphan_raw, dtype=int),
        "SelectivePrediction": np.asarray(cl_uc, dtype=int),
        "SelectiveLabel": [UCCodeToLabel(x) for x in cl_uc],
    }).to_csv(os.path.join(OutDir, f"{prefix}_predict_predictions_clean.tsv"), sep="\t", index=False)

    # CV/test metrics
    oof_binary = compute_binary_metrics(yal_arr, sal_arr, best_threshold)
    oof_selective, _ = compute_selective_metrics(yal_arr, sal_arr, t_low, t_high)
    test_binary = compute_binary_metrics(y_test.values, test_prob_arr, best_threshold)
    test_selective = compute_selective_metrics(y_test.values, test_prob_arr, t_low, t_high)[0]
    rows = []
    for dataset_name, binary_m, selective_m in [("CV_OOF", oof_binary, oof_selective), ("Test", test_binary, test_selective)]:
        row = OrderedDict([("Dataset", dataset_name)])
        row.update(binary_m)
        row.update(selective_m)
        rows.append(row)
    cv_test_df = pd.DataFrame(rows)
    cv_test_df.to_csv(os.path.join(OutDir, f"{prefix}_cv_test_metrics.tsv"), sep="\t", index=False)

    # fold mean+-sd table exactly reflecting original printed fold metrics
    pd.DataFrame([{
        "CV_Gmean_mean": float(np.mean(np.array(G_meanl))),
        "CV_Gmean_sd": float(np.std(np.array(G_meanl))),
        "CV_MCC_mean": float(np.mean(np.array(MCCl))),
        "CV_MCC_sd": float(np.std(np.array(MCCl))),
        "CV_AP_fold_mean": float(np.mean(np.array(APl))),
        "CV_AP_fold_sd": float(np.std(np.array(APl))),
        "CV_F1_mean": float(np.mean(np.array(F1_scorel))),
        "CV_F1_sd": float(np.std(np.array(F1_scorel))),
        "CV_Recall_mean": float(np.mean(np.array(Recalll))),
        "CV_Recall_sd": float(np.std(np.array(Recalll))),
        "CV_Precision_mean": float(np.mean(np.array(Precisionl))),
        "CV_Precision_sd": float(np.std(np.array(Precisionl))),
        "CV_AP_all_OOF": float(AP_all),
        "Test_AP": float(test_binary["AP"]),
        "Test_F1": float(test_binary["F1"]),
        "Test_Recall": float(test_binary["Recall"]),
        "Test_Precision": float(test_binary["Precision"]),
        "best_threshold": float(best_threshold),
        "t_low": float(t_low),
        "t_high": float(t_high),
    }]).to_csv(os.path.join(OutDir, f"{prefix}_baseline_metrics_exactstyle.tsv"), sep="\t", index=False)

    bootstrap_ci(y_test.values, test_prob_arr, best_threshold, n_boot=1000, seed=RANDOM_STATE).to_csv(
        os.path.join(OutDir, f"{prefix}_full_model_bootstrap_CI.tsv"), sep="\t", index=False
    )

    # PR curves
    cv_prec2, cv_rec2, _ = save_prcurve_npz(yal_arr, sal_arr, os.path.join(OutDir, f"{prefix}_cv_prcurve.npz"))
    test_prec2, test_rec2, _ = save_prcurve_npz(y_test.values, test_prob_arr, os.path.join(OutDir, f"{prefix}_test_prcurve.npz"))
    plot_cv_test_prcurve(
        cv_rec2, cv_prec2, test_rec2, test_prec2,
        average_precision_score(yal_arr, sal_arr, pos_label=1),
        test_binary["AP"],
        os.path.join(OutDir, f"{prefix}_cv_test_prcurve_overlay.png"),
        os.path.join(OutDir, f"{prefix}_cv_test_prcurve_overlay.pdf")
    )

    # full model summary
    full_row = OrderedDict()
    full_row["Model"] = "Full"
    full_row["n_features"] = int(X.shape[1])
    full_row["Best_CV_AP_grid"] = float(grid.best_score_)
    full_row.update(test_binary)
    full_row.update(test_selective)
    full_row["best_param_json"] = json.dumps(best_param, sort_keys=True)
    pd.DataFrame([full_row]).to_csv(os.path.join(OutDir, f"{prefix}_full_model_summary.tsv"), sep="\t", index=False)

    # orphan figures
    plot_prediction_probability_density(np.asarray(sl_orphan, dtype=float), t_low, t_high, os.path.join(OutDir, f"{prefix}_predict_probability_density.png"), os.path.join(OutDir, f"{prefix}_predict_probability_density.pdf"))
    pie_counts = plot_prediction_pie([UCCodeToLabel(x) for x in cl_uc], os.path.join(OutDir, f"{prefix}_predict_class_pie.png"), os.path.join(OutDir, f"{prefix}_predict_class_pie.pdf"))
    pie_counts.rename_axis("Class").reset_index(name="Count").to_csv(os.path.join(OutDir, f"{prefix}_predict_class_counts.tsv"), sep="\t", index=False)

    # calibration + feature importance
    plot_calibration(y_test.values, test_prob_arr, os.path.join(OutDir, f"{prefix}_calibration_table.tsv"), os.path.join(OutDir, f"{prefix}_calibration_curve.png"), os.path.join(OutDir, f"{prefix}_calibration_curve.pdf"))
    gain_df = plot_gain_importance(clf, 20, os.path.join(OutDir, f"{prefix}_gain_importance.tsv"), os.path.join(OutDir, f"{prefix}_gain_importance_top20.png"), os.path.join(OutDir, f"{prefix}_gain_importance_top20.pdf"))
    permutation_importance_table(clf, X_test, y_test, 30, os.path.join(OutDir, f"{prefix}_permutation_importance.tsv"), os.path.join(OutDir, f"{prefix}_permutation_importance_top20.png"), os.path.join(OutDir, f"{prefix}_permutation_importance_top20.pdf"), topk=20)
    grouped_permutation_importance(clf, X_test, y_test, feature_groups["families_detailed"], 30, os.path.join(OutDir, f"{prefix}_grouped_permutation_importance.tsv"), os.path.join(OutDir, f"{prefix}_grouped_permutation_importance.png"), os.path.join(OutDir, f"{prefix}_grouped_permutation_importance.pdf"))
    run_shap(clf, X_test, feature_membership, feature_groups["families_detailed"], OutDir, prefix, topk=20)

    # matched-negative analysis
    match_vars = [c for c in ["GeneLength", "GCContent", "PurineContent", "AverageCDSLength"] if c in X_test.columns]
    test_eval_df = pd.DataFrame({
        "TrueLabel": y_test.values,
        "Probability": test_prob_arr,
        "BinaryPrediction": test_binary_arr,
    }, index=X_test.index)
    test_eval_df = pd.concat([test_eval_df, X_test[match_vars]], axis=1)

    matched_df, pair_df = propensity_score_match(test_eval_df, match_vars=match_vars, label_col="TrueLabel", seed=RANDOM_STATE)
    pair_df.to_csv(os.path.join(OutDir, f"{prefix}_matched_pairs.tsv"), sep="\t", index=False)
    matched_df.to_csv(os.path.join(OutDir, f"{prefix}_matched_dataset.tsv"), sep="\t")

    pre_balance = standardized_mean_difference(test_eval_df.copy(), "TrueLabel", match_vars)
    post_balance = standardized_mean_difference(matched_df.copy(), "TrueLabel", match_vars)
    pre_balance.to_csv(os.path.join(OutDir, f"{prefix}_pre_matching_balance.tsv"), sep="\t", index=False)
    post_balance.to_csv(os.path.join(OutDir, f"{prefix}_post_matching_balance.tsv"), sep="\t", index=False)
    plot_love(
        pre_balance, post_balance,
        os.path.join(OutDir, f"{prefix}_matching_balance.tsv"),
        os.path.join(OutDir, f"{prefix}_matching_balance.png"),
        os.path.join(OutDir, f"{prefix}_matching_balance.pdf")
    )

    # ---- matched challenge set: binary + selective metrics ----
    matched_y = matched_df["TrueLabel"].values
    matched_prob = matched_df["Probability"].values

    matched_binary_metrics = compute_binary_metrics(matched_y, matched_prob, best_threshold)
    matched_selective_metrics, matched_selective_pred = compute_selective_metrics(matched_y, matched_prob, t_low, t_high)

    matched_row = OrderedDict()
    matched_row["n_pairs"] = int(pair_df.shape[0])
    matched_row["n_samples"] = int(matched_df.shape[0])
    matched_row.update(matched_binary_metrics)
    matched_row.update(matched_selective_metrics)

    pd.DataFrame([matched_row]).to_csv(
        os.path.join(OutDir, f"{prefix}_matched_negative_performance.tsv"),
        sep="\t", index=False
    )

    # Optional: also save a selective-focused version for clarity in supplementary tables
    matched_selective_only_row = OrderedDict()
    matched_selective_only_row["n_pairs"] = int(pair_df.shape[0])
    matched_selective_only_row["n_samples"] = int(matched_df.shape[0])
    matched_selective_only_row.update(matched_selective_metrics)

    pd.DataFrame([matched_selective_only_row]).to_csv(
        os.path.join(OutDir, f"{prefix}_matched_negative_selective_metrics.tsv"),
        sep="\t", index=False
    )

    # Save per-sample matched predictions
    matched_pred_df = matched_df.copy()
    matched_pred_df["MatchedBinaryPrediction"] = (matched_prob > best_threshold).astype(int)
    matched_pred_df["MatchedSelectivePrediction"] = matched_selective_pred
    matched_pred_df["MatchedSelectiveLabel"] = [UCCodeToLabel(x) for x in matched_selective_pred]
    matched_pred_df.to_csv(
        os.path.join(OutDir, f"{prefix}_matched_negative_predictions.tsv"),
        sep="\t"
    )

    # ---- matched negatives only: detailed safety metrics ----
    matched_negative_only = matched_df[matched_df["TrueLabel"] == 0].copy()
    matched_negative_only_prob = matched_negative_only["Probability"].values

    neg_only_metrics = compute_negative_only_metrics_detailed(
        matched_negative_only_prob,
        binary_threshold=best_threshold,
        t_low=t_low,
        t_high=t_high
    )

    pd.DataFrame([neg_only_metrics]).to_csv(
        os.path.join(OutDir, f"{prefix}_matched_negative_only_metrics.tsv"),
        sep="\t", index=False
    )

    # Optional: save negatives-only per-sample predictions
    matched_negative_only_pred = np.asarray(
        ApplyUnclassifiedThreshold(matched_negative_only_prob.tolist(), t_low, t_high), dtype=int
    )
    matched_negative_only_out = matched_negative_only.copy()
    matched_negative_only_out["BinaryPrediction"] = (matched_negative_only_prob > best_threshold).astype(int)
    matched_negative_only_out["SelectivePrediction"] = matched_negative_only_pred
    matched_negative_only_out["SelectiveLabel"] = [UCCodeToLabel(x) for x in matched_negative_only_pred]
    matched_negative_only_out.to_csv(
        os.path.join(OutDir, f"{prefix}_matched_negative_only_predictions.tsv"),
        sep="\t"
    )

    cm2 = confusion_matrix(
        matched_df["TrueLabel"].values,
        (matched_df["Probability"].values > best_threshold).astype(int),
        labels=[0, 1]
    )
    plot_confusion(
        cm2,
        labels=["Old gene", "de novo"],
        out_png=os.path.join(OutDir, f"{prefix}_matched_confusion_matrix.png"),
        out_pdf=os.path.join(OutDir, f"{prefix}_matched_confusion_matrix.pdf"),
        title="Matched negative challenge set"
    )

    # ablation models (trained AFTER full baseline)
    feature_sets = OrderedDict()
    feature_sets["Full"] = list(X.columns)
    feature_sets["Simple_only"] = feature_groups["simple"]
    feature_sets["Core_length_GC"] = [c for c in ["GeneLength", "ExonNumber", "AverageCDSLength", "GCContent", "PurineContent"] if c in X.columns]
    feature_sets["Expression_only"] = feature_groups["expression"]
    feature_sets["Chromatin_only"] = feature_groups["chromatin"]
    feature_sets["Full_minus_simple"] = [c for c in X.columns if c not in set(feature_groups["simple"])]
    feature_sets["Full_minus_expression"] = [c for c in X.columns if c not in set(feature_groups["expression"])]
    feature_sets["Full_minus_chromatin"] = [c for c in X.columns if c not in set(feature_groups["chromatin"])]

    ablation_rows = []
    for name, cols in feature_sets.items():
        if len(cols) == 0:
            continue
        row = run_original_style_subset_model(
            Classifier,
            X[cols], y,
            X_test[cols], y_test,
            RejectQuantile, UCMethod, PPV_target, NPV_target, MinPredPos, MinPredNeg,
            n_jobs=10
        )
        row["Model"] = name
        ablation_rows.append(row)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(OutDir, f"{prefix}_ablation_summary.tsv"), sep="\t", index=False)

    for metric in ["AP", "F1", "Recall", "Precision", "Specificity"]:
        dfm = ablation_df[["Model", metric]].sort_values(metric, ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        ax.bar(dfm["Model"], dfm[metric])
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(f"Ablation comparison by {metric}")
        fig.tight_layout()
        fig.savefig(os.path.join(OutDir, f"{prefix}_ablation_{metric}.png"), dpi=300)
        fig.savefig(os.path.join(OutDir, f"{prefix}_ablation_{metric}.pdf"))
        plt.close(fig)

    # summary note
    write_summary_text(os.path.join(OutDir, f"{prefix}_revision_patch_note.md"), {
        "best_param_json": json.dumps(best_param, sort_keys=True),
        "best_threshold": float(best_threshold),
        "t_low": float(t_low),
        "t_high": float(t_high),
        "cv_ap_all": float(AP_all),
        "test_ap": float(test_binary["AP"]),
        "test_f1": float(test_binary["F1"]),
    })

    print(f"Supplementary outputs were written to: {OutDir}")
