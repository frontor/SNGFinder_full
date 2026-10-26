# -*- coding: utf-8 -*-
"""
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.environ.get("SNGFINDER_DATA", os.getcwd())
OUT = os.path.join(BASE, "strata_outputs")
os.makedirs(OUT, exist_ok=True)
L = []
def p(s=""):
    print(s, flush=True); L.append(str(s))

tw = pd.read_csv(os.path.join(BASE, "threeway_outputs", "orphan_origins_v2.tsv"), sep="\t")
bs = pd.read_csv(os.path.join(BASE, "synteny_outputs", "orphan_block_state_precise.tsv"),
                 sep="\t")
orph = pd.read_csv(os.path.join(BASE, "Orphan_Dataset.csv"), sep="\t", index_col=0)
proba = np.load(os.path.join(BASE, "diagnosis_outputs", "orphan_proba_full.npy"))

# ---------------- 物种 -> 系统发育深度与节点年龄 ----------------
# Stein et al. 2018 Nat Genet 50:285-296: AA-BB 6.76, AA crown 2.41, Asian rice 0.55
STRATA = {
    "Oryza_rufipogon":  ("Asian AA (japonica sister)",      0.55,  2.41),
    "Oryza_nivara":     ("Asian AA (japonica sister)",      0.55,  2.41),
    "Oryza_indica":     ("Asian AA (japonica sister)",      0.55,  2.41),
    "Oryza_glaberrima": ("African AA",                      None,  2.41),
    "Oryza_barthii":    ("African AA",                      None,  2.41),
    "Oryza_glumipatula":("South-American AA",               None,  2.41),
    "Oryza_meridionalis":("Australian AA (AA crown)",       2.41,  6.76),
}

p("=" * 74)
p("A. 3,724 个 oryza_clade 基因的逐级分层")
p("=" * 74)
sub = tw[tw["class"] == "oryza_clade"].copy()
p("oryza_clade 总数: %d" % len(sub))

# 每基因：最远同源物种的祖先位点状态 + 全物种是否全非编码
state_cols = ["Oryza_barthii", "Oryza_glaberrima", "Oryza_glumipatula",
              "Oryza_indica", "Oryza_meridionalis", "Oryza_nivara",
              "Oryza_rufipogon", "Oryza_punctata", "Oryza_brachyantha",
              "Leersia_perrieri"]
bs = bs.set_index("gene")
sub["all_noncoding"] = sub["gene"].map(
    lambda g: bool((bs.loc[g, state_cols] == "noncoding").all()) if g in bs.index else None)
sub["any_outgroup_coding"] = sub["gene"].map(
    lambda g: bool((bs.loc[g, ["Oryza_punctata", "Oryza_brachyantha",
                               "Leersia_perrieri"]] == "coding_overlap").any())
    if g in bs.index else None)
sub["far_state"] = sub["farthest_homolog"].map(
    lambda g: (bs.loc[g, "far_state_tmp"] if False else None))

# 最远同源物种位点的编码状态
far_state = []
for _, r in sub.iterrows():
    g, fh = r["gene"], r["farthest_homolog"]
    if g in bs.index and fh in state_cols:
        far_state.append(bs.loc[g, fh])
    else:
        far_state.append(None)
sub["farthest_state"] = far_state

orph_c = orph.dropna()                     # 与 proba 对齐（7 个缺失值基因被剔除）
proba_map = dict(zip(orph_c.index, proba))
sub["proba"] = sub["gene"].map(proba_map)

TISSUE = ["Flagleaf", "Root", "Panicle", "Youngleaf", "Pollen", "Sperm", "Egg"]
oi = orph.reindex(sub["gene"].values)
sub["breadth"] = (oi[TISSUE] > 0).sum(axis=1).values
sub["DISO"] = oi["DISO"].values

rows = []
for fh in ["Oryza_meridionalis", "Oryza_glumipatula", "Oryza_glaberrima",
           "Oryza_barthii", "Oryza_indica", "Oryza_nivara", "Oryza_rufipogon"]:
    s = sub[sub["farthest_homolog"] == fh]
    if len(s) == 0:
        continue
    lab, lo, hi = STRATA[fh]
    rows.append({
        "Farthest homolog": fh.replace("Oryza_", "O. "),
        "Lineage": lab,
        "Origination window (Ma)": ("%.2f\u2013%.2f" % (lo, hi)) if lo else "< %.2f" % hi,
        "n": len(s),
        "% of oryza_clade": round(len(s) / len(sub) * 100, 2),
        "% of all 6,149 orphans": round(len(s) / len(tw) * 100, 2),
        "all-species noncoding ancestor (%)": round(s["all_noncoding"].mean() * 100, 1),
        "outgroup interval coding (%)": round(s["any_outgroup_coding"].mean() * 100, 1),
        "farthest-homolog locus coding (%)":
            round((s["farthest_state"] == "coding_overlap").mean() * 100, 1),
        "median classifier P(de novo)": round(float(s["proba"].median()), 3),
        "median tissue breadth": float(s["breadth"].median()),
        "median intrinsic disorder": round(float(s["DISO"].median()), 3),
    })
st = pd.DataFrame(rows)
p(st.to_string(index=False))

# 汇总：按窗口合并
p("\n--- 按起源窗口合并 ---")
grp = {
    "0.55\u20132.41 Ma (Asian-rice level)":
        sub[sub["farthest_homolog"].isin(["Oryza_indica", "Oryza_nivara", "Oryza_rufipogon"])],
    "< 2.41 Ma (African / South-American AA)":
        sub[sub["farthest_homolog"].isin(["Oryza_glaberrima", "Oryza_barthii",
                                          "Oryza_glumipatula"])],
    "2.41\u20136.76 Ma (AA crown level)":
        sub[sub["farthest_homolog"] == "Oryza_meridionalis"],
}
grows = []
for k, s in grp.items():
    grows.append({
        "Stratum": k, "n": len(s),
        "% of oryza_clade": round(len(s) / len(sub) * 100, 2),
        "all-species noncoding ancestor (%)": round(s["all_noncoding"].mean() * 100, 1),
        "outgroup interval coding (%)": round(s["any_outgroup_coding"].mean() * 100, 1),
        "median P(de novo)": round(float(s["proba"].median()), 3),
        "median breadth": float(s["breadth"].median()),
        "median DISO": round(float(s["DISO"].median()), 3),
    })
gt = pd.DataFrame(grows)
p(gt.to_string(index=False))

# de novo 下界：全 orphan 池按 strata 汇总
p("\n--- de novo 下界（全 orphan 池）---")
p("  oryza_clade 中 all-species noncoding = %d (%.1f%%)"
  % (int(sub["all_noncoding"].sum()), sub["all_noncoding"].mean() * 100))
p("  + de_novo_noncoding 类            = %d" % int((tw["class"] == "de_novo_noncoding").sum()))
p("  = 严格 de novo 下界               = %d / %d (%.1f%%)"
  % (int(sub["all_noncoding"].sum()) + int((tw["class"] == "de_novo_noncoding").sum()),
     len(tw),
     (int(sub["all_noncoding"].sum()) + int((tw["class"] == "de_novo_noncoding").sum()))
     / len(tw) * 100))
p("  + independent_occupier 类         = %d -> 宽下界 %d (%.1f%%)"
  % (int((tw["class"] == "independent_occupier").sum()),
     int(sub["all_noncoding"].sum()) + int((tw["class"] == "de_novo_noncoding").sum())
     + int((tw["class"] == "independent_occupier").sum()),
     (int(sub["all_noncoding"].sum()) + int((tw["class"] == "de_novo_noncoding").sum())
      + int((tw["class"] == "independent_occupier").sum())) / len(tw) * 100))


# ---------------- 导出 ----------------
with pd.ExcelWriter(os.path.join(OUT, "strata_tables.xlsx"), engine="openpyxl") as w:
    st.to_excel(w, sheet_name="S24_oryza_clade_strata", index=False)
    gt.to_excel(w, sheet_name="S24b_strata_merged", index=False)
with open(os.path.join(OUT, "strata_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
sub[["gene", "farthest_homolog", "homolog_species", "all_noncoding",
     "any_outgroup_coding", "farthest_state", "proba", "breadth", "DISO"]] \
    .to_csv(os.path.join(OUT, "oryza_clade_stratified.tsv"), sep="\t", index=False)
p("\n输出 -> %s" % OUT)
