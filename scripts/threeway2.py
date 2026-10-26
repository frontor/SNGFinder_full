#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threeway2.py — 三路判定 v2: 按物种分层（区分"近缘种同源"和"外群同源"）

关键修正: orphan 在近缘 AA 种(rufipogon/indica/nivara等)有序列几乎相同的同源物是**预期的**
(它们与 japonica 是近缘种, 序列本就 >95% 相同), 不代表老基因。
真正挑战"orphan 是新基因"的 = orphan 在**外群**(punctata/brachyantha/perrieri)有同源物。

对每个 orphan:
  最远有同源占据基因的物种 -> 决定它的"最早出现级别"
  de_novo_noncoding      位点无占据基因
  independent_occupier   位点被占据但占据基因与orphan无同源hit
  oryza_clade            最远同源只在近缘 AA 种 (Oryza属内起源, 相对外群是新的)
  old_preOryza           最远同源到外群 (Oryza属之前的老基因) <- 唯一挑战文章的
"""
import os, gzip, pickle
import pandas as pd

BASE = os.environ.get("SNGFINDER_DATA", os.getcwd())
ND = os.path.join(BASE, "New folder1")
OUT = os.path.join(BASE, "threeway_outputs")
os.makedirs(OUT, exist_ok=True)

L = []
def p(s=""):
    print(s, flush=True); L.append(str(s))

# 物种深度(越远越老) 与 分组
OUTGROUPS = {"Oryza_punctata", "Oryza_brachyantha", "Leersia_perrieri"}
DEPTH = {
    "Oryza_rufipogon": 7, "Oryza_nivara": 7, "Oryza_indica": 7,
    "Oryza_glumipatula": 6,
    "Oryza_glaberrima": 5, "Oryza_barthii": 5,
    "Oryza_meridionalis": 4,
    "Oryza_punctata": 3, "Oryza_brachyantha": 2, "Leersia_perrieri": 1,
}
REF = "Oryza_japonica"

p("loading pickles ...")
Blockd = pickle.load(open(os.path.join(ND,"OryzaSp11_ref_Oryza_japonica_Blockd.pickle"),"rb"))
G2T = pickle.load(open(os.path.join(ND,"OryzaSp11_GeneToTransd.pickle"),"rb"))

orph = pd.read_csv(os.path.join(BASE,"Orphan_Dataset.csv"),sep="\t",index_col=0)
orph_genes = list(orph.index)

orphan_trans = set()
for g in orph_genes:
    orphan_trans.update(G2T.get(g, []))
p(f"orphan genes={len(orph_genes)}  orphan transcripts={len(orphan_trans)}")

gene2block = {}
for bn, bd in Blockd.items():
    e = bd.get(REF)
    if e:
        for entry in e:
            gene2block[entry[2]] = (bn, entry[0], entry[1])

def overlap(a1,a2,b1,b2): return (a1<=b2) and (b1<=a2)

orphan_occupiers = {}
occupier_trans2info = {}
for g in orph_genes:
    rec = gene2block.get(g)
    if rec is None: continue
    bn, gs, ge = rec
    bd = Blockd[bn]
    occ = []
    for sp, entries in bd.items():
        if sp == REF: continue
        for (s,e,gn,st) in entries:
            if overlap(gs,ge,s,e):
                occ.append((sp, gn))
                for tr in G2T.get(gn, []):
                    occupier_trans2info.setdefault(tr, []).append((g, sp, gn))
    orphan_occupiers[g] = occ

# ---------------- 扫 BLAST，缓存 hits (带物种) ----------------
hits_cache = os.path.join(OUT, "orphan_occupier_hits.tsv")
if os.path.exists(hits_cache):
    p("发现缓存的 hits，直接读取")
    hitdf = pd.read_csv(hits_cache, sep="\t")
else:
    p("流式扫描 blast_out.txt.gz ...")
    recs = []
    with gzip.open(os.path.join(BASE,"OryzaSp11_blast_out.txt.gz"),"rt",errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12: continue
            q, t = parts[0], parts[4]
            try:
                ev=float(parts[10]); bs=float(parts[9]); pid=float(parts[11])
                qcov=(float(parts[3])-float(parts[2])+1)/float(parts[1])
                tcov=(float(parts[7])-float(parts[6])+1)/float(parts[5])
            except ValueError:
                continue
            if q in orphan_trans and t in occupier_trans2info:
                for (og,sp,ogene) in occupier_trans2info[t]:
                    recs.append((og,sp,ogene,ev,bs,pid,qcov,tcov))
            if t in orphan_trans and q in occupier_trans2info:
                for (og,sp,ogene) in occupier_trans2info[q]:
                    recs.append((og,sp,ogene,ev,bs,pid,qcov,tcov))
    hitdf = pd.DataFrame(recs, columns=["orphan","species","occupier","evalue","bitscore","pident","qcov","tcov"])
    hitdf.to_csv(hits_cache, sep="\t", index=False)
p(f"orphan-occupier hits: {len(hitdf)} 行")

# 每个 orphan 每个占据基因 的最强 hit
best = hitdf.sort_values("evalue").groupby(["orphan","species","occupier"], as_index=False).first()

# 同源判定阈值
def is_hom(r):
    return (r.evalue < 1e-5) and (r.pident >= 30) and (max(r.qcov, r.tcov) >= 0.5)
best["homolog"] = best.apply(is_hom, axis=1)

# 每个 orphan: 有同源的物种集合
hom_species = best[best.homolog].groupby("orphan")["species"].apply(set).to_dict()

rows = []
for g in orph_genes:
    occ = orphan_occupiers.get(g, [])
    hsp = hom_species.get(g, set())
    if len(occ) == 0:
        cls = "de_novo_noncoding"; far = ""
    elif len(hsp) == 0:
        cls = "independent_occupier"; far = ""
    else:
        far = min(hsp, key=lambda s: DEPTH.get(s, 99))
        if far in OUTGROUPS:
            cls = "old_preOryza"
        else:
            cls = "oryza_clade"
    rows.append({"gene": g, "class": cls,
                 "n_occupier": len(occ),
                 "n_homolog_species": len(hsp),
                 "homolog_species": ";".join(sorted(hsp)),
                 "farthest_homolog": far})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT,"orphan_origins_v2.tsv"), sep="\t", index=False)

p("\n"+"="*70); p("三路判定 v2（按物种分层）"); p("="*70)
vc = df["class"].value_counts()
total = len(df)
for k in ["de_novo_noncoding","independent_occupier","oryza_clade","old_preOryza"]:
    n = int(vc.get(k,0)); p(f"  {k:22s} {n:6d}  ({n/total*100:5.2f}%)")

p("\n关键解读:")
p(f"  de novo 候选 (noncoding + independent): {int(vc.get('de_novo_noncoding',0))+int(vc.get('independent_occupier',0))}")
p(f"  Oryza 属内起源 (oryza_clade, 近缘AA有同源): {int(vc.get('oryza_clade',0))}")
p(f"  ** 外群有同源 (old_preOryza, 挑战'orphan是新基因'): {int(vc.get('old_preOryza',0))} **")

# 外群同源的 orphan 详细
if int(vc.get('old_preOryza',0))>0:
    sub = df[df["class"]=="old_preOryza"]
    p("\nold_preOryza 的最远同源物种分布:")
    p(sub["farthest_homolog"].value_counts().to_string())

with open(os.path.join(OUT,"threeway2_report.txt"),"w",encoding="utf-8") as fh:
    fh.write("\n".join(L))
p(f"\n输出: {OUT}")
