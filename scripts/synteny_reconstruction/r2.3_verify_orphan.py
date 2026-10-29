#!/usr/bin/env python
#coding=utf-8
"""
R2.3 verification (light): re-produce / verify the article's own orphan definition
on the pred_denovo subset using the re-downloaded RBH results.

Orphan (article definition, GetGeneClassify -> FilterOrthoCluster): a japonica gene
that has NO valid syntenic RBH ortholog in ALL 3 outgroups
(Oryza_punctata, Oryza_brachyantha, Leersia_perrieri).

IMPORTANT -- correct interpretation of OrthoCluster[gene][sp]:
  * None / [] / missing  -> no ortholog
  * [0,0,0,1,'OPUNCxx']  -> ZERO-BITSCORE sentinel (syntenic neighbour gene found in
                            the block but NO BLAST-detected homology, Bitscore==0). This
                            is NOT a real ortholog; it must be counted as "no ortholog".
  * [Bitscore,Pident,Nident,Qlen,'OPUNCxx'] with Bitscore>0 -> real ortholog, BUT only
    valid if Bitscore >= GenePairsScored[gene] (the article's per-gene RBH threshold,
    see FilterOrthoCluster: below threshold the hit is marked "unused").

The naive `bool(oc.get(sp))` test is WRONG: it treats the [0,0,0,1,...] zero-bitscore
sentinel as "has ortholog" and inflates the contradiction count. We require Bitscore>0
and >= the article threshold instead.

Input:
  --orthocluster   OryzaSp11_ref_Oryza_japonica_OrthoCluster.pickle
  --bitscore-thr   OryzaSp11_ref_Oryza_japonica_BitScore_threshold.pickle  (GenePairsScored)
  --pred           pred_denovo.txt
  --outgroups      Oryza_punctata,Oryza_brachyantha,Leersia_perrieri
"""
import argparse, pickle


def has_valid_ortholog(v, gene, thr):
    """Return True only for a REAL ortholog entry.

    v = OrthoCluster[gene][sp].  Real ortholog requires a 5-element list with a
    positive Bitscore that also meets the article's per-gene RBH threshold.
    Zero-bitscore sentinels ([0,0,0,1,gene]) and None/[] return False.
    """
    if not isinstance(v, list) or len(v) < 5:
        return False
    bs = v[0]
    if not bs or bs <= 0:
        return False
    if thr is not None:
        t = thr.get(gene)
        if t and bs < t:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orthocluster", required=True)
    ap.add_argument("--bitscore-thr", default="")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outgroups", default="Oryza_punctata,Oryza_brachyantha,Leersia_perrieri")
    args = ap.parse_args()

    with open(args.orthocluster, "rb") as f:
        OC = pickle.load(f)
    thr = None
    if args.bitscore_thr:
        with open(args.bitscore_thr, "rb") as f:
            thr = pickle.load(f)
    outgroups = [s.strip() for s in args.outgroups.split(",")]
    preds = [l.strip().split("\t")[0] for l in open(args.pred) if l.strip()]

    n_orphan = 0
    n_partial = 0
    per_sp = {sp: 0 for sp in outgroups}
    for g in preds:
        oc = OC.get(g, {})
        all_na = True
        for sp in outgroups:
            if has_valid_ortholog(oc.get(sp), g, thr):
                all_na = False
            else:
                per_sp[sp] += 1
        if all_na:
            n_orphan += 1
        else:
            n_partial += 1

    n = len(preds)
    print(f"pred_denovo genes: {n}")
    print(f"  orphan by article RBH (all 3 outgroups no valid syntenic ortholog): {n_orphan} ({100*n_orphan/n:.1f}%)")
    print(f"  NOT fully orphan (valid ortholog in >=1 outgroup):                 {n_partial} ({100*n_partial/n:.1f}%)")
    print("  per-outgroup genes with NO valid syntenic RBH ortholog:")
    for sp in outgroups:
        print(f"    {sp}: {per_sp[sp]} ({100*per_sp[sp]/n:.1f}%)")


if __name__ == "__main__":
    main()
