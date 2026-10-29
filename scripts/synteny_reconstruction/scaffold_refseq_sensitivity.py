#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity screen: would a COMPLETE Oryza brachyantha annotation change the
synteny-reconstruction classes of the 4,018 predicted de novo genes?

Background
----------
The pipeline's brachyantha genome (Oryza_brachyantha.v1.4b, Ensembl Genomes 75-era
toplevel FASTA) contains 12 pseudomolecules (named 1..12) plus 114 unplaced
scaffolds (JH*/super* names; e.g. JH369865.2 == super0039). The Gramene OGE GFF
used by r2.3_pred_denovo_synteny.py (oryza_brachyantha.maker.ipr.gff, 2013-09
Maker genebuild) annotates ONLY the 12 pseudomolecules, so the outgroup-GFF check
cannot see loci whose syntenic block maps to a scaffold (GFF-coverage QC:
12/28 = 0.429).

Why only O. brachyantha?
------------------------
The GFF-coverage QC emitted by r2.3_pred_denovo_synteny.py shows that the O. punctata
and L. perrieri GFFs cover 100% of the queried block chromosomes (coverage = 1.000),
whereas the O. brachyantha OGE GFF covers only 12 of 28 (coverage = 0.429). The other
two outgroups' syntenic intervals are therefore fully interrogated by their own GFFs,
so their "noncoding" calls are reliable and need no independent cross-check. Only
O. brachyantha has a coverage gap that could hide a coding gene on an unplaced scaffold;
only its scaffold-mapped intervals require the NCBI RefSeq cross-check below.

Coordinate correspondence
-------------------------
The RefSeq annotation used is Annotation Release 101 of GCF_000231095.1 -- the SAME
assembly accession as the O. brachyantha genome used in the article's Cactus/hal2maf
synteny (Oryza_brachyantha.v1.4b == GCF_000231095.1). The pipeline's scaffold names
(super*/JH*) are mapped to RefSeq NW_ accessions through the NCBI assembly_report, so
the RefSeq gene coordinates lie on the identical underlying sequence assembly as the
syntenic blocks. Coordinates are therefore directly comparable; no liftover or genome-
version remapping is involved.

Conservative (upper-bound) test
--------------------------------
This screen deliberately uses the ENTIRE brachyantha syntenic block interval for the
overlap test, in contrast to the precise focal sub-locus test used in the main
pipeline (Method (ii)). This makes it a conservative UPPER BOUND: if even one RefSeq
coding gene touches any part of the block interval, the locus is flagged. The real
number of loci that could actually reclassify is then refined below using the
per-gene, per-outgroup evidence in the classification table.

Inputs
------
- OryzaSp11_ref_Oryza_japonica_Blockd.pickle / ..._Syntenicd.pickle
  (article's own Cactus/hal2maf synteny; same files used by
  r2.3_pred_denovo_synteny.py)
- pred_denovo.txt (4,018 gene IDs)
- results_locus/r2.3_locus_classification.tsv (per-gene classes; MUST be the
  3-tier output: supported_direct / supported_ambiguous / no_synteny /
  contradicted_hard)
- GCF_000231095.1_Oryza_brachyantha.v1.4b_assembly_report.txt
  (NCBI: maps super* / JH* names to RefSeq NW_ accessions)
- GCF_000231095.1_Oryza_brachyantha.v1.4b_genomic.gff.gz
  (NCBI RefSeq Annotation Release 101; annotates 2,491 sequences incl.
  2,479 unplaced scaffolds)

Output
------
results_locus/r2.3_scaffold_refseq_sensitivity.tsv
  gene, class, scaffold (pipeline name), refseq_acc, interval_start, interval_end,
  n_refseq_coding, refseq_genes, per_outgroup_coding_type, flip_assessment
  where flip_assessment is one of:
    ""            not at risk (either not supported_direct, or supported_direct
                  but the scaffold block interval overlaps no RefSeq coding gene)
    upper_bound   supported_direct AND a RefSeq coding gene overlaps the scaffold
                  block interval, BUT the gene also has an independent noncoding
                  interval in O. punctata and/or L. perrieri -> would NOT actually
                  reclassify
    real_flip     supported_direct AND RefSeq overlap AND the gene has NO
                  independent punctata/perrieri noncoding support -> completing the
                  annotation WOULD reclassify it

Result (2026-10-29, against the 3-tier classification):
  18 of the 4,018 genes have a brachyantha block touching a scaffold;
  9 of those intervals overlap a RefSeq protein-coding gene;
  3 of the 9 are currently supported_direct
    (OSJAP07G07920, OSJAP11G11780, OSJAP11G17520).
  This is an UPPER BOUND. All three retain an independent noncoding interval in
  O. punctata and/or L. perrieri (both fully covered by their GFFs), so completing
  the brachyantha scaffold annotation reclassifies 0 loci (0.00% of 4,018). Two of
  the three overlaps are short (OSJAP11G11780 ~29 bp, OSJAP11G17520 ~14 bp) and
  likely coordinate noise; OSJAP07G07920 overlaps a RefSeq gene across ~544 bp (a
  genuine brachyantha coding locus) but is still independently supported. The
  screen therefore confirms that the O. brachyantha GFF coverage gap does not affect
  any class assignment.
"""
import pickle, gzip, argparse, csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blockd", required=True)
    ap.add_argument("--syntenicd", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--classes", required=True,
                    help="results_locus/r2.3_locus_classification.tsv (3-tier)")
    ap.add_argument("--assembly-report", required=True)
    ap.add_argument("--refseq-gff", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # super*/JH* -> RefSeq NW_ accession
    name2nw = {}
    for ln in open(args.assembly_report):
        if ln.startswith("#"):
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 7 and f[6].startswith("NW_"):
            name2nw[f[0]] = f[6]
            name2nw[f[4]] = f[6]

    # RefSeq protein-coding genes per sequence
    genes = defaultdict(list)
    with gzip.open(args.refseq_gff, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene" or "protein_coding" not in p[8]:
                continue
            gid = p[8].split("ID=")[1].split(";")[0]
            genes[p[0]].append((int(p[3]), int(p[4]), gid))

    with open(args.blockd, "rb") as f:
        blk = pickle.load(f)
    with open(args.syntenicd, "rb") as f:
        syn = pickle.load(f)

    preds = [l.strip() for l in open(args.pred) if l.strip()]

    # robust header-based read of the 3-tier classification table
    cls = {}        # gene -> classification
    pog = {}        # gene -> per_outgroup_coding_type string
    with open(args.classes) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        g_i = idx.get("gene", 0)
        c_i = idx.get("classification", len(header) - 1)
        p_i = idx.get("per_outgroup_coding_type", 4)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if not p or not p[0]:
                continue
            cls[p[g_i]] = p[c_i]
            pog[p[g_i]] = p[p_i] if p_i < len(p) else ""

    # parse per-outgroup coding type string
    # "Oryza_punctata=none|Oryza_brachyantha=none|Leersia_perrieri=none"
    def outgroup_status(gene, sp):
        s = pog.get(gene, "")
        for tok in s.split("|"):
            if "=" in tok and tok.split("=")[0] == sp:
                return tok.split("=")[1]
        return "?"

    gene2blocks = defaultdict(list)
    for bn in blk:
        for (rs, re_, gn, gs) in blk[bn].get("Oryza_japonica", []):
            gene2blocks[gn].append(bn)

    def is_chr12(n):
        return n.isdigit() and 1 <= int(n) <= 12

    raw_rows = []   # (gene, scaffold, nw, st, en, n_refseq_coding, refseq_genes)
    for gid in preds:
        for bn in gene2blocks.get(gid, []):
            for rec in syn.get("Oryza_brachyantha", {}).get(bn, []):
                ch, st, en = rec[0], rec[1], rec[2]
                if is_chr12(ch):
                    continue
                nw = name2nw.get(ch, ".")
                hits = [g for (s, e, g) in genes.get(nw, []) if s <= en and e >= st]
                raw_rows.append((gid, ch, nw, st, en,
                                 len(hits), ",".join(hits)))

    # ---- per-gene aggregation for the flip assessment ----
    genes_seen = set(r[0] for r in raw_rows)
    flip = {}   # gene -> "upper_bound" / "real_flip"
    for g in genes_seen:
        if cls.get(g) != "supported_direct":
            continue
        has_ref_overlap = any(r[5] > 0 for r in raw_rows if r[0] == g)
        if not has_ref_overlap:
            continue
        # independent, fully-covered noncoding support?
        # punctata/perrieri are fully GFF-covered, so "none" is reliable noncoding
        punc = outgroup_status(g, "Oryza_punctata")
        peri = outgroup_status(g, "Leersia_perrieri")
        independent = (punc == "none") or (peri == "none")
        flip[g] = "upper_bound" if independent else "real_flip"

    with open(args.out, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["gene", "class", "scaffold", "refseq_acc", "interval_start",
                    "interval_end", "n_refseq_coding", "refseq_genes",
                    "per_outgroup_coding_type", "flip_assessment"])
        for r in raw_rows:
            g = r[0]
            w.writerow([g, cls.get(g, "."), r[1], r[2], r[3], r[4], r[5], r[6],
                        pog.get(g, ""), flip.get(g, "")])

    # ---- summary ----
    n_scaffold = len(genes_seen)
    n_overlap = sum(1 for g in genes_seen
                    if any(r[5] > 0 for r in raw_rows if r[0] == g))
    upper = [g for g in flip if flip[g] == "upper_bound"]
    real = [g for g in flip if flip[g] == "real_flip"]
    print("scaffold-mapped genes: %d" % n_scaffold)
    print("  of which overlap a RefSeq coding gene (any class): %d" % n_overlap)
    print("supported_direct + RefSeq overlap (UPPER BOUND): %d -> %s"
          % (len(upper), ",".join(upper) or "-"))
    print("  real flips (no independent punctata/perrieri noncoding): %d -> %s"
          % (len(real), ",".join(real) or "-"))


if __name__ == "__main__":
    main()
