#!/usr/bin/env python
#coding=utf-8
"""
R2.3 Prong 1 (revised): Direct syntenic reconstruction of ancestral noncoding regions
for the ML-predicted de novo (pred_denovo) gene subset, using the ARTICLE'S OWN
Cactus/hal2maf synteny (Blockd + Syntenicd) -- NOT a new method (Fagin/Satsuma2).

For each pred_denovo gene:
  - locate its syntenic block(s) in japonica (RefSp) and in each of the 3 outgroups
    (Oryza_punctata, Oryza_brachyantha, Leersia_perrieri);
  - extract the outgroup genome interval(s) of the syntenic block;
  - classify the interval as coding vs noncoding using TWO independent checks:
      (a) OrthoCluster[gene][sp] holds a REAL syntenic RBH ortholog (article method).
          NOTE: OrthoCluster[gene][sp] can be a ZERO-BITSCORE sentinel
          [0,0,0,1,'OPUNCxx'] (block neighbour with NO BLAST homology) which must NOT
          count as ortholog; we require Bitscore>0 AND Bitscore>=GenePairsScored[gene]
          (the article's per-gene RBH threshold, loaded via --bitscore-thr).
      (b) outgroup GFF coding-gene overlap at the PRECISE gene sub-locus within the
          syntenic block (strand-independent relative frame) -> independent annotation
          check. Checking the sub-locus (not the whole block) avoids counting unrelated
          genes merely co-present in a multi-gene syntenic block.
  - emit per-gene/per-outgroup locus data for the locus-track figure + a tiered summary.

This is the POSITIVE counterpart to the article's negative orphan definition:
instead of only "no homolog found", we show the ancestral (outgroup) interval and
that it is noncoding/gene-free -- directly answering the reviewer's
"direct syntenic reconstruction of ancestral noncoding regions".

NOTE on circularity (methodological honesty):
  - The data structures come from the SAME synteny+RBH pipeline that defined orphan,
    so this is NOT an independent re-discovery of "no homolog".
  - The NEW, non-circular contribution is: (i) the ACTUAL ancestral interval
    COORDINATES in each outgroup, and (ii) the POSITIVE demonstration that this
    interval is noncoding/gene-free (cross-checked against the outgroup GFF), i.e.
    "the ancestral locus WAS noncoding DNA" rather than merely "we found no homolog".
  - This locus-track is therefore a VISUAL/SPATIAL corroboration, to be paired with
    the GenEra HDF real-absence test (abSENCE) and cross-method concordance
    (GenEra youngest 827 / Vakirlis 641) as the non-circular pillars.

Inputs:
  --orthocluster  OryzaSp11_ref_Oryza_japonica_OrthoCluster.pickle  (15 MB)
  --blockd        OryzaSp11_ref_Oryza_japonica_Blockd.pickle        (164 MB)
  --syntenicd     OryzaSp11_ref_Oryza_japonica_Syntenicd.pickle     (3.4 GB; needs ~8 GB RAM)
  --bitscore-thr  OryzaSp11_ref_Oryza_japonica_BitScore_threshold.pickle (GenePairsScored;
                  REQUIRED for the article-correct RBH ortholog test -- see NOTE (a) above)
  --pred          pred_denovo.txt  (one gene ID per line, no header)
  --outgroups     Oryza_punctata,Oryza_brachyantha,Leersia_perrieri
  --gff-dir       dir with <outgroup>.gff3 (coding-gene annotation for independent check)
  --gff-pattern   {sp}.gff3
  --refspecies    Oryza_japonica
  --max-genes     0 = all (use e.g. 50 for a smoke test first)
  --out           output directory
"""
import argparse, os, sys, json, pickle
from collections import defaultdict, Counter


def load_gff_coding(gff):
    """Return dict chr -> list of (start,end,gene_id,biotype) for protein-coding gene rows.

    Gene IDs and biotypes are captured so that a 'supported_ambiguous' call can be
    audited: we record WHICH outgroup gene/annotation overlaps the syntenic sub-locus.
    """
    d = defaultdict(list)
    if not gff or not os.path.exists(gff):
        return d
    with open(gff) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            if p[2] not in ("gene", "mRNA"):
                continue
            attr = p[8].lower()
            # keep only protein-coding; drop noncoding biotypes
            if any(k in attr for k in ("trna", "rrna", "snorna", "mirna", "ncrna",
                                        "noncoding", "pseudogene", "transposable")):
                continue
            try:
                s, e = int(p[3]), int(p[4])
            except ValueError:
                continue
            gid, biotype = "", ""
            for tok in p[8].replace(";", " ").split():
                if tok.startswith("ID="):
                    gid = tok[3:]; break
                if tok.startswith("gene_id="):
                    gid = tok[8:]; break
            for tok in p[8].replace(";", " ").split():
                if tok.startswith("gene_biotype="):
                    biotype = tok[13:]; break
                if tok.startswith("biotype="):
                    biotype = tok[8:]; break
            d[p[0]].append((s, e, gid, biotype))
    return d


def overlaps(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def has_valid_ortholog(v, gene, thr):
    """Article-correct test for a REAL outgroup RBH ortholog.

    OrthoCluster[gene][sp] can be:
      * None / [] / missing            -> no ortholog
      * [0,0,0,1,'OPUNCxx']            -> ZERO-BITSCORE sentinel (block neighbour with
                                          NO BLAST homology, Bitscore==0) -> NOT ortholog
      * [Bitscore,Pident,Nident,Qlen,'OPUNCxx'] (Bitscore>0) -> real ortholog, valid
                                          only if Bitscore >= GenePairsScored[gene]
                                          (the article's per-gene RBH threshold).
    The naive `bool(v)` test wrongly counts the zero-bitscore sentinel as ortholog.
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
    ap.add_argument("--blockd", required=True)
    ap.add_argument("--syntenicd", required=True)
    ap.add_argument("--bitscore-thr", default="",
                    help="OryzaSp11_ref_Oryza_japonica_BitScore_threshold.pickle "
                         "(GenePairsScored); required for the article-correct RBH "
                         "ortholog test (Bitscore must meet the per-gene threshold).")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--outgroups", default="Oryza_punctata,Oryza_brachyantha,Leersia_perrieri")
    ap.add_argument("--gff-dir", default="")
    ap.add_argument("--gff-pattern", default="{sp}.gff3")
    ap.add_argument("--refspecies", default="Oryza_japonica")
    ap.add_argument("--max-genes", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    RefSp = args.refspecies
    outgroups = [s.strip() for s in args.outgroups.split(",") if s.strip()]

    sys.stderr.write("[load] OrthoCluster ...\n")
    with open(args.orthocluster, "rb") as f:
        OrthoCluster = pickle.load(f)
    sys.stderr.write("[load] Blockd ...\n")
    with open(args.blockd, "rb") as f:
        Blockd = pickle.load(f)
    sys.stderr.write("[load] Syntenicd (this is large, ~3.4 GB) ...\n")
    with open(args.syntenicd, "rb") as f:
        Syntenicd = pickle.load(f)

    # GenePairsScored / per-gene RBH bitscore threshold (article's orphan criterion)
    GenePairsScored = None
    if args.bitscore_thr:
        sys.stderr.write("[load] BitScore threshold (GenePairsScored) ...\n")
        with open(args.bitscore_thr, "rb") as f:
            GenePairsScored = pickle.load(f)

    # GeneToBlock: RefSp gene -> list of (BlockName, RelpStart, RelpEnd, gStrand)
    sys.stderr.write("[build] GeneToBlock from Blockd ...\n")
    GeneToBlock = defaultdict(list)
    for bn in Blockd:
        for (rs, re_, gn, gs) in Blockd[bn].get(RefSp, []):
            GeneToBlock[gn].append((bn, rs, re_, gs))

    # outgroup GFFs (independent coding-gene annotation)
    gff_data = {}
    if args.gff_dir:
        for sp in outgroups:
            gf = os.path.join(args.gff_dir, args.gff_pattern.format(sp=sp))
            if os.path.exists(gf):
                gff_data[sp] = load_gff_coding(gf)
                sys.stderr.write(f"[load] GFF {sp}: {sum(len(v) for v in gff_data[sp].values())} coding intervals\n")
            else:
                sys.stderr.write(f"[warn] GFF missing for {sp}: {gf}\n")

    gff_chr = {sp: set(gff_data.get(sp, {}).keys()) for sp in outgroups}
    seen_chr = {sp: set() for sp in outgroups}

    preds = []
    with open(args.pred) as f:
        for ln in f:
            g = ln.strip().split("\t")[0]
            if g:
                preds.append(g)
    if args.max_genes:
        preds = preds[:args.max_genes]
    sys.stderr.write(f"[info] pred_denovo genes to process: {len(preds)}\n")

    summary = {sp: {"coding": 0, "noncoding": 0} for sp in outgroups}
    tsv_rows = []
    json_recs = []
    n_no_block = 0

    for gid in preds:
        blocks = GeneToBlock.get(gid, [])
        if not blocks:
            n_no_block += 1
        # japonica-side coords (for locus figure, left track)
        jap_intervals = []
        for (bn, rs, re_, gs) in blocks:
            for (ch, st, en, strand, seq) in Syntenicd.get(RefSp, {}).get(bn, []):
                jap_intervals.append((ch, st, en, strand))
        if jap_intervals:
            jc, js, je, jstr = jap_intervals[0]
        else:
            jc, js, je, jstr = (".", -1, -1, ".")
        rec = {"gene": gid, "jap": [jc, js, je, jstr], "outgroups": {}}

        for sp in outgroups:
            out_intervals = []
            for (bn, rs, re_, gs) in blocks:
                for (ch, st, en, strand, seq) in Syntenicd.get(sp, {}).get(bn, []):
                    out_intervals.append((ch, st, en, strand))
            if not out_intervals:
                rec["outgroups"][sp] = {"status": "no_synteny", "intervals": [], "ortholog": ""}
                continue
            # coding check (a): article RBH / synteny ortholog
            oc = OrthoCluster.get(gid, {}).get(sp)
            has_ortho = has_valid_ortholog(oc, gid, GenePairsScored)
            # independent GFF-coverage bookkeeping: which outgroup block chromosomes
            # were actually queried against the GFF (to QC for naming mismatch later)
            for (bn, rs, re_, gs) in blocks:
                for (ch, st, en, strand, seq) in Syntenicd.get(sp, {}).get(bn, []):
                    seen_chr[sp].add(ch)
            # coding check (b): independent GFF overlap at the PRECISE gene sub-locus
            # (relative frame within each syntenic block -> strand-independent). We only
            #  test the outgroup gene that is the SYNTENIC COUNTERPART of the current
            #  japonica gene (same relative position `rs,re_` inside the block), NOT every
            #  gene in the block -- otherwise a GFF gene at a neighbouring gene's locus
            #  would be wrongly attributed to this gene.
            gff_coding_hit = False
            gff_hit_details = []  # auditable evidence for supported_ambiguous
            for (bn, rs, re_, gs) in blocks:
                for (ch, st, en, strand, seq) in Syntenicd.get(sp, {}).get(bn, []):
                    for (gRelS, gRelE, gName, gStrand) in Blockd.get(bn, {}).get(sp, []):
                        if not overlaps(gRelS, gRelE, rs, re_):
                            continue  # not the counterpart of the current japonica gene
                        for (gs_, ge_, ggid, gbio) in gff_data.get(sp, {}).get(ch, []):
                            relS, relE = gs_ - st, ge_ - st
                            if overlaps(relS, relE, gRelS, gRelE):
                                gff_coding_hit = True
                                gff_hit_details.append({
                                    "block": bn, "chr": ch,
                                    "syntenic_counterpart_gene": gName,
                                    "gff_gene": ggid, "biotype": gbio,
                                    "gff_start": gs_, "gff_end": ge_,
                                })
                                break
                    if gff_coding_hit:
                        break
                if gff_coding_hit:
                    break
            # distinguish EVIDENCE STRENGTH: RBH ortholog (hard) vs GFF-only (ambiguous).
            # A GFF-only hit means a coding gene sits at the syntenic locus but is NOT the
            # RBH ortholog -> could be lineage-specific insertion, a BLAST-missed distant
            # ortholog, or coordinate noise. It must NOT be lumped with a real RBH ortholog.
            if has_ortho and gff_coding_hit:
                status, coding_type = "coding", "rbh+gff"
            elif has_ortho:
                status, coding_type = "coding", "rbh"
            elif gff_coding_hit:
                status, coding_type = "coding", "gff"
            else:
                status, coding_type = "noncoding", "none"
            summary[sp][status] += 1
            ortho_gene = oc[4] if (has_ortho and oc and len(oc) >= 5) else ""
            rec["outgroups"][sp] = {
                "status": status,
                "coding_type": coding_type,
                "intervals": [[c, s, e, strand] for (c, s, e, strand) in out_intervals],
                "ortholog": ortho_gene,
                "gff_hits": gff_hit_details,
            }

        # 3+1-tier classification that separates EVIDENCE STRENGTH:
        #   contradicted_hard  : >=1 outgroup has a real RBH ortholog (hard contradiction)
        #   supported_ambiguous: no RBH ortholog, but a non-orthologous coding gene sits
        #                        at the syntenic locus in >=1 outgroup (discuss, not contradiction)
        #   supported_direct   : no RBH, no GFF coding, AND >=1 outgroup has a noncoding
        #                        syntenic interval (direct ancestral-noncoding evidence)
        #   no_synteny         : no RBH, no GFF coding, AND no outgroup has a syntenic
        #                        counterpart (cannot reconstruct; NOT direct evidence).
        #   NOTE (2026-09-09 fix): the old code lumped no_synteny into supported_clean,
        #        inflating "ancestral noncoding" to 78.1%; the correct split is
        #        direct=34.1% / no_synteny=44.0% / ambiguous=21.9%.
        ctypes = [rec["outgroups"][sp].get("coding_type", "none") for sp in outgroups if sp in rec["outgroups"]]
        has_rbh = any(t in ("rbh", "rbh+gff") for t in ctypes)
        has_gffonly = any(t == "gff" for t in ctypes)
        has_noncoding = any(rec["outgroups"][sp].get("status") == "noncoding"
                            for sp in outgroups if sp in rec["outgroups"])
        if has_rbh:
            classification = "contradicted_hard"
        elif has_gffonly:
            classification = "supported_ambiguous"
        elif has_noncoding:
            classification = "supported_direct"
        else:
            classification = "no_synteny"
        rec["classification"] = classification
        json_recs.append(rec)
        tsv_rows.append((gid, jc, js, je,
                         "|".join(f"{sp}={rec['outgroups'].get(sp, {}).get('coding_type', '?')}"
                                  for sp in outgroups),
                         classification))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "r2.3_locus_classification.tsv"), "w") as f:
        f.write("gene\tjap_chr\tjap_start\tjap_end\tper_outgroup_coding_type\tclassification\n")
        for r in tsv_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(os.path.join(args.out, "r2.3_locus_data.json"), "w") as f:
        json.dump(json_recs, f, indent=1)

    # --- supported_ambiguous audit trail: one row per (gene, outgroup) GFF hit ---
    # This is the explicit "analysis basis" for the ambiguous tier: for every gene
    # flagged ambiguous we record exactly which outgroup, which independently-annotated
    # GFF gene (and its biotype) overlaps the syntenic counterpart sub-locus.
    amb_rows = []
    for r in json_recs:
        if r["classification"] != "supported_ambiguous":
            continue
        for sp in outgroups:
            o = r["outgroups"].get(sp, {})
            if o.get("coding_type") != "gff":
                continue
            for h in o.get("gff_hits", []):
                amb_rows.append((r["gene"], sp, h.get("syntenic_counterpart_gene", ""),
                                 h.get("gff_gene", ""), h.get("biotype", ""),
                                 h.get("chr", ""), h.get("gff_start", ""), h.get("gff_end", ""),
                                 h.get("block", "")))
    with open(os.path.join(args.out, "r2.3_ambiguous_detail.tsv"), "w") as f:
        f.write("gene\toutgroup\tarticle_syntenic_counterpart\tgff_gene\tbiotype\t"
                "chrom\tgff_start\tgff_end\tblock\n")
        for row in amb_rows:
            f.write("\t".join(str(x) for x in row) + "\n")

    n = len(json_recs)
    tier = {t: sum(1 for r in json_recs if r["classification"] == t)
            for t in ("contradicted_hard", "supported_ambiguous", "supported_direct", "no_synteny")}
    # RBH-orphan (article method): no RBH ortholog in ANY of the 3 outgroups
    rbh_orphan = n - tier["contradicted_hard"]
    with open(os.path.join(args.out, "r2.3_summary.tsv"), "w") as f:
        f.write(f"# examined_genes\t{n}\n")
        f.write(f"# genes_with_no_syntenic_block\t{n_no_block}\n")
        f.write(f"# RBH_orphan(article method: no RBH ortholog in any of 3 outgroups)\t{rbh_orphan}\t({100 * rbh_orphan / n:.1f}%)\n")
        f.write(f"# contradicted_hard(RBH ortholog in >=1 outgroup)\t{tier['contradicted_hard']}\t({100 * tier['contradicted_hard'] / n:.1f}%)\n")
        f.write(f"# supported_direct(no RBH, no GFF coding, >=1 outgroup noncoding interval)\t{tier['supported_direct']}\t({100 * tier['supported_direct'] / n:.1f}%)\n")
        f.write(f"# no_synteny(no RBH, no GFF coding, NO outgroup syntenic counterpart)\t{tier['no_synteny']}\t({100 * tier['no_synteny'] / n:.1f}%)\n")
        f.write(f"# supported_ambiguous(no RBH, but GFF coding in >=1 outgroup)\t{tier['supported_ambiguous']}\t({100 * tier['supported_ambiguous'] / n:.1f}%)\n\n")
        f.write("outgroup\tcoding(rbh)\tcoding(gff-only)\tnoncoding\n")
        for sp in outgroups:
            rbh_c = sum(1 for r in json_recs
                        if r["outgroups"].get(sp, {}).get("coding_type") in ("rbh", "rbh+gff"))
            gff_c = sum(1 for r in json_recs
                        if r["outgroups"].get(sp, {}).get("coding_type") == "gff")
            nc = sum(1 for r in json_recs
                     if r["outgroups"].get(sp, {}).get("coding_type") == "none")
            f.write(f"{sp}\t{rbh_c}\t{gff_c}\t{nc}\n")
        f.write("\n# GFF coverage QC (fraction of queried outgroup block-chromosomes present in the GFF)\n")
        for sp in outgroups:
            sc = seen_chr.get(sp, set())
            cov = (len(sc & gff_chr.get(sp, set())) / len(sc)) if sc else 0.0
            f.write(f"# {sp}: queried_chr={len(sc)} covered_chr={len(sc & gff_chr.get(sp, set()))} coverage={cov:.3f}\n")
        # --- supported_ambiguous analysis basis breakdown ---
        # The ambiguous tier is the one a reviewer will probe: a coding gene sits at the
        # syntenic locus per the INDEPENDENT outgroup GFF, yet no RBH ortholog exists.
        # We break it down by (i) which outgroup carries the hit, (ii) whether the hit is
        # confined to the CLOSEST outgroup (consistent with young-de-novo decay: the syntenic
        # site still occupied in the nearest relative, lost deeper) vs reaches DEEPER
        # outgroups (more likely annotation / coordinate noise), and (iii) the biotype of the
        # overlapping GFF gene (auditable via r2.3_ambiguous_detail.tsv).
        amb_genes = [r for r in json_recs if r["classification"] == "supported_ambiguous"]
        amb_per_og = {sp: 0 for sp in outgroups}
        amb_closest_only = amb_deeper = 0
        bio_counter = Counter()
        for r in amb_genes:
            hits_ogs = [sp for sp in outgroups
                        if r["outgroups"].get(sp, {}).get("coding_type") == "gff"]
            for sp in hits_ogs:
                amb_per_og[sp] += 1
            if hits_ogs and all(sp == outgroups[0] for sp in hits_ogs):
                amb_closest_only += 1
            if any(sp != outgroups[0] for sp in hits_ogs):
                amb_deeper += 1
            for sp in hits_ogs:
                for h in r["outgroups"][sp].get("gff_hits", []):
                    bio_counter[h.get("biotype", "") or "(unspecified)"] += 1
        f.write("\n# supported_ambiguous analysis basis (independent outgroup GFF coding hits)\n")
        for sp in outgroups:
            f.write(f"#   ambiguous genes with GFF hit in {sp}: {amb_per_og[sp]}\n")
        f.write(f"#   ambiguous w/ GFF hit ONLY in closest outgroup({outgroups[0]}): {amb_closest_only}\n")
        f.write(f"#   ambiguous w/ GFF hit in a DEEPER outgroup(brachyantha/perrieri): {amb_deeper}\n")
        f.write("#   overlapping GFF gene biotype distribution: " +
                ", ".join(f"{k}={v}" for k, v in bio_counter.most_common()) + "\n")

    print(f"[done] examined={n}")
    print(f"  RBH_orphan(article)={rbh_orphan} ({100*rbh_orphan/n:.1f}%)")
    print(f"  contradicted_hard={tier['contradicted_hard']} ({100*tier['contradicted_hard']/n:.1f}%)")
    print(f"  supported_direct={tier['supported_direct']} ({100*tier['supported_direct']/n:.1f}%)")
    print(f"  no_synteny={tier['no_synteny']} ({100*tier['no_synteny']/n:.1f}%)")
    print(f"  supported_ambiguous={tier['supported_ambiguous']} ({100*tier['supported_ambiguous']/n:.1f}%)")


if __name__ == "__main__":
    main()
