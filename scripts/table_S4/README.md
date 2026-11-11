# Table S4 reproduction (t3.py chain)

Regenerates Supplementary Table S4 (996 pairwise two-sided Mann–Whitney U
tests across gene classes) end-to-end from the deposited inputs.

## Contents

| Item | Description |
|---|---|
| `t3.py` | the S4 generation script (verbatim from the analysis environment) |
| `GetTag.py` + `tags/*.txt` | gene-class label assignment (7 gene lists) |
| `inputs_regenerated/` | 8 input files regenerated from raw data in this study (6 DNA-methylation window tables + 2 FUBAR ω tables) |
| `utest.xls` | reference output (996 rows; rerunning `t3.py` overwrites it in place) |
| `DATA_BUNDLE_URL.md` | link to the Zenodo deposit holding the remaining 22 input tables (~171 MB) |

## How to run

1. Download the Zenodo data bundle (see `DATA_BUNDLE_URL.md`) and unpack the
   22 feature/avscore/omics tables into this directory.
2. Copy `inputs_regenerated/*` and `tags/*` into this directory as well
   (they must sit next to `t3.py`, which uses relative paths).
3. Run:

```bash
python t3.py
```

This writes `utest.xls` (996 rows × 4 columns: Tag, Features, Alternative,
Pvalues). The published Table S4 additionally contains Benjamini–Hochberg
adjusted values (Column E), computed across all 996 tests.

## Verified reproduction (2026-11-11)

Tested environment: Python 3.11.18, numpy 1.26.4, pandas 1.5.3,
scipy 1.17.1 (Linux, 64-bit). Against the published Table S4:

- 996 / 996 rows with identical significance calls (P < 0.05)
- 958 / 996 P values bit-identical; 989 / 996 within 1e-4 relative error
- the 66 differing rows involve the 8 regenerated input files
  (54 DNA-methylation tests + 12 ω tests); median relative error 3.7e-10,
  maximum 2.7e-3
- BH-adjusted column recomputed accordingly; 784 tests with unadjusted
  P < 0.05, 776 significant at q < 0.05 (unchanged)

## Input provenance

- `*_DNAmeth.xls`: mean of 100·m/(m+u) over all context (CG+CHG+CHH)
  cytosine sites with coverage ≥ 1 within each window, recomputed from
  Bismark CX reports (WGBS raw data SRR10763664–67; tissues: FlagLeaf,
  Panicle, Root, YoungLeaf). Window intervals are 1 bp point intervals.
- `Jap_nochimeric_{geneTree,speciesTree}_select_omega.xls`: FUBAR
  (HyPhy 2.5.62) ω = sum(β)/sum(α) per gene over the codon alignment of the
  longest protein transcript per gene, computed under the gene-tree and
  species-tree topologies respectively.
- Feature/avscore/omics tables (Zenodo bundle): per-gene sequence,
  structural, expression, R-loop and epigenomic window features as used by
  the original analysis environment.

## Number of tests and the duplicated rows

`t3.py` prints 996 rows, of which **936 are independent tests**: the
comparisons `NewDup vs. Random` and `NewCnv vs. Random` are each evaluated
twice because the random control set is applied in two groupings
(`("Denovo","NewDup","NewCnv","Orphan","OldSingleton")` and the two
class-vs-random pairs). The duplicate rows are byte-identical. BH correction
is computed over all 996 rows, which is slightly conservative; removing the
60 duplicates does not change any significance call in the paper
(verified independently).

## Prior de novo reference set: 343 genes vs 324 positives

`tags/denovo_343.txt` holds the 343 genes of the prior de novo reference set
used for the `PriorDenovo` class in Table S4. Of these, 324 were classified as
orphan genes in the present study and are the positives used to train and test
the classifier (`Training_Dataset.csv`, `Test_Dataset.csv`); the remaining 19
were assigned to other classes in this study (one duplicated gene, the rest
unclassified) and are therefore excluded from the classifier positives. Table
S4 uses all 343 genes for the `PriorDenovo` class; the classifier uses the
324-orphan subset. The two numbers differ only by these 19 genes.

## Statistical test variant

All P values in Table S4 are two-sided Mann-Whitney U statistics computed with
`scipy.stats.mannwhitneyu` using that function's default method selection
(exact for small samples without ties, otherwise the normal approximation with
continuity correction). A hand-rolled exact test or a different tie-handling
rule will not reproduce the table to the last digit.
