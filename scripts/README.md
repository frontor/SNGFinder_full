# Supplementary analysis scripts

Article-specific downstream analyses built on the SNGFinder orphan set and the
de-novo classifier output. These are research scripts (not a general pipeline)
and depend on intermediate datasets produced by the SNGFinder pipeline / classifier and other public softwares or datasets.

## threeway2.py
Three-way outgroup-homology audit of the orphan set. Classifies each orphan into
`de_novo_noncoding` / `independent_occupier` / `oryza_clade` / `old_preOryza`.
Required inputs (set via `SNGFINDER_DATA`):
- `New folder1/` — species pickles of gene models / homolog hits
- `threeway_outputs/` — output dir (auto-created)

## strata_and_audit.py
Origination-stratum analysis of the `oryza_clade` orphans by phylogenetic depth
(Stein et al. 2018 Nat Genet 50:285-296 node ages).
Required inputs (set via `SNGFINDER_DATA`):
- `threeway_outputs/orphan_origins_v2.tsv`
- `synteny_outputs/orphan_block_state_precise.tsv`
- `Orphan_Dataset.csv`
- `diagnosis_outputs/orphan_proba_full.npy`

## synteny_reconstruction/

Syntenic reconstruction of ancestral loci for the predicted de novo genes
(manuscript R2.3). Reconstructs each candidate's outgroup syntenic interval and
asks whether it is noncoding / gene-free, using two independent checks (RBH
ortholog + outgroup GFF coding-gene overlap at the precise sub-locus). Emits a
3-tier classification (supported_direct / supported_ambiguous / no_synteny) plus a
RefSeq sensitivity screen for the O. brachyantha scaffold coverage gap.

Required external inputs (not deposited in this repository; the Gramene OGE
release and assembly versions used to derive them are listed in the Data
availability section of the manuscript):
- OryzaSp11_ref_Oryza_japonica_{Blockd,Syntenicd,OrthoCluster,BitScore_threshold}.pickle
- outgroup GFFs: Oryza_punctata.gff3 / Oryza_brachyantha.gff3 / Leersia_perrieri.gff3
- NCBI RefSeq of GCF_000231095.1 (assembly_report.txt + genomic.gff.gz) for the screen

## table_S4/ — Supplementary Table S4 regeneration (t3.py chain)

End-to-end regeneration of Supplementary Table S4 (996 two-sided
Mann–Whitney tests). Includes `t3.py`, `GetTag.py` + tag lists, the 8 input
files regenerated from raw data (6 DNA-methylation window tables, 2 FUBAR ω
tables), the reference output `utest.xls`, and a link to the Zenodo data
bundle holding the remaining 22 feature/avscore/omics tables. See
`table_S4/README.md` for the verified reproduction statistics.

## stringent_model/ — stringent-positive-set retraining

Reproduces the 167-gene (set 1) retraining analysis: 10-fold CV AP 0.8973,
test AP 0.9229, 2,877 / 4,241 (67.84%) predicted de novo among
high-confidence calls. See `stringent_model/README.md`.

## bootstrap/ — de novo fraction bootstrap CI

`denovo_fraction_bootstrap.py`: 20,000-resample bootstrap CI of the de novo
fraction (standard set: 78.83–81.04%). See `bootstrap/README.md`.

## figure3/ — Figure 3 / Supplementary Fig. S6 scripts

ω boxplots (`t_omega_boxplot.py`, gene-tree and species-tree variants) and
feature-comparison scripts. See `figure3/README.md`.

Other manuscript downstream analyses are omitted here:
they each depend on additional external tools (and datasets) and ad-hoc intermediate inputs and were not engineered as a standalone pipeline; available on request.
