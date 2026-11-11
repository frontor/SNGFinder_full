# Figure 3 / Supplementary Fig. S6 analysis scripts

The per-feature statistical comparisons and ω boxplots of Figure 3 and
Supplementary Figure 6 (and the underlying Supplementary Table S4) are
produced by these scripts, verbatim from the analysis environment.

- `t_omega_boxplot.py` — ω boxplots for the three class sets
  (main Fig. 3B; Supplementary Fig. 6B upper/lower). Run once per topology:

```bash
python t_omega_boxplot.py Jap_nochimeric_geneTree_select_omega.xls omega_geneTree
python t_omega_boxplot.py Jap_nochimeric_speciesTree_select_omega.xls omega_speciesTree
```

- `t_features.py`, `t_structure2.py`, `t_rna*.py` — feature-level
  statistical comparisons feeding Figure 3C–E.
- `work.sh` — the original invocation order.

## Dependencies

python (pandas, scipy), seaborn, statannotations. `GetTag.py` and the
gene-class tag lists are provided under `../table_S4/`; the two ω tables are
provided under `../table_S4/inputs_regenerated/`; remaining feature tables
(StructureProperty, avscore, omics, expression) are in the Zenodo data
bundle linked from `../table_S4/DATA_BUNDLE_URL.md`. All inputs are read
from the working directory (relative paths).
