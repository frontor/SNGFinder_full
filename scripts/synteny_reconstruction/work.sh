#python r2.3_verify_orphan.py --orthocluster OryzaSp11_ref_Oryza_japonica_OrthoCluster.pickle --pred pred_denovo.txt
#python r2.3_pred_denovo_synteny.py \
#  --orthocluster OryzaSp11_ref_Oryza_japonica_OrthoCluster.pickle \
#  --blockd        OryzaSp11_ref_Oryza_japonica_Blockd.pickle \
#  --syntenicd     OryzaSp11_ref_Oryza_japonica_Syntenicd.pickle \
#  --pred pred_denovo.txt --gff-dir ../OUTGFF_DIR \
#  --max-genes 50 --out smoke50
#python r2.3_pred_denovo_synteny.py \
#  --orthocluster OryzaSp11_ref_Oryza_japonica_OrthoCluster.pickle \
#  --blockd       OryzaSp11_ref_Oryza_japonica_Blockd.pickle \
#  --syntenicd    OryzaSp11_ref_Oryza_japonica_Syntenicd.pickle \
#  --pred pred_denovo.txt --gff-dir ../OUTGFF_DIR \
#  --bitscore-thr OryzaSp11_ref_Oryza_japonica_BitScore_threshold.pickle \
#  --max-genes 0 --out results_locus
#wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/231/095/GCF_000231095.1_Oryza_brachyantha.v1.4b/GCF_000231095.1_Oryza_brachyantha.v1.4b_assembly_report.txt
#wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/231/095/GCF_000231095.1_Oryza_brachyantha.v1.4b/GCF_000231095.1_Oryza_brachyantha.v1.4b_genomic.gff.gz
python scaffold_refseq_sensitivity.py \
  --blockd OryzaSp11_ref_Oryza_japonica_Blockd.pickle \
  --syntenicd OryzaSp11_ref_Oryza_japonica_Syntenicd.pickle \
  --pred pred_denovo.txt \
  --classes results_locus/r2.3_locus_classification.tsv \
  --assembly-report GCF_000231095.1_Oryza_brachyantha.v1.4b_assembly_report.txt \
  --refseq-gff GCF_000231095.1_Oryza_brachyantha.v1.4b_genomic.gff.gz \
  --out results_locus/r2.3_scaffold_refseq_sensitivity.tsv
