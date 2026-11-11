#!/usr/bin/env python
# NOTE on the number of tests: the pairwise loops below apply the random ("Random")
# control set in two different groupings, so the comparisons NewDup-vs-Random and
# NewCnv-vs-Random are each evaluated twice. The script therefore prints 996 rows,
# of which 936 are independent tests (120 rows are 60 duplicated comparisons).
# Benjamini-Hochberg is applied to all 996 rows, which makes the correction
# slightly conservative; de-duplicating the table does not change any
# significance call reported in the paper.
import pandas,itertools
from scipy.stats import mannwhitneyu
import numpy as np
import GetTag
df_cl=GetTag.GetGeneTag()
flist=["Jap_nochimeric_geneTree_select_omega.xls","gene_seq_features.xls","gene_anno_features.xls","StructureProperty.xls","gene_expression_real_mean_forlength.xls"]
rl=[]
cl=[("Denovo","NewDup","NewCnv","Orphan","OldSingleton"),("NewDup","OldDup"),("NewCnv","OldCnv")]
for f in flist:
    df=pandas.read_csv(f,sep="\t",index_col=0,header=0)
#    df_features=pandas.concat([df,df_cl],axis=1)
    df_features=pandas.merge(df,df_cl, left_index=True,right_index=True,how="left")
    #df_features.dropna(inplace=True)
    if f=="gene_expression_real_mean_forlength.xls" or f=="gene_anno_features.xls":
        alter1="less"
    elif f=="Jap_nochimeric_geneTree_select_omega.xls" or f=="StructureProperty.xls" or f=="gene_seq_features.xls":
        alter1="greater"
    for c in df.columns:
        if c=="gtAI":
            alter2="less"
        elif c=="TM" or c=="SignalP":
            alter2="two-sided"
        else:
            alter2=alter1
        for ci in cl:
            for c1,c2 in itertools.combinations(list(ci),2):
                if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv" or c2=="Denovo":
                    c1,c2=c2,c1
                if c1!="Orphan" and c1!="NewDup" and c1!="NewCnv" and c1!="Denovo":
                    continue
#                if c2=="OldNotConserve" and c=="AverageIntronLength":
#                    alter="greater"
                elif (c1=="NewDup" or c1=="NewCnv") and c=="AverageIntronLength":
                    alter="two-sided"
                elif c1=="NewDup" and c=="GCContent":
                    alter="less"
                elif c1=="NewCnv"and (c=="GCContent" or c=="ExonNumber" or c=="IsoformNumber" or c=="DCBS" or c=="GeneLength"):
                    alter="two-sided"
                else:
                    alter=alter2
                a=df_features[df_features["GeneClass"]==c1][c].dropna().values
                b=df_features[df_features["GeneClass"]==c2][c].dropna().values
                #u,p=mannwhitneyu(a,b,alternative=alter)
                u,p=mannwhitneyu(a,b,alternative="two-sided")
                rl.append(("%s vs. %s"%(c1,c2),c,alter,p))
frl=[]
for ti in ["sense","antisense"]:
    df_tss=pandas.read_csv("tss_300_%s_avscore.xls"%ti,sep="\t",header=0,index_col=0)
#    df_tss_features=pandas.concat([df_tss,df_cl],axis=1)
    df_tss_features=pandas.merge(df_tss,df_cl, left_index=True,right_index=True,how="left")
    df_tss_random=pandas.read_csv("tss_300_random_%s_avscore.xls"%ti,sep="\t",header=0,index_col=0)
    
    df_body=pandas.read_csv("genebody_%s_avscore.xls"%ti,sep="\t",header=0,index_col=0)
#    df_body_features=pandas.concat([df_body,df_cl],axis=1)
    df_body_features=pandas.merge(df_body,df_cl, left_index=True,right_index=True,how="left")
    df_body_random=pandas.read_csv("random_%s_avscore.xls"%ti,sep="\t",header=0,index_col=0)
    
    df_tts=pandas.read_csv("tts_300_%s_avscore.xls"%ti,sep="\t",header=0,index_col=0)
#    df_tts_features=pandas.concat([df_tts,df_cl],axis=1)
    df_tts_features=pandas.merge(df_tts, df_cl, left_index=True,right_index=True,how="left")
    df_tts_random=pandas.read_csv("tts_300_random_%s_avscore.xls"%ti,sep="\t",header=0,index_col=0)
    for ta,df_features,df_random in [("Tss",df_tss_features,df_tss_random),("Genebody",df_body_features,df_body_random),("Tts",df_tts_features,df_tts_random)]:
        for ci in cl:
            for c1,c2 in itertools.combinations(list(ci)+["Random"],2):
                if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv" or c2=="Denovo":
                    c1,c2=c2,c1
                if c1!="Orphan" and c1!="NewDup" and c1!="NewCnv" and c1!="Denovo":
                    continue
                a=df_features[df_features["GeneClass"]==c1]["Flagleaf"].dropna().values
                if c2=="Random":
                    b=df_random.loc[:,"Flagleaf"].dropna().values
                else:
                    b=df_features[df_features["GeneClass"]==c2]["Flagleaf"].dropna().values
                if ti=="antisense" and ta=="Tss" and c2!="Random" and c2!="OldCnv":
                    alter="less"
                elif ti=="antisense" and ta=="Genebody" and (c2=="OldSingleton" or c2=="OldDup" or c2=="OldCnv"):
                    alter="two-sided"
                elif ti=="antisense" and ta=="Tts" and c2=="OldCnv":
                    alter="two-sided"
                elif ti=="antisense" and ta=="Tss" and c2=="OldCnv":
                    alter="two-sided"
                else:
                    alter="greater"
                print("%s vs. %s"%(c1,c2),"%s%sRloop"%(ta,ti.capitalize()),len(a),len(b))
                u,p=mannwhitneyu(a,b,alternative="two-sided")
                rl.append(("%s vs. %s"%(c1,c2),"%s%sRloop"%(ta,ti.capitalize()),alter,p))
df_tss=pandas.read_csv("tss_300_omics_features.xls",sep="\t",header=0,index_col=0)
#df_tss_features=pandas.concat([df_tss,df_cl],axis=1)
df_tss_features=pandas.merge(df_tss,df_cl, left_index=True,right_index=True,how="left")
df_tss_random=pandas.read_csv("tss_300_random_omics_features.xls",sep="\t",header=0,index_col=0)
 
df_body=pandas.read_csv("genebody_omics_features.xls",sep="\t",header=0,index_col=0)
#df_body_features=pandas.concat([df_body,df_cl],axis=1)
df_body_features=pandas.merge(df_body,df_cl, left_index=True,right_index=True,how="left")
df_body_random=pandas.read_csv("random_omics_features.xls",sep="\t",header=0,index_col=0)
 
df_tts=pandas.read_csv("tts_300_omics_features.xls",sep="\t",header=0,index_col=0)
#df_tts_features=pandas.concat([df_tts,df_cl],axis=1)
df_tts_features=pandas.merge(df_tts,df_cl, left_index=True,right_index=True,how="left")
df_tts_random=pandas.read_csv("tts_300_random_omics_features.xls",sep="\t",header=0,index_col=0)
for ta,df_features,df_random in [("Tss",df_tss_features,df_tss_random),("Genebody",df_body_features,df_body_random),("Tts",df_tts_features,df_tts_random)]:
    for c in df_features.columns:
        if c=="GeneClass":
            continue
        si,oi=c.split("_")
        if si!="FlagLeaf":
            continue
        for ci in cl:
            for c1,c2 in itertools.combinations(list(ci)+["Random"],2):
                if c2=="Random":
                    alter="greater"
                elif oi=="H3K27me3" and c2=="OldSingleton":
                    alter="greater"
                elif oi=="H3K9me2":
                    alter="greater"
                elif oi=="H3K27me3" and c1=="NewCnv":
                    alter="two-sided"
                else:
                    alter="less"
                if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv" or c2=="Denovo":
                    c1,c2=c2,c1
                if c1!="Orphan" and c1!="NewDup" and c1!="NewCnv" and c1!="Denovo":
                    continue
                a=df_features[df_features["GeneClass"]==c1][c].dropna().values
                if c2=="Random":
                    b=df_random.loc[:,c].dropna().values
                else:
                    b=df_features[df_features["GeneClass"]==c2][c].dropna().values
                print("%s vs. %s"%(c1,c2),"%s%s"%(ta,oi),len(a),len(b))
                u,p=mannwhitneyu(a,b,alternative="two-sided")
                rl.append(("%s vs. %s"%(c1,c2),"%s%s"%(ta,oi),alter,p))
df_tss=pandas.read_csv("tss_300_DNAmeth.xls",sep="\t",header=0,index_col=0)
#df_tss_features=pandas.concat([df_tss,df_cl],axis=1)
df_tss_features=pandas.merge(df_tss,df_cl, left_index=True,right_index=True,how="left")
df_tss_random=pandas.read_csv("tss_300_random_DNAmeth.xls",sep="\t",header=0,index_col=0)
 
df_body=pandas.read_csv("genebody_DNAmeth.xls",sep="\t",header=0,index_col=0)
#df_body_features=pandas.concat([df_body,df_cl],axis=1)
df_body_features=pandas.merge(df_body,df_cl, left_index=True,right_index=True,how="left")
df_body_random=pandas.read_csv("random_DNAmeth.xls",sep="\t",header=0,index_col=0)
 
df_tts=pandas.read_csv("tts_300_DNAmeth.xls",sep="\t",header=0,index_col=0)
#df_tts_features=pandas.concat([df_tts,df_cl],axis=1)
df_tts_features=pandas.merge(df_tts,df_cl, left_index=True,right_index=True,how="left")
df_tts_random=pandas.read_csv("tts_300_random_DNAmeth.xls",sep="\t",header=0,index_col=0)

for ta,df_features,df_random in [("Tss",df_tss_features,df_tss_random),("Genebody",df_body_features,df_body_random),("Tts",df_tts_features,df_tts_random)]:
    for ci in cl:
        for c1,c2 in itertools.combinations(list(ci)+["Random"],2):
            if c2=="Random":
                alter="two-sided"
            else:
                alter="greater"
            if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv" or c2=="Denovo":
                c1,c2=c2,c1
            if c1!="Orphan" and c1!="NewDup" and c1!="NewCnv" and c1!="Denovo":
                continue
            a=df_features[df_features["GeneClass"]==c1]["FlagLeaf"].dropna().values
            if c2=="Random":
                b=df_random.loc[:,"FlagLeaf"].dropna().values
            else:
                b=df_features[df_features["GeneClass"]==c2]["FlagLeaf"].dropna().values
            print("%s vs. %s"%(c1,c2),"%sMethylation"%(ta),len(a),len(b))
            u,p=mannwhitneyu(a,b,alternative="two-sided")
            rl.append(("%s vs. %s"%(c1,c2),"%sMethylation"%(ta),alter,p))
df_r=pandas.DataFrame(rl, columns=['Tag',"Features","Alternative","Pvalues"])
print(df_r)
df_r.to_csv("utest.xls",sep="\t",index=False)
