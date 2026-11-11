#!/usr/bin/env python
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import GetTag
df_cl=GetTag.GetGeneTag()
df_omega=pandas.read_csv("Jap_nochimeric_geneTree_select_omega.xls",sep="\t",index_col=0,header=0)
#df_features=pandas.concat([df_omega,df_cl],axis=1)
df_features=pandas.merge(df_omega, df_cl, left_index=True, right_index=True, how="left")
print(df_features)
rl=[]
TagLabel="GeneClass"
for index,row in df_features.iterrows():
    d=row.dropna().to_dict()
    if not TagLabel in d:
        continue
    ftag=d[TagLabel]
    for s in d:
        if s==TagLabel:
            continue
        rl.append((d[s],ftag))
df_final=pandas.DataFrame(rl,columns=["Omega","Class"])
#fig, ax=plt.subplots(dpi=300)
cl=[("NewDup","OldDup"),("NewCnv","OldCnv")]
for ci in cl:
    sns.set_context("paper")
    sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["legend.fontsize"] = 10
    my_pal = GetTag.GetPattle()
    a=sns.kdeplot(data=df_final, x="Omega", hue="Class",hue_order=ci,fill=True,common_norm=False, palette=my_pal,alpha=.6, linewidth=1,) #Set3
    sns.despine(ax=a)
    a.spines['left'].set_color('black')
    a.spines['bottom'].set_color('black')
    a.set_xlabel("Omega",fontsize=15)
    a.set_ylabel("Density",fontsize=15)
    a.tick_params(axis='both', labelsize=10)
    a.get_legend().set_title("")
    fa=a.get_figure()
    fa.tight_layout()
    fa.savefig("omega_%s_density.svg"%ci[0],format="svg",dpi=300)
    fa.savefig("omega_%s_density.png"%ci[0],format="png",dpi=300)
    #plt.show()
    fa.clf()
    plt.close(fa)
