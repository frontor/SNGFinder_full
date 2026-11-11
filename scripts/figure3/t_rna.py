#!/usr/bin/env python
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import GetTag
df_cl=GetTag.GetGeneTag()
df_rna=pandas.read_csv("gene_TPM_mean.xls",sep="\t",header=0,index_col=0)
#df_rna=pandas.concat([df_rna,df_cl],axis=1)
df_rna=pandas.merge(df_rna,df_cl, left_index=True,right_index=True, how="left")
print(df_rna)
rl=[]
TagLabel="GeneClass"
for index,row in df_rna.iterrows():
    d=row.dropna().to_dict()
    if not TagLabel in d:
        continue
    ftag=d[TagLabel]
    for s in d:
        if s==TagLabel:
            continue
        rl.append((d[s],s,ftag))
df_final=pandas.DataFrame(rl,columns=["Expression","Sample","Class"])
print(df_final)
df_final["logExpression"]=np.log2(df_final["Expression"]+1)
cl=[("Denovo","NewDup","NewCnv","Orphan","OldSingleton")]
for ci in cl:
    sns.set_context("paper")
    sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})
    plt.rcParams['svg.fonttype'] = 'none'
#    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax=plt.subplots(dpi=300)
    sns.boxplot(data=df_final[df_final["Class"].isin(ci)], x="Class", y="logExpression", hue="Sample",linewidth=1,ax=ax,flierprops={"marker":"x","markerfacecolor":'grey',"markersize":0.1,"markeredgecolor":'grey'},order=ci,legend=True,showfliers=False)
    #sns.violinplot(data=df_final,x="Class", y="logExpression", hue="Sample",ax=ax)
    ax.set_ylabel("log2(Expression+1)",fontsize=15)
    #ax.set_ylim(0,0.5)
    sns.despine(ax=ax)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_xlabel("")
    ax.tick_params(axis='x',labelsize=15)
    #ax.set_ylabel(ax.get_ylabel(),fontsize=15)
    ax.tick_params(axis='y', labelsize=10)
    fig.tight_layout()
    fig.savefig("rna_%s_boxplot.svg"%ci[0],format="svg",dpi=300)
    fig.savefig("rna_%s_boxplot.png"%ci[0],format="png",dpi=300)
#    plt.show()
    fig.clf()
    plt.close(fig)
