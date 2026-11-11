#!/usr/bin/env python
from statannotations.Annotator import Annotator
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys,itertools
import GetTag
from scipy.stats import mannwhitneyu
f=sys.argv[1]
prefix=sys.argv[2]
df_cl=GetTag.GetGeneTag()
df_features=pandas.read_csv(f,sep="\t",header=0,index_col=0)
if prefix=="genefeatures":
    df_features.rename(columns={"AverageExonLength":"ExonL","AverageIntronLength":"IntronL","AverageCDSLength":"CDSL","IsoformNumber":"IsoformN","ExonNumber":"ExonN","GeneLength":"GeneL"},inplace=True)
elif prefix=="seqfeatures":
    df_features.rename(columns={"GCContent":"GC","PurineContent":"GA"},inplace=True)
#df_features=pandas.concat([df_features,df_cl],axis=1)
df_features=pandas.merge(df_features,df_cl, left_index=True, right_index=True, how="left")
print(df_features)
rl=[]
rl_iso=[]
TagLabel="GeneClass"
for index,row in df_features.iterrows():
    d=row.dropna().to_dict()
    if not TagLabel in d:
        continue
    ftag=d[TagLabel]
    for s in d:
        if s==TagLabel:
            continue
        elif s=="IsoformN":
            rl_iso.append((d[s],s,ftag))
        rl.append((d[s],s,ftag))
df_final=pandas.DataFrame(rl,columns=["N","Features","Class"])
print(df_final)
df_final["logN"]=np.log2(df_final["N"]+1)
if prefix=="genefeatures":
    df_iso=pandas.DataFrame(rl_iso,columns=["N","Features","Class"])
    df_iso["logN"]=np.log2(df_iso["N"]+1)
cl=[("Denovo","NewDup","NewCnv","Orphan","OldSingleton"),("NewDup","OldDup"),("NewCnv","OldCnv")]
alterd={}
transdic={"AverageExonLength":"ExonL","AverageIntronLength":"IntronL","AverageCDSLength":"CDSL","IsoformNumber":"IsoformN","ExonNumber":"ExonN","GeneLength":"GeneL","GCContent":"GC","PurineContent":"GA"}
for x in open("utest.xls").readlines()[1:]:
    x=x.rstrip()
    l=x.split("\t")
    a,b,c=l[0].split()
    tps=l[1]
    if l[1] in transdic:
        tps=transdic[l[1]]
    alterd[a+"_"+c+"_"+tps]=l[2]
print (alterd)
for ci in cl:
    sns.set_context("paper") 
    sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})
    plt.rcParams['svg.fonttype'] = 'none'
#    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax=plt.subplots(dpi=300,figsize=(6.5,8))
    my_pal = GetTag.GetPattle()
    sns.boxplot(data=df_final[df_final["Class"].isin(ci)], x="Features", y="logN", hue="Class",hue_order=ci,linewidth=1,ax=ax,flierprops={"marker": "x","markerfacecolor":'grey', 'markersize':0.1,"markeredgecolor":'grey'},palette=my_pal,order=df_final['Features'].unique(),showfliers=False)
    ax.set_ylabel("log2(N+1)",fontsize=15)
    if prefix=="seqfeatures":
        ax.set_ylim(top=1,bottom=0)
    sns.despine(ax=ax)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_xlabel("")
    ax.tick_params(axis='x',labelsize=15)
    ax.tick_params(axis='y', labelsize=10)
    #ax.set_ylabel(ax.get_ylabel(),fontsize=15)
    ax.legend(fontsize=10)
    pairs=[]
    pvalues=[]
    for fi in df_final["Features"].unique():
        for c1,c2 in itertools.combinations(ci,2):
            if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv" or c2=="Denovo":
                c1,c2=c2,c1 
#            if (c1=="OldConserve" and c2=="OldNotConserve") or (c1=="OldNotConserve" and c2=="OldConserve"):
#                continue
            alabel=c1+"_"+c2+"_"+fi
            pairs.append([(fi,c1),(fi,c2)])
            a=df_final[(df_final["Class"]==c1)&(df_final["Features"]==fi)]["N"]
            b=df_final[(df_final["Class"]==c2)&(df_final["Features"]==fi)]["N"]
            print(a.shape,b.shape)
#            pvalues.append(mannwhitneyu(list(a),list(b),alternative=alterd[alabel]).pvalue)
            pvalues.append(mannwhitneyu(list(a),list(b),alternative="two-sided").pvalue)
            print(fi,c1,c2)
            print("pvalue",pvalues[-1])
    print(pairs)
    annot = Annotator(ax,pairs,data=df_final[df_final["Class"].isin(ci)], x="Features", y="logN",order=df_final["Features"].unique(),hue_order=ci,hue="Class")
    annot.configure(test=None, loc='outside',text_format='star')
    annot.set_pvalues(pvalues)
    annot.annotate()
    fig.tight_layout()
    fig.savefig("%s_%s_boxplot.svg"%(prefix,ci[0]),format="svg",dpi=300)
    #fig.savefig("%s_%s_boxplot.png"%(prefix,ci[0]),format="png",dpi=300)
#    plt.show()
    fig.clf()
    plt.close(fig)
#------------------------------------------------
    #if prefix=="genefeatures":
        #sns.set_context("paper") 
        #sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})
        #plt.rcParams['svg.fonttype'] = 'none'
        #fig, ax=plt.subplots(dpi=300)
        #my_pal = GetTag.GetPattle()
        #sns.boxplot(data=df_iso[df_iso["Class"].isin(ci)], x="Features", y="N", hue="Class",hue_order=ci,linewidth=1,ax=ax,flierprops={"marker": "x","markerfacecolor":'grey', 'markersize':0.1,"markeredgecolor":'grey'},palette=my_pal,order=df_iso["Features"].unique())
        #ax.set_ylim(0,6)
        #ax.set_ylabel("log2(N+1)")
        #print(df_iso[df_iso["Class"].isin(ci)])
        #sns.kdeplot(data=df_iso[df_iso["Class"].isin(ci)], x="N", hue="Class",hue_order=ci,fill=True,common_norm=False, palette=my_pal,alpha=.6, linewidth=2,ax=ax)
        #ma=df_iso[df_iso["Class"].isin(ci)]["N"].max()
        #bins=np.arange(0,ma+1,1)
        #print(bins)
        #sns.histplot(data=df_iso[df_iso["Class"].isin(ci)], x="N", hue="Class",hue_order=ci,palette=my_pal,stat="density",common_norm=True,linewidth=1,alpha=.6,bins=bins,ax=ax)
        #sns.despine(ax=ax)
        #ax.spines['left'].set_color('black')
        #ax.spines['bottom'].set_color('black')
        #ax.set_xlabel("Isoform",fontsize=15)
        #ax.set_xlim(left=1)
        #ax.set_xticks([])
        #ax.tick_params(axis='x',labelsize=15)
        #ax.set_ylabel("Density",fontsize=15)
"""
        pairs=[]
        pvalues=[]
        for fi in df_iso["Features"].unique():
            for c1,c2 in itertools.combinations(ci,2):
                if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv":
                    c1,c2=c2,c1 
                pairs.append([(fi,c1),(fi,c2)])
                a=df_iso[(df_iso["Class"]==c1)&(df_iso["Features"]==fi)]["N"]
                b=df_iso[(df_iso["Class"]==c2)&(df_iso["Features"]==fi)]["N"]
                print(a.shape,b.shape)
                pvalues.append(mannwhitneyu(list(a),list(b),alternative='two-sided').pvalue)
                print(fi,c1,c2)
                print("pvalue",pvalues[-1])
        print(pairs)
        annot = Annotator(ax,pairs,data=df_iso[df_iso["Class"].isin(ci)], x="Features", y="logN",order=df_iso["Features"].unique(),hue_order=ci,hue="Class")
        annot.configure(test=None, loc='outside',text_format='star')
        annot.set_pvalues(pvalues)
        annot.annotate()
"""
        #fig.tight_layout()
        #fig.savefig("%s_%s_Isoform_boxplot.svg"%(prefix,ci[0]),format="svg",dpi=300)
        #fig.savefig("%s_%s_Isoform_boxplot.png"%(prefix,ci[0]),format="png",dpi=300)
        #plt.show()
        #fig.clf()
        #plt.close(fig)
