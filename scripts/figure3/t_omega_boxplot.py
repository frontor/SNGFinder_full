#!/usr/bin/env python
from statannotations.Annotator import Annotator
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys,itertools
import GetTag
from scipy.stats import mannwhitneyu
feature_file=sys.argv[1]
prefix=sys.argv[2]
df_cl=GetTag.GetGeneTag()
df_features_ori=pandas.read_csv(feature_file,sep="\t",header=0,index_col=0)
features_list=["Omega"] #"SignalP"
df_features=df_features_ori.loc[:,features_list]
#df_features=pandas.concat([df_features,df_cl],axis=1)
df_features=pandas.merge(df_features, df_cl, left_index=True, right_index=True, how="left")
print(df_features)
rl=[]
TagLabel="GeneClass"
for index,row in df_features.iterrows():
    d=row.dropna().to_dict()
    if not TagLabel in d:
        continue
    ftag=d[TagLabel]
    for s in d:
        #rl.append((d[s],s,"All"))
        if s==TagLabel:
            continue
        rl.append((d[s],s,ftag))
df_final=pandas.DataFrame(rl,columns=["Omega","Features","Class"])
print(df_final)
cl=[("Denovo","NewDup","NewCnv","Orphan","OldSingleton"),("NewDup","OldDup"),("NewCnv","OldCnv")]
alterd={}
for x in open("utest.xls").readlines()[1:]:
    x=x.rstrip()
    l=x.split("\t")
    a,b,c=l[0].split()
    alterd[a+"_"+c+"_"+l[1]]=l[2]
for ci in cl:
    sns.set_context("paper") #font_scale=1.5
    sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})
    plt.rcParams['svg.fonttype'] = 'none'
#    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax=plt.subplots(dpi=300,figsize=(6.5,8)) #figsize=(6,4.5)
    my_pal = GetTag.GetPattle()
    sns.boxplot(data=df_final[df_final["Class"].isin(ci)], x="Features", y="Omega", hue="Class",order=features_list,hue_order=ci,linewidth=1,ax=ax,flierprops={"marker": "x","markerfacecolor":'grey','markersize':0.1,"markeredgecolor":'grey'},palette=my_pal,showfliers=False) #Set3
    #sns.violinplot(data=df_final[df_final["Class"].isin(ci)], x="Features", y="Probability", hue="Class",order=features_list,hue_order=ci,linewidth=1,ax=ax,palette=my_pal)
    sns.despine(ax=ax)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    pairs=[]
    pvalues=[]
    for fi in features_list:
        for c1,c2 in itertools.combinations(ci,2):
            if c2=="Orphan" or c2=="NewDup" or c2=="NewCnv" or c2=="Denovo":
                c1,c2=c2,c1 
#            if (c1=="OldConserve" and c2=="OldNotConserve") or (c1=="OldNotConserve" and c2=="OldConserve"):
#                continue
            alabel=c1+"_"+c2+"_"+fi
            pairs.append([(fi,c1),(fi,c2)])
            a=df_final[(df_final["Class"]==c1)&(df_final["Features"]==fi)]["Omega"]
            b=df_final[(df_final["Class"]==c2)&(df_final["Features"]==fi)]["Omega"]
            print(a.shape,b.shape)
#            pvalues.append(mannwhitneyu(list(a),list(b),alternative=alterd[alabel]).pvalue)
            pvalues.append(mannwhitneyu(list(a),list(b),alternative="two-sided").pvalue)
            print(fi,c1,c2)
            print("pvalue",pvalues[-1])
    print(pairs)
    formatted_pvalues=[]
    for p in pvalues:
        psc="{:.2e}".format(p)
        if float(psc)>0 and float(psc)<1:
            formatted_pvalues.append("pvalue=%s"%psc)
        else:
            formatted_pvalues.append("pvalue=%s"%round(p,2))
    annot = Annotator(ax,pairs,data=df_final[df_final["Class"].isin(ci)], x="Features", y="Omega",order=features_list,hue_order=ci,hue="Class")
    #annot.new_plot(order=features_list, hue="Class", hue_order=ci)
    annot.configure(test=None, loc='outside',text_format='star')
    #annot.set_custom_annotations(formatted_pvalues)
    annot.set_pvalues(pvalues)
    annot.annotate()
    ax.set_xlabel("")
    ax.tick_params(axis='x',labelsize=15)
    ax.set_ylabel(ax.get_ylabel(),fontsize=15)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(loc='upper right',fontsize=10)
#    ax.set_ylim(top=1)
    fig.tight_layout()
    fig.savefig("%s_%s_boxplot.svg"%(prefix,ci[0]),format="svg",dpi=300)
    fig.savefig("%s_%s_boxplot.png"%(prefix,ci[0]),format="png",dpi=300)
    fig.clf()
    plt.close(fig)
    #break
