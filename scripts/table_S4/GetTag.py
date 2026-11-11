#!/usr/bin/env python
import pandas
def GetGeneTag():
    fld={"OldSingleton":"OldSingletonConserve.txt","Denovo":"denovo_343.txt","NewCnv":"NewGenesCnv.txt","NewDup":"NewGenesDup.txt","Orphan":"NewGenesOrphan.txt","OldCnv":"OldGenesCnv.txt","OldDup":"OldGenesDup.txt"}
    rl=[]
    for tag in fld:
        f=fld[tag]
        for x in open(f):
            x=x.rstrip()
            rl.append([x,tag])  
    df=pandas.DataFrame(rl,columns=["Gene","GeneClass"])
    df.reset_index(drop=True,inplace=True)
    df.set_index("Gene",inplace=True)
    return df
def GetPattle():
    my_pal = {"OldSingleton":"#F8B072","Denovo":"#F9B3AD","Orphan":"#4F99C9","NewDup":"#A8D3A0","NewCnv":"#A6D0E6","OldDup":"#bebada","OldCnv":"#8dd3c7","Random":"#7F7F7F"}
    return my_pal
if __name__=="__main__":
    df=GetGeneTag()
    print(df)
