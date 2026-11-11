import sys

fin=open(sys.argv[1],'r')
denovo=[]
for i in fin:
    i=i.strip()
    if i not in denovo:
        denovo.append(i)
    else:
        print ('175 de novo有重复：i')
fin.close()
fin=open(sys.argv[2],'r')
fout=open(sys.argv[3],'w')
for j in fin:
    j=j.strip()
    if j.startswith('Gene'):
        print (j,file=fout)
    else:
        line=j.split('\t')
        if line[-1]=='0':
            print (j,file=fout)
        else:
            if line[0] in denovo:
                print (j,file=fout)
            else:
                continue
fin.close()
fout.close()
