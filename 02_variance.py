import numpy as np
import pandas as pd
import sys
import re

file1=open(sys.argv[1],'r')
data=[]
site=[]
label=[]
pattern=re.compile('^\t')
for line in file1:
    if pattern.findall(line):
        line=line.strip('\n').split('\t')
        label=line[1:]
    else:
        line=line.strip().split('\t')
        line1=line[0]
        line2=[float(x) for x in line[1:]]
        site.append(line1)
        data.append(line2)

data=np.array(data).T
variances = np.var(data, axis=0)
mask = variances > 0.004
filtered_data = np.array(data[:,mask]).T
site=np.array(site)
filtered_site=site[mask]

df=pd.DataFrame(filtered_data,columns=label,index=filtered_site)
df.to_csv(sys.argv[2],sep='\t')
