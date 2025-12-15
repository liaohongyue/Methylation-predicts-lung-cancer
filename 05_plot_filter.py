import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_train')
parser.add_argument('--train_diff')
parser.add_argument('--label_train')
parser.add_argument('--data_test')
parser.add_argument('--label_test')

args=parser.parse_args()

data_train=pd.read_csv(args.data_train,sep=',',index_col=None,header=0)
train_diff=pd.read_csv(args.train_diff,sep='\t',index_col=None,header=0)
label_train=pd.read_csv(args.label_train,sep='\t',index_col=None,header=None)
labels=[]

for i in range(len(label_train)):
    labels.append(str(label_train.iloc[i][1]))

df_fc_high=train_diff[train_diff["logFC"]<0]
df_fc_low=train_diff[train_diff["logFC"]>0]

df_selected_high=data_train.loc[data_train['Unnamed: 0'].isin(df_fc_high['chrBase'])]
df_selected_low=data_train.loc[data_train['Unnamed: 0'].isin(df_fc_low['chrBase'])]
df_selected_high.rename(columns={'Unnamed: 0':'chrBase'},inplace=True)
df_selected_low.rename(columns={'Unnamed: 0':'chrBase'},inplace=True)

df_selected_high.set_index('chrBase',inplace=True)
df_selected_low.set_index('chrBase',inplace=True)

df_selected_high.columns=[str(x) for x in labels]
df_selected_low.columns=[str(x) for x in labels]

#hypomethylation in tumor,normal tissue >70% beta value <0.4

# row_condition1=(df_selected_high['0']<0.5).sum(axis=1)
# df_selected_high_filterd=df_selected_high[row_condition1>=24]

#hypermethylation in tumor,normal tissue >70% beta value >=0.6
# row_condition2=(df_selected_low['0']>=0.5).sum(axis=1)
# df_selected_low_filterd=df_selected_low[row_condition2>=24] 
# df_selected=pd.concat([df_selected_high_filterd,df_selected_low_filterd])
df_selected=pd.concat([df_selected_high,df_selected_low])

data_test=pd.read_csv(args.data_test,sep=',',index_col=0,header=0)
label_test=np.array(pd.read_csv(args.label_test,sep='\t',index_col=0,header=None)).reshape(-1)
data_test_selected=pd.DataFrame(index=df_selected.index,columns=[label_test])

for line in df_selected.index.values:
    if line in data_test.index.values:
        data_test_selected.loc[line]=list(np.array(data_test.loc[line]))

df_selected=df_selected.T
data_test_selected=data_test_selected.T

path1=args.data_train.split('/')
path1.pop()
path2=args.data_test.split('/')
path2.pop()
file_train='/'.join(path1+['top_train_value.csv'])
df_selected.to_csv(file_train)
file_test='/'.join(path2+['top_test_value.csv'])
data_test_selected.to_csv(file_test)
data_ti=pd.concat([df_selected,data_test_selected])
data_ti=data_ti.T
data_ti=data_ti.astype(float)
data_ti[data_ti>1]=1

print(data_ti)

plt.figure()
cmap = 'jet'
p = sns.clustermap(data=data_ti, row_cluster=True,col_cluster=False,method='weighted',metric='chebyshev',figsize=(20, 8), yticklabels=False, cmap=cmap)
p.ax_col_dendrogram.set_title('training and test',fontsize=20)
plt.tight_layout()
file_pdf='/'.join(path1+['training_test.pdf'])
plt.savefig(file_pdf)
plt.close()
