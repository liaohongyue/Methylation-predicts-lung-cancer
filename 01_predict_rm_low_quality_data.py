import pandas as pd
import numpy as np
import sys

###处理CT_count part####
data=pd.read_csv(sys.argv[1],header=0,index_col=2,sep='\t')
data_selected=data[data['样品类型'].isin(['肺癌对照','肺癌'])]
data_selected=data_selected[data_selected['lamda_rate']<0.05]
data_selected=data_selected[data_selected['puc19_rate']>0.95]
data_selected=data_selected[data_selected['chhg_rate']<0.02]
data_selected=data_selected[data_selected['q20/raw:']>0.85]
data_selected[['panel','批次','样品类型','性别','年龄','诊断','分期','sampleName','rawdata:','q20filter:','q20/raw:','aligned pairs:','unique pairs:','align rate:','dup rate:','target_rate_new','mead_depth_new','target_rate_old','mead_depth_old','lamda_rate','chhg_rate','puc19_rate','lamda_base','chhg_base','puc19_base']].to_csv('./data/crc_data.txt',sep='\t')

data_selected['样品类型'].replace('肺癌对照','0',inplace=True)
data_selected['样品类型'].replace('肺癌','1',inplace=True)

# data_code = data_selected['样品编号']
data_selected=data_selected.drop(['panel','批次','样品编号','性别','年龄','诊断','分期','sampleName','rawdata:','q20filter:','q20/raw:','aligned pairs:','unique pairs:','align rate:','dup rate:','target_rate_new','mead_depth_new','target_rate_old','mead_depth_old','lamda_rate','chhg_rate','puc19_rate','lamda_base','chhg_base','puc19_base'],axis=1)
data_selected=data_selected.rename(columns={'样品类型':'sample_type'})

df=data_selected.T
df = df.astype(float)
df_1=df.iloc[1:]
df_1[df_1<40]=pd.NA
types=pd.DataFrame([df.iloc[0]])
df_1_1=pd.concat([types,df_1],ignore_index=False)

####处理beta value part#####
data2=pd.read_csv(sys.argv[2],header=0,index_col=2,sep='\t')
data2_selected=data2[data2['样品类型'].isin(['肺癌对照','肺癌'])]

data2_selected['样品类型'].replace('肺癌对照','0',inplace=True)
data2_selected['样品类型'].replace('肺癌','1',inplace=True)

data2_selected=data2_selected.drop(['panel','批次','样品编号','性别','年龄','诊断','分期','sampleName','rawdata:','q20filter:','q20/raw:','aligned pairs:','unique pairs:','align rate:','dup rate:','target_rate_new','mead_depth_new','target_rate_old','mead_depth_old','lamda_rate','chhg_rate','puc19_rate','lamda_base','chhg_base','puc19_base'],axis=1)
data2_selected=data2_selected.rename(columns={'样品类型':'sample_type'})

df2=data2_selected.T
df2=df2.astype(float)

df2_selected_row=df2.loc[df_1_1.index]
df2_selected_row_columns=df2_selected_row[df_1_1.columns]
###将CT_count表中NA的数据的位置，ratio的表的相同位置也替换为NA###
na_positions_df1=df_1_1.isna()
df2_selected_row_columns[na_positions_df1]=pd.NA
###删除缺失值大于20%的行（探针）###
# row_na_percent=df2_selected_row_columns.isna().mean(axis=1)
# selected_rows=df2_selected_row_columns[row_na_percent <= 0.2]

###删除缺失值大于20%的列（样本）###
# col_na_percent=selected_rows.isna().mean()
# selected_columns=selected_rows.loc[:, col_na_percent <= 0.2]

###前后特征的数据对样本的缺失值进行填充#####
selected_columns=df2_selected_row_columns
selected_columns.fillna(method='bfill',inplace=True,axis=1)
selected_columns.fillna(method='ffill',inplace=True,axis=1)
selected_columns.to_csv('%s'%(sys.argv[3]),sep='\t')
sample_label_2=selected_columns.iloc[0]
sample_label_2=sample_label_2.astype(int)
sample_label_2.to_csv('%s'%(sys.argv[4]),header=False,index=True,sep='\t')
# data_code.to_csv('./data/control_crc_sample_code.csv',header=True,index=True,sep='\t')
print(selected_columns)
