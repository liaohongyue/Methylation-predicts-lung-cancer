import joblib
import argparse
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_score,accuracy_score,log_loss
import random

random.seed(42)
np.random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--features')
parser.add_argument('--model')
parser.add_argument('--independent_label')
args=parser.parse_args()

df=pd.read_csv(args.data,sep='\t',header=0,index_col=0)

#####获得features#####
features=np.array(pd.read_csv(args.features,header=None,index_col=None,sep='\t')).tolist()[0]
independent_label=pd.read_csv(args.independent_label,sep='\t',header=None,index_col=None).to_numpy()
# features_tmp=open(args.features,'r').readlines()
# features=[line.strip() for line in features_tmp]

df=df.T
X_independent=df[features]

clf = joblib.load(args.model) 
predict_independent=np.array(clf.predict_proba(X_independent)).T[1]
predict_independent=[float(x) for x in predict_independent]
label_independent=list(np.array(df['sample_type']))
loss=str(log_loss(label_independent,predict_independent).mean())
auc=str(roc_auc_score(label_independent,predict_independent))
accuracy=str(accuracy_score(label_independent,clf.predict(X_independent)))
# precision=str(precision_score(label_independent,clf.predict(X_independent)))

independent_predict_0_1=clf.predict(X_independent)
TP_independent=0
TN_independent=0
FP_independent=0
FN_independent=0
for i in range(len(label_independent)):
    if label_independent[i]==independent_predict_0_1[i]==1:
        TP_independent+=1
    elif label_independent[i]==independent_predict_0_1[i]==0:
        TN_independent+=1
    elif label_independent[i]==1 and independent_predict_0_1[i]==0:
        FN_independent+=1
    elif label_independent[i]==0 and independent_predict_0_1[i]==1:
        FP_independent+=1

sensitivity_independent=TP_independent/(TP_independent+FN_independent)
specificity_independent=TN_independent/(TN_independent+FP_independent)
ppv_independent=TP_independent/(TP_independent+FP_independent)
npv_independent=TN_independent/(TN_independent+FN_independent)

file1=open("./success/score.txt",'a')
file1.write("log_loss"+"\t"+"auc"+"\t"+"accuracy"+"\t"+"\t"+"sensitivity"+"\t"+"specificity")
score='\t'.join([loss,auc,accuracy,str(sensitivity_independent),str(specificity_independent)])+'\n'
print(score)
file1.write("independent"+"\n")
file1.write(score)
file1.close()

predict_independent_label=np.array(clf.predict(X_independent))
file2=open("./success/predicted.txt",'a')
file2.write("independent"+"\n")
for i in range(len(independent_label)):
    txt_independent=[independent_label[i][0],str(independent_label[i][1]),str(predict_independent_label[i])]
    str_txt_independent='\t'.join(txt_independent)+'\n'
    file2.write(str_txt_independent)
file2.close()

