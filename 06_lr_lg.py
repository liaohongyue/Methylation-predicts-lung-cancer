import argparse
from sklearn.metrics import roc_auc_score,roc_curve,log_loss,precision_score,accuracy_score
from sklearn.model_selection import LeaveOneOut,GridSearchCV
import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import RandomForestClassifier

import pickle
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--data_train')
parser.add_argument('--data_test')
parser.add_argument('--label_train_name')
parser.add_argument('--label_test_name')
args=parser.parse_args()

data_train=pd.read_csv(args.data_train,sep=',',index_col=0,header=0)
txt=str(args.data_train).strip().split('/')
txt.pop()
label_train_name=pd.read_csv(args.label_train_name,sep='\t',index_col=None,header=None).to_numpy()

path='/'.join(txt)+'/'

label_train=data_train.index.to_numpy()
data_test=pd.read_csv(args.data_test,sep=',',index_col=0,header=0)
label_test=data_test.index.to_numpy()
label_test_name=pd.read_csv(args.label_test_name,sep='\t',index_col=None,header=None).to_numpy()
######select from random forest######
'''
selected_feature_indices_rf = []
num_selections_rf = 50
for _ in range(num_selections_rf):
    selector_rf=SelectFromModel(estimator=RandomForestClassifier(n_estimators=150,criterion='gini',class_weight='balanced_subsample',random_state=_))
    selector_rf.fit(data_train, label_train)
    selected_feature_indices_rf.append(selector_rf.get_support(indices=True))

feature_counts_rf = np.bincount(np.concatenate(selected_feature_indices_rf))
selected_features_rf = np.array(data_train.columns)[np.where(feature_counts_rf >= num_selections_rf*0.6)[0]]
print(len(selected_features_rf))

#####select from lasso ######

selected_feature_indices_lasso = []
num_selections_lasso = 50
for _ in range(num_selections_lasso):
    selector_lasso=SelectFromModel(Lasso(max_iter=2000,tol=2e-2,alpha=1e-4,selection='random',random_state=_))
    selector_lasso.fit(data_train, label_train)
    selected_feature_indices_lasso.append(selector_lasso.get_support(indices=True))
feature_counts_lasso = np.bincount(np.concatenate(selected_feature_indices_lasso))
selected_features_lasso = np.array(data_train.columns)[np.where(feature_counts_lasso >= num_selections_lasso*0.5)[0]]
print(len(selected_features_lasso))

features_selected=list(set(selected_features_rf) & set(selected_features_lasso))
print(len(features_selected))

####lg part####

features_path=path+"features_top.txt"
str_features='\t'.join(features_selected)
file1=open(features_path,'w')
file1.write(str_features)
'''
features_path=path+"features_top.txt"
features_selected=np.array(pd.read_csv(features_path,header=None,index_col=None,sep='\t')).tolist()[0]

#####data deal part####

X_train=data_train[features_selected]
X_test=data_test[features_selected]

###training part Logistic regression###
param_grid = {
    'C': [5],
    'solver':['lbfgs'],
    'penalty':['l2'],
    'tol':[1e-4]
}
loo=LeaveOneOut()
base_estimator=LogisticRegression(class_weight='balanced',max_iter=5000,multi_class='ovr',random_state=42)
grid_search = GridSearchCV(base_estimator,param_grid=param_grid,cv=loo,scoring='accuracy',refit=True)
grid_search.fit(X_train,label_train)
model = grid_search.best_estimator_
with open(path+'model.pkl','wb') as file:
    pickle.dump(model,file) 
parameter=grid_search.best_params_
print(parameter)

train_predict=np.array(model.predict_proba(X_train)).T[1]
train_loss=log_loss(label_train,train_predict).mean()
train_auc=roc_auc_score(label_train,train_predict)
train_accuracy=str(accuracy_score(label_train,model.predict(X_train)))
train_precision=str(precision_score(label_train,model.predict(X_train)))
train_predict_label=np.array(model.predict(X_train))

test_predict=np.array(model.predict_proba(X_test)).T[1]
test_auc=roc_auc_score(label_test,test_predict)
test_loss=log_loss(label_test,test_predict).mean()
test_accuracy=str(accuracy_score(label_test,model.predict(X_test)))
test_precision=str(precision_score(label_test,model.predict(X_test)))
test_predict_label=np.array(model.predict(X_test))

file1=open("./success/score.txt",'a')
file1.write("train"+"\n")
train_score='\t'.join([str(train_loss),str(train_auc),str(train_accuracy),str(train_precision)])+'\n'
print(train_score)
file1.write(train_score)
file1.write("test"+"\n")
test_score='\t'.join([str(test_loss),str(test_auc),str(test_accuracy),str(test_precision)])+'\n'
print(test_score)
file1.write(test_score)
file1.close()

file2=open("./success/predicted.txt",'a')
file2.write("train"+"\n")
for i in range(len(label_train_name)):
    txt_train=[str(label_train_name[i][0]),str(label_train_name[i][1]),str(train_predict_label[i])]
    str_txt_train='\t'.join(txt_train)+'\n'
    file2.write(str_txt_train)
file2.write("test"+"\n")

for j in range(len(label_test_name)):
    txt_test=[str(label_test_name[j][0]),str(label_test_name[j][1]),str(test_predict_label[j])]
    str_txt_test='\t'.join(txt_test)+'\n'
    file2.write(str_txt_test)
file2.close()

fpr_train,tpr_train,thres_train=roc_curve([label_train[i] for i in range(len(label_train))],[train_predict[i] for i in range(len(train_predict))],pos_label=1,drop_intermediate=True)
# fpr_test,tpr_test,thres_test=roc_curve([label_test[i] for i in range(len(label_test))],[test_predict[i] for i in range(len(test_predict))],pos_label=1,drop_intermediate=True)
plt.plot(figsize=(10,8))
plt.plot(fpr_train,tpr_train,linewidth=2,label='AUC=%0.2f'%(round(train_auc,3)))
# plt.plot(fpr_test,tpr_test,linewidth=2,label='AUC=%0.2f'%(round(test_auc,3)))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
auc_fig=path+'auc.pdf'

plt.savefig(auc_fig)
plt.close()
