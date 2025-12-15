import sys
import pandas as pd
import random
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--label')
args=parser.parse_args()

file1=pd.read_csv(args.label,sep='\t',header=None,index_col=None)
file2=pd.read_csv(args.dataset,sep='\t',header=0,index_col=0)
file1.columns=["sample","label"]

cancer=list(file1[file1["label"]==1]["sample"])
control=list(file1[file1["label"]==0]["sample"])

random.seed(42)
cancer_training=random.sample(cancer,32)

random.seed(42)
control_training=random.sample(control,40)

cancer_testing=sorted(list(set(cancer)-set(cancer_training)))
control_testing=sorted(list(set(control)-set(control_training)))

training_table=open("./workdir/train_in/label.txt",'a')
for line in cancer_training:
    txt='\t'.join([str(line),str(1)])+'\n'
    training_table.write(txt)
for line in control_training:
    txt='\t'.join([str(line),str(0)])+'\n'
    training_table.write(txt)
training_table.close()

testing_table=open("./workdir/test_in/label.txt",'a')
for line in cancer_testing:
    txt='\t'.join([str(line),str(1)])+'\n'
    testing_table.write(txt)
for line in control_testing:
    txt='\t'.join([str(line),str(0)])+'\n'
    testing_table.write(txt)
testing_table.close()

training=cancer_training+control_training
testing=cancer_testing+control_testing
dataset_training=file2[[str(x) for x in training]]
dataset_testing=file2[[str(x) for x in testing]]

dataset_training.to_csv("./workdir/train_in/dataset.txt")
dataset_testing.to_csv("./workdir/test_in/dataset.txt")
