import numpy as np
from numpy import array
from random import sample,seed
import time
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
import h5py
import seaborn as sns
from scipy.stats import wilcoxon,pearsonr
from re import search
import math
import os
import warnings
import argparse

import time
import sys

from sklearn.metrics import roc_curve,auc,f1_score,recall_score,precision_score,accuracy_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset 
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
warnings.filterwarnings("ignore")

class CNN_single1(nn.Module):
    def __init__(self, num_kernels=(128, 256), dropout_rate=0.1):
        super(CNN_single1, self).__init__()

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=num_kernels[0], kernel_size=4)
        self.Conv2 = nn.Conv1d(in_channels=num_kernels[0], out_channels=num_kernels[1], kernel_size=4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.Drop = nn.Dropout(p=dropout_rate)

        self.Linear1 = nn.Linear(61 * num_kernels[1], 128)
        self.Linear2 = nn.Linear(128, 32)
        self.Linear3 = nn.Linear(32, 2)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        
        x = x.view(-1,self.Linear1.in_features)
                
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        x = self.Linear3(x)
        
        x = self.logSoftmax(x)
        return x

def onehot(fafile):
    x=[]
    for seq_record in SeqIO.parse(fafile, "fasta"):

        seq_array = array(list(seq_record.seq))
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        x.append(onehot_encoded_seq)        
    x = array(x)
    return x

# In[8]:

def eval_model(preds,predsProb,y_test,verbose=0):    
    y_test_prob = predsProb
    y_test_classes=preds
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    auc_test = auc(fpr, tpr)
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_prob)
    auprc_test = auc(recall, precision)
    
    acc_test=accuracy_score(y_test_classes, y_test)
    f1_test = f1_score(y_test_classes, y_test, average='binary')
    recall_test = recall_score(y_test_classes, y_test, average='binary')
    precision_test = precision_score(y_test_classes, y_test, average='binary')
    R_test=pearsonr(y_test, y_test_prob)[0]
    
    acc_test=round(acc_test,3)
    auc_test=round(auc_test,3)
    auprc_test = round(auprc_test, 3)
    f1_test=round(f1_test,3)
    precision_test=round(precision_test,3)
    recall_test=round(recall_test,3)
    R_test=round(R_test,3)
    
    if verbose==1:
        get_ipython().run_line_magic('matplotlib', 'inline')
        print(f'Test: acc {acc_test:.3f}, auc {auc_test:.3f}, auprc {auprc_test:.3f}, f1 {f1_test:.3f}, precision {precision_test:.3f}, recall {recall_test:.3f}, R {R_test:.3f}\n')
    return [acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test]


def testModel(model,model_savename,testData_seq,BATCH_SIZE=64,verbose=0,predictonly = 0):

    device = torch.device("cpu")

    testDataLoader = DataLoader(testData_seq, batch_size=BATCH_SIZE)
    
    resume(model, model_savename)

    with torch.no_grad():
        model.eval()

        preds = []
        predsProb = []
        ys = []

        for x in testDataLoader:
            x = x.to(device,dtype=torch.float)
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
            predsProb.extend(pred[:,1].cpu().numpy())

    preds=np.array(preds)
    predsProb=np.array(predsProb)
    ys=np.array(ys)
    if predictonly == 0:
        acc_test, auc_test, f1_test, precision_test, recall_test,R_test=eval_model(preds,predsProb,ys,verbose=verbose)
        return [acc_test, auc_test, f1_test, precision_test, recall_test,R_test,predsProb]
    else:
        return predsProb

def resume(model, filename):
    model.load_state_dict(torch.load(filename,map_location=torch.device('cpu')))

    
def getProbability_CNN_py(modelname, seq_input_path, outfolder=""):
    seq_input = onehot(seq_input_path)
    seq_input_inv = seq_input.transpose((0, 2, 1))
    
    model = CNN_single1()
    
    predsProb = testModel(model, modelname, seq_input_inv, BATCH_SIZE=64, verbose=0, predictonly=1)
    
    df = pd.DataFrame(np.exp(predsProb))
    
    outpath = "./" + outfolder + "probs_out.csv"
    df.to_csv(outpath, header=None, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CNN prediction model.")
    parser.add_argument("--modelname", type=str, required=True, help="Name of the model to use for prediction.")
    parser.add_argument("--seq_input_path", type=str, required=True, help="Path to the input fasta file.")
    parser.add_argument("--outfolder", type=str, required=True, help="Output folder for the results.")

    args = parser.parse_args()

    getProbability_CNN_py(args.modelname, args.seq_input_path, args.outfolder)





