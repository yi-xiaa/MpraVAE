import numpy as np
from numpy import array
from random import sample,seed
import time
import matplotlib.pyplot as plt
#from statannot import add_stat_annotation
import pandas as pd
#import numpy as np
from Bio import SeqIO
import h5py
import seaborn as sns
from scipy.stats import wilcoxon,pearsonr
from re import search
import math
import os
import warnings

import time
import sys


#from numpy import argmax
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
        # Convolutional layers
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=num_kernels[0], kernel_size=4)
        self.Conv2 = nn.Conv1d(in_channels=num_kernels[0], out_channels=num_kernels[1], kernel_size=4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        # Dropout
        self.Drop = nn.Dropout(p=dropout_rate)
        # Linear layers - adjust the input size based on num_kernels
        self.Linear1 = nn.Linear(61 * num_kernels[1], 128)  # Update size accordingly
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
        #print(seq_record.id)
        #print(seq_record.seq)
        #get sequence into an array
        seq_array = array(list(seq_record.seq))
        #integer encode the sequence
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)
        #one hot the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        #reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        x.append(onehot_encoded_seq)        
    x = array(x)
    return x

def testModel(model,model_savename,testData_seq,BATCH_SIZE=64,verbose=0,predictonly = 0):

    device = torch.device("cpu")

    testDataLoader = DataLoader(testData_seq, batch_size=BATCH_SIZE)

    #print("[INFO]  resume best model...")
    
    resume(model, model_savename)

    # we can now evaluate the network on the test set
    #print("[INFO] evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        predsProb = []
        ys = []

        # loop over the test set
        for x in testDataLoader:
            # send the input to the device
            x = x.to(device,dtype=torch.float)
            # make the predictions and add them to the list
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







