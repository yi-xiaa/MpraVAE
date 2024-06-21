#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import array
from random import sample,seed
import time
import matplotlib.pyplot as plt
#from statannot import add_stat_annotation
import pandas as pd
#import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import h5py
import seaborn as sns
from scipy.stats import wilcoxon,pearsonr
from re import search
import math
import os
import warnings
import time
import sys
from collections import Counter
import itertools
import statistics


#from numpy import argmax
from sklearn.metrics import roc_curve,roc_auc_score,auc,f1_score,recall_score,precision_score,accuracy_score,precision_recall_curve
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset 
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn


import matplotlib
# import the necessary packages
#from pyimagesearch.lenet import LeNet


# In[3]:


def onehot_old(fafile):
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


def onehot(fafile):
    x = []
    for seq_record in SeqIO.parse(fafile, "fasta"):
        #print(seq_record.id)
        #print(seq_record.seq)
        #get sequence into an array
        seq_array = np.array(list(seq_record.seq))

        # Check if sequence length is as expected (1001)
        if len(seq_array) != 1001:
            print(f"Warning: Sequence length {len(seq_array)} is not 1001")
            # Handle the case where sequence length is not 1001, e.g., by skipping or padding
            continue

        # Integer encode the sequence
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)

        # One-hot encode the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

        # Ensure the one-hot encoded sequence has the shape (1001, 4)
        if onehot_encoded_seq.shape != (1001, 4):
            print(f"Unexpected shape: {onehot_encoded_seq.shape}")
            continue
        x.append(onehot_encoded_seq)
    x = array(x)
    return x


# In[ ]:


def onehot_to_seq(onehot_encoded):
    """
    Convert one-hot encoded DNA sequences back to nucleotide sequences.
    """
    base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = []
    for seq in onehot_encoded:
        nucleotide_seq = ""
        for base in seq:
            nucleotide_seq += base_dict[np.argmax(base)]
        sequences.append(nucleotide_seq)
    return sequences

def save_to_fastafile(sequences, filename, output_dir):
    """
    Save a list of DNA sequences to a FASTA file.
    """
    seq_records = [SeqRecord(Seq(seq), id=f"seq_{i}", description="") for i, seq in enumerate(sequences)]
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")


# In[ ]:


def read_fasta_and_label(file_path, label):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return pd.DataFrame({'sequence': sequences, 'label': label})

def get_all_kmers(k):
    """Generate all possible k-mers given the alphabet."""
    return [''.join(p) for p in itertools.product('ATCG', repeat=k)]

def extract_features(sequence, k=3):
    """Extract k-mer frequency features from a given sequence."""
    kmers = get_all_kmers(k)
    kmer_counts = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
    total_kmers = sum(kmer_counts.values())
    return np.array([kmer_counts.get(kmer, 0) / total_kmers for kmer in kmers])

def sequence_to_kmer_features(sequences, k=3):
    kmer_set = [''.join(p) for p in itertools.product('ATCG', repeat=k)]
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmer_set)}
    features = np.zeros((len(sequences), len(kmer_set)))
    
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_index:
                features[i, kmer_index[kmer]] += 1
    return features


# In[ ]:


def RF_baseline_auc_with_downsampling(celltypes, idata, downsample=1.0):
    if downsample <= 0 or downsample > 1:
        raise ValueError("downsample must be between 0 and 1")
        
    print(f'########### downsample: {downsample} ###########')
    combine = 'true'
    for celltype in celltypes:
        print(f'#### celltype: {celltype} ####')

        pos_file = f'seq.{idata}.{celltype}.pos.fasta'
        neg_file = f'seq.{idata}.{celltype}.neg.fasta'
        pos_data = read_fasta_and_label(pos_file, 1)
        neg_data = read_fasta_and_label(neg_file, 0)
        
        if downsample < 1.0:
            pos_sample = pos_data.sample(frac=downsample, random_state=42)
            neg_sample = neg_data.sample(frac=downsample, random_state=42)
            data = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
        else:
            data = pd.concat([pos_data, neg_data]).reset_index(drop=True)

        
        data['features'] = data['sequence'].apply(lambda x: extract_features(x, k=3))
        X_train, X_test, y_train, y_test = train_test_split(
            np.stack(data['features'].values), 
            data['label'].values, 
            test_size=0.2, 
            random_state=42
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        print(f'AUC Score: {auc_score:.4f}')


# In[1]:


def RF_baseline_auc_with_5foldCV(celltype, idata, random_state, fraction=1.0):
    if fraction <= 0 or fraction > 1:
        raise ValueError("downsample must be between 0 and 1")
    print(f'########### fraction: {fraction} ###########')

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    pos_file = f'seq.{idata}.{celltype}.pos.fasta'
    neg_file = f'seq.{idata}.{celltype}.neg.fasta'
    pos_data = read_fasta_and_label(pos_file, 1)
    neg_data = read_fasta_and_label(neg_file, 0)

    if fraction < 1.0:
        pos_sample = pos_data.sample(frac=fraction, random_state=random_state)
        neg_sample = neg_data.sample(frac=fraction, random_state=random_state)
        data = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
    else:
        data = pd.concat([pos_data, neg_data]).reset_index(drop=True)

    data['features'] = data['sequence'].apply(lambda x: extract_features(x, k=3))
    X = np.stack(data['features'].values)
    y = data['label'].values

    auc_scores = []
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc_score)

    mean_auc = np.mean(auc_scores)
    print(f'Mean AUC Score(5-fold CV): {mean_auc:.4f}')
    baseline_auc = mean_auc
    return baseline_auc


# In[ ]:


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)


# In[4]:


def readData_nounlabeled(idata,celltype):
    # true data
    seq_pos_file='seq.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq=onehot(data_folder/seq_pos_file)
    seq_neg_file='seq.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq=onehot(data_folder/seq_neg_file)
    # rev data
    seq_pos_file_rev='seq.rev.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq_rev=onehot(data_folder/seq_pos_file_rev)
    seq_neg_file_rev='seq.rev.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq_rev=onehot(data_folder/seq_neg_file_rev)
    # crop data
    seq_pos_file_crop='seq.crop.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq_crop=onehot(data_folder/seq_pos_file_crop)
    seq_neg_file_crop='seq.crop.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq_crop=onehot(data_folder/seq_neg_file_crop)

    print('true, pos and neg: ',[x_pos_seq.shape,x_neg_seq.shape])
    print('rev, pos and neg: ',[x_pos_seq_rev.shape,x_neg_seq_rev.shape])
    print('crop, pos and neg: ',[x_pos_seq_crop.shape,x_neg_seq_crop.shape])
    
    return x_pos_seq,x_neg_seq,x_pos_seq_rev,x_neg_seq_rev,x_pos_seq_crop,x_neg_seq_crop


def readData(idata,celltype):
    # true data
    seq_pos_file='seq.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq=onehot(data_folder/seq_pos_file)
    seq_neg_file='seq.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq=onehot(data_folder/seq_neg_file)
    seq_unlabeled_file='seq.'+idata+'.'+celltype+'.unlabeled.fasta' #unlabeled data
    x_unlabeled_seq=onehot(data_folder/seq_unlabeled_file)
    # rev data
    seq_pos_file_rev='seq.rev.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq_rev=onehot(data_folder/seq_pos_file_rev)
    seq_neg_file_rev='seq.rev.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq_rev=onehot(data_folder/seq_neg_file_rev)
    # crop data
    seq_pos_file_crop='seq.crop.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq_crop=onehot(data_folder/seq_pos_file_crop)
    seq_neg_file_crop='seq.crop.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq_crop=onehot(data_folder/seq_neg_file_crop)

    print('true, pos and neg: ',[x_pos_seq.shape,x_neg_seq.shape])
    print('rev, pos and neg: ',[x_pos_seq_rev.shape,x_neg_seq_rev.shape])
    print('crop, pos and neg: ',[x_pos_seq_crop.shape,x_neg_seq_crop.shape])
    print('unlabeled: ',[x_unlabeled_seq.shape])
    
    return x_pos_seq,x_neg_seq,x_pos_seq_rev,x_neg_seq_rev,x_pos_seq_crop,x_neg_seq_crop, x_unlabeled_seq


# In[ ]:


# Helper function to downsample data and get indices
def downsample_data(x, y, fraction):
        indices = np.random.choice(len(x), int(len(x) * fraction), replace=False)
        return x[indices], y[indices], indices


# In[ ]:


def split_testdata(x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, test_size=0.2, seed=1234, verbose=0):
    np.random.seed(int(seed))
    
    # For True data
    y_pos = np.ones(x_pos_seq.shape[0])
    y_neg = np.zeros(x_neg_seq.shape[0])
    original_len_pos = x_pos_seq.shape[0]
    original_len_neg = x_neg_seq.shape[0]
    x_pos_seq_test, y_pos_test, pos_indices = downsample_data(x_pos_seq, y_pos, fraction=test_size)
    x_neg_seq_test, y_neg_test, neg_indices = downsample_data(x_neg_seq, y_neg, fraction=test_size)
    
    # Determine training/validation indices (indices not in test set)
    pos_trainval_indices = np.setdiff1d(np.arange(x_pos_seq.shape[0]), pos_indices)
    neg_trainval_indices = np.setdiff1d(np.arange(x_neg_seq.shape[0]), neg_indices)
    #print(f'pos_trainval_indices: {pos_trainval_indices}')
    #print(f'neg_trainval_indices: {neg_trainval_indices}')

    x_pos_seq_trainval = x_pos_seq[pos_trainval_indices]
    x_neg_seq_trainval = x_neg_seq[neg_trainval_indices]
    
    # Crop data
    x_pos_seq_crop_trainval = np.concatenate((x_pos_seq_crop[pos_trainval_indices], x_pos_seq_crop[original_len_pos + pos_trainval_indices]), axis=0)
    x_neg_seq_crop_trainval = np.concatenate((x_neg_seq_crop[neg_trainval_indices], x_neg_seq_crop[original_len_neg + neg_trainval_indices]), axis=0)
    # Rev data
    x_pos_seq_rev_trainval = np.concatenate((x_pos_seq_rev[pos_trainval_indices], x_pos_seq_rev[original_len_pos + pos_trainval_indices], x_pos_seq_rev[2*original_len_pos + pos_trainval_indices]), axis=0)
    x_neg_seq_rev_trainval = np.concatenate((x_neg_seq_rev[neg_trainval_indices], x_neg_seq_rev[original_len_neg + neg_trainval_indices], x_neg_seq_rev[2*original_len_neg + neg_trainval_indices]), axis=0)
    
    # Combine positive and negative samples for the test set
    y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)
    x_test_noswap = np.concatenate((x_pos_seq_test, x_neg_seq_test), axis=0)
    x_test = np.swapaxes(x_test_noswap,2,1)
    testData_seqs = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).long())
    
    testData_indices = np.concatenate((pos_indices, original_len_pos + neg_indices), axis=0)
    
    if verbose==1:
        print('For testData size from split_testdata(): ', testData_seqs.tensors[0].shape[0])
        #print(f'testData_indices: {testData_indices}')
        #print(f'testData_indices sahpe: {testData_indices.shape}')

    return x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, testData_seqs, testData_indices, x_test_noswap, y_test


# In[5]:


def shuffleXY(x_seq,y):
    indices = np.arange(len(y))
    indices = np.random.permutation(indices)
    y=y[indices]
    x_seq=x_seq[indices]
    return x_seq,y


# In[ ]:


def shuffleXY_preVAE(x_seq, y, original_indices):
    indices = np.arange(len(y))
    indices = np.random.permutation(indices)
    y = y[indices]
    x_seq = x_seq[indices]
    original_indices = original_indices[indices]
    return x_seq, y, original_indices


# In[6]:


def genData(x_pos_seq,x_neg_seq,x_pos_seq_rev,x_neg_seq_rev,x_pos_seq_crop,x_neg_seq_crop,seed=1234):
    
    y_pos=np.ones(x_pos_seq.shape[0])
    y_neg=np.zeros(x_neg_seq.shape[0])
    y=np.concatenate((y_pos,y_neg),axis=0)
    x_seq=np.concatenate((x_pos_seq,x_neg_seq),axis=0)
    x_seq=np.swapaxes(x_seq,2,1)  

    y_pos_rev=np.ones(x_pos_seq_rev.shape[0])
    y_neg_rev=np.zeros(x_neg_seq_rev.shape[0])
    y_rev=np.concatenate((y_pos_rev,y_neg_rev),axis=0)
    x_seq_rev=np.concatenate((x_pos_seq_rev,x_neg_seq_rev),axis=0)
    x_seq_rev=np.swapaxes(x_seq_rev,2,1)

    y_pos_crop=np.ones(x_pos_seq_crop.shape[0])
    y_neg_crop=np.zeros(x_neg_seq_crop.shape[0])
    y_crop=np.concatenate((y_pos_crop,y_neg_crop),axis=0)
    x_seq_crop=np.concatenate((x_pos_seq_crop,x_neg_seq_crop),axis=0)
    x_seq_crop=np.swapaxes(x_seq_crop,2,1)

    np.random.seed(int(seed))
    x_seq,y=shuffleXY(x_seq,y)
    x_seq_rev,y_rev=shuffleXY(x_seq_rev,y_rev)
    x_seq_crop,y_crop=shuffleXY(x_seq_crop,y_crop)
    
    # print('true, pos and neg: ',[x_pos_seq.shape,x_neg_seq.shape])
    # print('rev, pos and neg: ',[x_pos_seq_rev.shape,x_neg_seq_rev.shape])
    # print('crop, pos and neg: ',[x_pos_seq_crop.shape,x_neg_seq_crop.shape])
    
    return y,x_seq,y_rev,x_seq_rev,y_crop,x_seq_crop


# In[ ]:


def genData_downsample(x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, seed=1234, fraction=1.0, verbose=0):
    np.random.seed(int(seed))
    
    # True data
    y_pos = np.ones(x_pos_seq.shape[0])
    y_neg = np.zeros(x_neg_seq.shape[0])
    original_len_pos = x_pos_seq.shape[0]
    original_len_neg = x_neg_seq.shape[0]
    x_pos_seq_downsample, y_pos_downsample, pos_indices = downsample_data(x_pos_seq, y_pos, fraction)
    x_neg_seq_downsample, y_neg_downsample, neg_indices = downsample_data(x_neg_seq, y_neg, fraction)
    
    
    # Generate the crop data based on the downsampled indices
    #x_pos_seq_crop = np.concatenate((x_pos_seq_crop[pos_indices], x_pos_seq_crop[len(x_pos_seq) + pos_indices]), axis=0)
    #x_neg_seq_crop = np.concatenate((x_neg_seq_crop[neg_indices], x_neg_seq_crop[len(x_neg_seq) + neg_indices]), axis=0)
    x_pos_seq_crop_downsample = np.concatenate((x_pos_seq_crop[pos_indices], x_pos_seq_crop[original_len_pos + pos_indices]), axis=0)
    x_neg_seq_crop_downsample = np.concatenate((x_neg_seq_crop[neg_indices], x_neg_seq_crop[original_len_neg + neg_indices]), axis=0)
    
    
    # Generate the rev data based on the downsampled indices
    x_pos_seq_rev_downsample = np.concatenate((x_pos_seq_rev[pos_indices], x_pos_seq_rev[original_len_pos + pos_indices], x_pos_seq_rev[2*original_len_pos + pos_indices]), axis=0)
    x_neg_seq_rev_downsample = np.concatenate((x_neg_seq_rev[neg_indices], x_neg_seq_rev[original_len_neg + neg_indices], x_neg_seq_rev[2*original_len_neg + neg_indices]), axis=0)
    

    y = np.concatenate((y_pos_downsample, y_neg_downsample), axis=0)
    x_seq = np.concatenate((x_pos_seq_downsample, x_neg_seq_downsample), axis=0)
    x_seq = np.swapaxes(x_seq,2,1)

    y_rev = np.concatenate((np.ones(x_pos_seq_rev_downsample.shape[0]), np.zeros(x_neg_seq_rev_downsample.shape[0])), axis=0)
    x_seq_rev = np.concatenate((x_pos_seq_rev_downsample, x_neg_seq_rev_downsample), axis=0)
    x_seq_rev = np.swapaxes(x_seq_rev,2,1)
    
    y_crop = np.concatenate((np.ones(x_pos_seq_crop_downsample.shape[0]), np.zeros(x_neg_seq_crop_downsample.shape[0])), axis=0)
    x_seq_crop = np.concatenate((x_pos_seq_crop_downsample, x_neg_seq_crop_downsample), axis=0)
    x_seq_crop = np.swapaxes(x_seq_crop,2,1)

    if verbose==1:
        print('For True after genData_downsample(), x_pos_seq_downsample and x_neg_seq_downsample shape: ', [x_pos_seq_downsample.shape, x_neg_seq_downsample.shape])
        print('For True after genData_downsample(), y and x_seq shape: ', [y.shape, x_seq.shape])
    
    return y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample


# In[3]:


def genData_downsample_old(x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, seed=1234, fraction=1.0, combine='true', verbose=0):
    
    np.random.seed(int(seed))
    
    # True data
    y_pos = np.ones(x_pos_seq.shape[0])
    y_neg = np.zeros(x_neg_seq.shape[0])
    original_len_pos = x_pos_seq.shape[0]
    original_len_neg = x_neg_seq.shape[0]
    x_pos_seq_downsample, y_pos_downsample, pos_indices = downsample_data(x_pos_seq, y_pos, fraction)
    x_neg_seq_downsample, y_neg_downsample, neg_indices = downsample_data(x_neg_seq, y_neg, fraction)
    
    
    # Generate the crop data based on the downsampled indices
    #x_pos_seq_crop = np.concatenate((x_pos_seq_crop[pos_indices], x_pos_seq_crop[len(x_pos_seq) + pos_indices]), axis=0)
    #x_neg_seq_crop = np.concatenate((x_neg_seq_crop[neg_indices], x_neg_seq_crop[len(x_neg_seq) + neg_indices]), axis=0)
    x_pos_seq_crop_downsample = np.concatenate((x_pos_seq_crop[pos_indices], x_pos_seq_crop[original_len_pos + pos_indices]), axis=0)
    x_neg_seq_crop_downsample = np.concatenate((x_neg_seq_crop[neg_indices], x_neg_seq_crop[original_len_neg + neg_indices]), axis=0)
    
    
    # Generate the rev data based on the downsampled indices
    if combine=='true':
        x_pos_seq_rev_downsample = x_pos_seq_rev[pos_indices]
        x_neg_seq_rev_downsample = x_neg_seq_rev[neg_indices]
    elif combine=='VAE':
        x_pos_seq_rev_downsample = x_pos_seq_rev[pos_indices]
        x_neg_seq_rev_downsample = x_neg_seq_rev[neg_indices]
    elif combine=='rev':
        x_pos_seq_rev_downsample = x_pos_seq_rev[pos_indices]
        x_neg_seq_rev_downsample = x_neg_seq_rev[neg_indices]
    elif combine=='crop':
        x_pos_seq_rev_downsample = x_pos_seq_rev[pos_indices]
        x_neg_seq_rev_downsample = x_neg_seq_rev[neg_indices]
    elif combine=='true+rev':
        x_pos_seq_rev_downsample = x_pos_seq_rev[pos_indices]
        x_neg_seq_rev_downsample = x_neg_seq_rev[neg_indices]
    elif combine=='true+crop':
        x_pos_seq_rev_downsample = x_pos_seq_rev[pos_indices]
        x_neg_seq_rev_downsample = x_neg_seq_rev[neg_indices]
    elif combine=='true+rev+crop':  
        x_pos_seq_rev_downsample = np.concatenate((x_pos_seq_rev[pos_indices], x_pos_seq_rev[original_len_pos + pos_indices], x_pos_seq_rev[2*original_len_pos + pos_indices]), axis=0)
        x_neg_seq_rev_downsample = np.concatenate((x_neg_seq_rev[neg_indices], x_neg_seq_rev[original_len_neg + neg_indices], x_neg_seq_rev[2*original_len_neg + neg_indices]), axis=0)
    elif combine=='Semi':
        x_pos_seq_rev_downsample = np.concatenate((x_pos_seq_rev[pos_indices], x_pos_seq_rev[original_len_pos + pos_indices], x_pos_seq_rev[2*original_len_pos + pos_indices]), axis=0)
        x_neg_seq_rev_downsample = np.concatenate((x_neg_seq_rev[neg_indices], x_neg_seq_rev[original_len_neg + neg_indices], x_neg_seq_rev[2*original_len_neg + neg_indices]), axis=0)
    elif combine=='Semi+truecroprev_v1':
        x_pos_seq_rev_downsample = np.concatenate((x_pos_seq_rev[pos_indices], x_pos_seq_rev[original_len_pos + pos_indices], x_pos_seq_rev[2*original_len_pos + pos_indices]), axis=0)
        x_neg_seq_rev_downsample = np.concatenate((x_neg_seq_rev[neg_indices], x_neg_seq_rev[original_len_neg + neg_indices], x_neg_seq_rev[2*original_len_neg + neg_indices]), axis=0)
    

    y = np.concatenate((y_pos_downsample, y_neg_downsample), axis=0)
    x_seq = np.concatenate((x_pos_seq_downsample, x_neg_seq_downsample), axis=0)
    x_seq = np.swapaxes(x_seq,2,1)
    x_seq, y = shuffleXY(x_seq, y)
    
    y_rev = np.concatenate((np.ones(x_pos_seq_rev_downsample.shape[0]), np.zeros(x_neg_seq_rev_downsample.shape[0])), axis=0)
    x_seq_rev = np.concatenate((x_pos_seq_rev_downsample, x_neg_seq_rev_downsample), axis=0)
    x_seq_rev = np.swapaxes(x_seq_rev,2,1)
    x_seq_rev, y_rev = shuffleXY(x_seq_rev, y_rev)
    
    y_crop = np.concatenate((np.ones(x_pos_seq_crop_downsample.shape[0]), np.zeros(x_neg_seq_crop_downsample.shape[0])), axis=0)
    x_seq_crop = np.concatenate((x_pos_seq_crop_downsample, x_neg_seq_crop_downsample), axis=0)
    x_seq_crop = np.swapaxes(x_seq_crop,2,1)
    x_seq_crop, y_crop = shuffleXY(x_seq_crop, y_crop)

    if verbose==1:
        print('For True after genData_downsample(), x_pos_seq_downsample and x_neg_seq_downsample shape: ', [x_pos_seq_downsample.shape, x_neg_seq_downsample.shape])
        print('For True after genData_downsample(), y and x_seq shape: ', [y.shape, x_seq.shape])
    
    return y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_pos_seq_downsample, x_neg_seq_downsample


# In[ ]:


def genTrainData(y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, random_state, combine='true', verbose=0, train_fraction=0.75):
    
    # for true
    trainData_seq, valData_seq, ytrain, yval = train_test_split(x_seq, y, stratify=y, test_size= 1-train_fraction, random_state=random_state)
    #print('For true, Train, Val and Test Shape: ', [trainData_seq.shape, valData_seq.shape])
    
    # for rev
    trainData_seq_rev, valData_seq_rev, ytrain_rev, yval_rev = train_test_split(x_seq_rev, y_rev, stratify=y_rev, test_size=1-train_fraction, random_state=random_state)
    #print('For rev, Train, Val Shape: ', [trainData_seq_rev.shape, valData_seq_rev.shape])

    # for crop
    trainData_seq_crop, valData_seq_crop, ytrain_crop, yval_crop = train_test_split(x_seq_crop, y_crop, stratify=y_crop, test_size=1-train_fraction, random_state=random_state)
    #print('For crop, Train, Val Shape: ', [trainData_seq_crop.shape, valData_seq_crop.shape])
    


    if combine in ["true", "Semi"]:
        trainData_seqs=trainData_seq
        valData_seqs=valData_seq
        ytrains=ytrain
        yvals=yval
    elif combine=='true+rev':
        trainData_seqs=np.concatenate([trainData_seq,trainData_seq_rev],axis=0)
        ytrains=np.concatenate([ytrain,ytrain_rev])
        valData_seqs=np.concatenate([valData_seq,valData_seq_rev],axis=0)
        yvals=np.concatenate([yval,yval_rev])
    elif combine=='true+crop':
        trainData_seqs=np.concatenate([trainData_seq,trainData_seq_crop],axis=0)
        ytrains=np.concatenate([ytrain,ytrain_crop])
        valData_seqs=np.concatenate([valData_seq,valData_seq_crop],axis=0)
        yvals=np.concatenate([yval,yval_crop])
    elif combine in ['true+rev+crop', "Semi+truecroprev_v1", "Semi+truecroprev_v2"]:
        trainData_seqs=np.concatenate((trainData_seq, trainData_seq_rev, trainData_seq_crop),axis=0)
        ytrains=np.concatenate((ytrain, ytrain_rev, ytrain_crop))
        valData_seqs=np.concatenate((valData_seq, valData_seq_rev, valData_seq_crop),axis=0)
        yvals=np.concatenate([yval, yval_rev, yval_crop])
    else:
        raise ValueError(f"Unexpected value for 'combine': {combine}")
        

    if verbose==1:
        #print('For combined Train, Val and Test shape: ',[trainData_seqs.shape, valData_seqs.shape])
        print('For trainData, valData shape from genTrainData(): ',[trainData_seqs.shape, valData_seqs.shape])

    trainData = TensorDataset(torch.from_numpy(trainData_seqs), torch.from_numpy(ytrains).long())
    valData = TensorDataset(torch.from_numpy(valData_seqs), torch.from_numpy(yvals).long())

    return trainData, valData


# In[2]:


def genTrainData_old(y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, random_state, combine, verbose=0):
    
    # for true
    trainData_seq, xx, ytrain, yy = train_test_split(x_seq, y, stratify=y, test_size=0.4, random_state=random_state)
    testData_seq, valData_seq, ytest, yval = train_test_split(xx, yy, stratify=yy, test_size=0.5, random_state=random_state)
    #print('For true, Train, Val and Test Shape: ', [trainData_seq.shape, valData_seq.shape, testData_seq.shape])
    
    # for rev
    trainData_seq_rev, valData_seq_rev, ytrain_rev, yval_rev = train_test_split(x_seq_rev, y_rev, stratify=y_rev, test_size=0.25, random_state=random_state)
    #print('For rev, Train, Val Shape: ', [trainData_seq_rev.shape, valData_seq_rev.shape])

    # for crop
    trainData_seq_crop, valData_seq_crop, ytrain_crop, yval_crop = train_test_split(x_seq_crop, y_crop, stratify=y_crop, test_size=0.25, random_state=random_state)
    #print('For crop, Train, Val Shape: ', [trainData_seq_crop.shape, valData_seq_crop.shape])
    
    if combine=='true':
        trainData_seqs=trainData_seq
        valData_seqs=valData_seq
        ytrains=ytrain
        yvals=yval
    elif combine=='Semi':
        trainData_seqs=trainData_seq
        valData_seqs=valData_seq
        ytrains=ytrain
        yvals=yval
    elif combine=='VAE':
        trainData_seqs=trainData_seq
        valData_seqs=valData_seq
        ytrains=ytrain
        yvals=yval
    elif combine=='rev':
        trainData_seqs=trainData_seq_rev
        valData_seqs=valData_seq_rev
        ytrains=ytrain_rev
        yvals=yval_rev
    elif combine=='crop':
        trainData_seqs=trainData_seq_crop
        valData_seqs=valData_seq_crop
        ytrains=ytrain_crop
        yvals=yval_crop
    elif combine=='true+rev':
        trainData_seqs=np.concatenate([trainData_seq,trainData_seq_rev],axis=0)
        ytrains=np.concatenate([ytrain,ytrain_rev])
        valData_seqs=np.concatenate([valData_seq,valData_seq_rev],axis=0)
        yvals=np.concatenate([yval,yval_rev])
    elif combine=='true+crop':
        trainData_seqs=np.concatenate([trainData_seq,trainData_seq_crop],axis=0)
        ytrains=np.concatenate([ytrain,ytrain_crop])
        valData_seqs=np.concatenate([valData_seq,valData_seq_crop],axis=0)
        yvals=np.concatenate([yval,yval_crop])
    elif combine=='true+rev+crop':
        trainData_seqs=np.concatenate((trainData_seq,trainData_seq_rev,trainData_seq_crop),axis=0)
        ytrains=np.concatenate((ytrain,ytrain_rev,ytrain_crop))
        valData_seqs=np.concatenate((valData_seq,valData_seq_rev,valData_seq_crop),axis=0)
        yvals=np.concatenate([yval,yval_rev,yval_crop])
    elif combine=='Semi+truecroprev_v1':
        trainData_seqs=np.concatenate((trainData_seq,trainData_seq_rev,trainData_seq_crop),axis=0)
        ytrains=np.concatenate((ytrain,ytrain_rev,ytrain_crop))
        valData_seqs=np.concatenate((valData_seq,valData_seq_rev,valData_seq_crop),axis=0)
        yvals=np.concatenate([yval,yval_rev,yval_crop])
        
    
    trainData_seqs,ytrains=shuffleXY(trainData_seqs,ytrains)
    valData_seqs,yvals=shuffleXY(valData_seqs,yvals)
    testData_seq,ytest=shuffleXY(testData_seq,ytest)

    if verbose==1:
        #print('For combined Train, Val and Test shape: ',[trainData_seqs.shape, valData_seqs.shape,testData_seq.shape])
        print('For Train, Val and Test shape from genTrainData(): ',[trainData_seqs.shape, valData_seqs.shape, testData_seq.shape])

    trainData_seqs = TensorDataset(torch.from_numpy(trainData_seqs), torch.from_numpy(ytrains).long())
    valData_seqs = TensorDataset(torch.from_numpy(valData_seqs), torch.from_numpy(yvals).long())
    testData_seqs = TensorDataset(torch.from_numpy(testData_seq), torch.from_numpy(ytest).long())
    

    return trainData_seqs,valData_seqs,testData_seqs


# In[ ]:


def genTrainData_vae(y, x_seq, y_vae, x_seq_vae, random_state, verbose=0):
    
    # for true
    trainData_seq, valData_seq, ytrain, yval = train_test_split(x_seq, y, stratify=y, test_size=0.25, random_state=random_state)

    # for vae
    trainData_seq_vae, valData_seq_vae, ytrain_vae, yval_vae = train_test_split(x_seq_vae, y_vae, stratify=y_vae, test_size=0.25, random_state=random_state)
    #print('For vae, Train, Val Shape: ', [trainData_seq_vae.shape, valData_seq_vae.shape])
    
    trainData_seqs=np.concatenate([trainData_seq,trainData_seq_vae],axis=0)
    ytrains=np.concatenate([ytrain,ytrain_vae])
    valData_seqs=np.concatenate([valData_seq,valData_seq_vae],axis=0)
    yvals=np.concatenate([yval,yval_vae])      
    
    trainData_seqs,ytrains=shuffleXY(trainData_seqs,ytrains)
    valData_seqs,yvals=shuffleXY(valData_seqs,yvals)

    if verbose==1:
        #print('For combined Train, Val and Test shape: ',[trainData_seqs.shape, valData_seqs.shape,testData_seq.shape])
        print('For Train, Val shape from genTrainData_vae(): ',[trainData_seqs.shape, valData_seqs.shape])

    trainData_seqs = TensorDataset(torch.from_numpy(trainData_seqs), torch.from_numpy(ytrains).long())
    valData_seqs = TensorDataset(torch.from_numpy(valData_seqs), torch.from_numpy(yvals).long())
    
    return trainData_seqs, valData_seqs


# In[ ]:


def semi_supervised_learning(celltype, y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_unlabeled_seq, random_state, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, num_kernels, dropout_rate, iteration, combine):
    print(f'Start semi_supervised learning method: {combine}')
    trainData_semi, valData_semi = genTrainData(y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, random_state, combine=combine, verbose=(iteration == 0), train_fraction=0.75)
            
    original_truelabeled_samplesize = x_seq.shape[0]
                
    if iteration < 1:
        print('Before Semi-supervised Learning, the true labeled sample size: ', original_truelabeled_samplesize)
                
    x_unlabeled_seq_original = x_unlabeled_seq
    y_unlabeled_semi = np.zeros(x_unlabeled_seq_original.shape[0])
    x_unlabeled_seq_semi = np.swapaxes(x_unlabeled_seq_original, 2, 1)
    unlabeledData_seqs = TensorDataset(torch.from_numpy(x_unlabeled_seq_semi), torch.from_numpy(y_unlabeled_semi).long())
            
#    combine = 'Semi'
    model_savename = combine + celltype + '.pth'
    model_semi = trainModel(trainData_semi, valData_semi, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                
    combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0], unlabeledData_seqs.tensors[0]])
    combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1], unlabeledData_seqs.tensors[1]])
    combinedData_seqs = TensorDataset(combined_x, combined_y)
                
    predsProb = testModel(model_semi, model_savename, combinedData_seqs, BATCH_SIZE, verbose=0, predonly=1)
    predsProb = np.exp(predsProb)
                
    true_probs = predsProb[:len(trainData_semi) + len(valData_semi)]
    quantile_90 = np.percentile(true_probs, 90)
    quantile_10 = np.percentile(true_probs, 10)
    unlabeled_probs = predsProb[len(trainData_semi) + len(valData_semi):]
    
    # set initial value to avoid error if early stop at iteration1
    newly_combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0]])
    newly_combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1]])
                
    MAX_ITERATIONS_semi = 20
    iteration_semi = 1
    while iteration_semi <= MAX_ITERATIONS_semi:
        #print(f"iteration_semi {iteration_semi}")
                    
        high_confidence_positive = np.where(unlabeled_probs > quantile_90)[0]
        high_confidence_negative = np.where(unlabeled_probs < quantile_10)[0]
                    
        if len(high_confidence_positive) == 0 and len(high_confidence_negative) == 0:
            print(f'semi early stopped at iteration_semi: ',iteration_semi)
            break
                        
        y_unlabeled_semi[high_confidence_positive] = 1
        y_unlabeled_semi[high_confidence_negative] = 0
                
        #print("trainData_semi and valData_semi shape: ",[trainData_semi.tensors[0].shape, valData_semi.tensors[0].shape])
        newly_labeled_x = x_unlabeled_seq_semi[np.concatenate([high_confidence_positive, high_confidence_negative])]
        newly_labeled_y = y_unlabeled_semi[np.concatenate([high_confidence_positive, high_confidence_negative])]
        newly_combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0], torch.from_numpy(newly_labeled_x)])
        newly_combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1], torch.from_numpy(newly_labeled_y)])
        newly_combinedData_seqs = TensorDataset(newly_combined_x, newly_combined_y)
        print(f'newly_combined_x.shape[0]: {newly_combined_x.shape[0]}')
                    
        high_conf_positive_data_x = torch.from_numpy(x_unlabeled_seq_semi[high_confidence_positive])
        high_conf_positive_data_y = torch.from_numpy(y_unlabeled_semi[high_confidence_positive]).long()

        high_conf_negative_data_x = torch.from_numpy(x_unlabeled_seq_semi[high_confidence_negative])
        high_conf_negative_data_y = torch.from_numpy(y_unlabeled_semi[high_confidence_negative]).long()

        mask_positive = np.ones(x_unlabeled_seq_semi.shape[0], dtype=bool)
        mask_negative = np.ones(x_unlabeled_seq_semi.shape[0], dtype=bool)

        mask_positive[high_confidence_positive] = False
        mask_negative[high_confidence_negative] = False

        remaining_unlabeled_data_x = torch.from_numpy(x_unlabeled_seq_semi[mask_positive & mask_negative])
        remaining_unlabeled_data_y = torch.from_numpy(y_unlabeled_semi[mask_positive & mask_negative]).long()
                    
        combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0], high_conf_positive_data_x, high_conf_negative_data_x, remaining_unlabeled_data_x])
        combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1], high_conf_positive_data_y, high_conf_negative_data_y, remaining_unlabeled_data_y])
        combinedData_seqs = TensorDataset(combined_x, combined_y)
                    
        x_seq_semiupdated, y_semiupdated = newly_combinedData_seqs.tensors
        x_seq_semiupdated = x_seq_semiupdated.numpy()
        y_semiupdated = y_semiupdated.numpy()

        train_features_np, val_features_np, train_labels_np, val_labels_np = train_test_split(x_seq_semiupdated, y_semiupdated, test_size=0.25, random_state=random_state, stratify=y_semiupdated)
                    
        train_features = torch.from_numpy(train_features_np)
        train_labels = torch.from_numpy(train_labels_np).long()
        val_features = torch.from_numpy(val_features_np)
        val_labels = torch.from_numpy(val_labels_np).long()

        trainData_semi = TensorDataset(train_features, train_labels)
        valData_semi = TensorDataset(val_features, val_labels)

        model_semi = trainModel(trainData_semi, valData_semi, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
    
        predsProb = testModel(model_semi, model_savename, combinedData_seqs, BATCH_SIZE, verbose=0, predonly=1)
        predsProb = np.exp(predsProb)
    
        true_probs = predsProb[:len(newly_combinedData_seqs)]
        quantile_90 = np.percentile(true_probs, 90)
        quantile_10 = np.percentile(true_probs, 10)
        unlabeled_probs = predsProb[len(newly_combinedData_seqs):]
                
        unlabeledData_seqs = TensorDataset(remaining_unlabeled_data_x, remaining_unlabeled_data_y)
        y_unlabeled_semi = y_unlabeled_semi[mask_positive & mask_negative]
        x_unlabeled_seq_semi = x_unlabeled_seq_semi[mask_positive & mask_negative]
            
        iteration_semi += 1
        
    print(f'End semi_supervised learning method: {combine}')
    return newly_combined_x, model_semi


# In[ ]:


def calculate_average_sample_sizes_interlaced(sample_sizes, num_iterations, fractions):
    fraction_averages = {}

    for i, fraction in enumerate(fractions):
        fraction_samples = sample_sizes[i::len(fractions)]
        average_size = sum(fraction_samples) / num_iterations
        fraction_averages[fraction] = average_size

    return fraction_averages


# In[9]:


def plotMetric(metrics,metricname):

    get_ipython().run_line_magic('matplotlib', 'inline')

    combines=['true','true+rev','true+crop','true+rev+crop']

    # Create boxplots for each column
    plt.boxplot(metrics, labels=combines)

    # Customize the plot (add title, y-axis label, etc.)
    plt.xlabel('Method')
    plt.ylabel('Values')
    plt.title(metricname)

    # Show the plot
    plt.show()


# In[3]:


def plot_results(celltype, results):
    combination_colors = {
        'true': '#808080', 
        'true+rev': '#A9A9A9',
        'true+crop': '#C0C0C0',
        'true+rev+crop': '#D3D3D3',
        'VAE': '#32CD32',
        'VAE1.0': '#32CD32',
        'Semi': '#4169E1',
        'Semi+truecroprev_v1': '#FF6347',
        'Semi+truecroprev_v2': '#FFFF00',
        'Score13': '#E5E5E5',
        'Score14': '#696969',
        'Score11': '#BEBEBE',
        'Score12': '#989898',
        'Score19': '#787878',
        'Random_forest': '#787878',
        'VAE_notrimer': '#00CED1',
        'VAE0.5': '#8A2BE2',
        'VAE0.1': '#FA8072'
    }
    fractions_float = [float(fraction) for fraction in fractions_json]
#    baseline_aucs = {}
#    for fraction in fractions_float:
#        baseline_auc = RF_baseline_auc_with_5foldCV(celltype, idata, fraction=fraction)
#        baseline_aucs[fraction] = baseline_auc

################################# AUC
    all_aucs = []
    for fraction in fractions_json:
        for combine in combinations:
            all_aucs.extend(results[celltype][fraction][combine]['AUC'])
    global_min_auc = min(all_aucs)
    global_max_auc = max(all_aucs)

    # Add padding
    auc_range = global_max_auc - global_min_auc
    padding = auc_range * 0.05
    global_min_auc -= padding
    global_max_auc += padding

    fig, axs = plt.subplots(len(fractions_json), 1, figsize=(6, 7))
    fig.suptitle(f'AUC Results for {celltype}', fontsize=16, y=0.98)

    for i, fraction in enumerate(fractions_json):
        ax = axs[i]

        median_aucs = {combine: np.median(results[celltype][fraction][combine]['AUC']) for combine in combinations}
        sorted_combinations = sorted(combinations, key=lambda x: median_aucs[x], reverse=False)

        data_to_plot = [results[celltype][fraction][combine]['AUC'] for combine in sorted_combinations]

        bplot = ax.boxplot(data_to_plot, patch_artist=True, vert=False)

        for patch, combine in zip(bplot['boxes'], sorted_combinations):
            patch.set_facecolor(combination_colors[combine])

        ax.set_title(f'Fraction: {fraction}')
        ax.set_yticklabels(sorted_combinations, ha='right')
        if i == len(fractions_json) - 1:
            ax.set_xlabel('AUC')

        ax.set_xlim(global_min_auc, global_max_auc)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
#        baseline_auc_value = baseline_aucs[float(fraction)]
#        ax.axvline(x=baseline_auc_value, color='r', linestyle='--', linewidth=1.5, label='Baseline AUC')
    
    plt.tight_layout()
    fig.savefig(f"{celltype}_AUC_results.png", dpi=300, bbox_inches='tight')
    plt.show()
############################# AUPRC    
    all_auprcs = []
    for fraction in fractions_json:
        for combine in combinations:
            all_auprcs.extend(results[celltype][fraction][combine]['AUPRC'])
    global_min_auprc = min(all_auprcs)
    global_max_auprc = max(all_auprcs)

    # Add padding
    auprc_range = global_max_auprc - global_min_auprc
    padding = auprc_range * 0.05
    global_min_auprc -= padding
    global_max_auprc += padding

    fig, axs = plt.subplots(len(fractions_json), 1, figsize=(6, 7))
    fig.suptitle(f'AUPRC Results for {celltype}', fontsize=16, y=0.98)

    for i, fraction in enumerate(fractions_json):
        ax = axs[i]

        median_auprcs = {combine: np.median(results[celltype][fraction][combine]['AUPRC']) for combine in combinations}
        sorted_combinations = sorted(combinations, key=lambda x: median_auprcs[x], reverse=False)

        data_to_plot = [results[celltype][fraction][combine]['AUPRC'] for combine in sorted_combinations]

        bplot = ax.boxplot(data_to_plot, patch_artist=True, vert=False)

        for patch, combine in zip(bplot['boxes'], sorted_combinations):
            patch.set_facecolor(combination_colors[combine])

        ax.set_title(f'Fraction: {fraction}')
        ax.set_yticklabels(sorted_combinations, ha='right')
        if i == len(fractions_json) - 1:
            ax.set_xlabel('AUPRC')

        ax.set_xlim(global_min_auprc, global_max_auprc)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
#        baseline_auprc_value = baseline_auprcs[float(fraction)]
#        ax.axvline(x=baseline_auprc_value, color='r', linestyle='--', linewidth=1.5, label='Baseline AUPRC')
    
    plt.tight_layout()
    fig.savefig(f"{celltype}_AUPRC_results.png", dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:


def plot_results_auprc(celltype, results):
    combination_colors = {
        'true': '#808080', 
        'true+rev': '#A9A9A9',
        'true+crop': '#C0C0C0',
        'true+rev+crop': '#D3D3D3',
        'VAE': '#32CD32',
        'VAE1.0': '#32CD32',
        'Semi': '#4169E1',
        'Semi+truecroprev_v1': '#FF6347',
        'Semi+truecroprev_v2': '#FFFF00',
        'Score13': '#E5E5E5',
        'Score14': '#696969',
        'Score11': '#BEBEBE',
        'Score12': '#989898',
        'Score19': '#787878',
        'Random_forest': '#787878',
        'FATHMM.MKL': '#1e1e1e',
        'FATHMM.XF': '#232323',
        'CADD': '#282828',
        'LINSIGHT': '#2d2d2d',
        'FIRE': '#323232',
        'ncER': '#373737',
        'PAFA': '#3c3c3c',
        'CScape': '#414141',
        'ReMM': '#464646',
        'fitCons': '#4b4b4b',
        'FitCons2': '#505050',
        'DVAR': '#555555',
        'FunSeq2': '#5a5a5a',
        'CDTS': '#5f5f5f',
        'Orion': '#646464',
        'GenoCanyon': '#6e6e6e',
        'VAE_notrimer': '#00CED1',
        'VAE0.5': '#8A2BE2',
        'VAE0.1': '#FA8072'
        
    }
    fractions_float = [float(fraction) for fraction in fractions_json]
#    baseline_aucs = {}
#    for fraction in fractions_float:
#        baseline_auc = RF_baseline_auc_with_5foldCV(celltype, idata, fraction=fraction)
#        baseline_aucs[fraction] = baseline_auc

############################# AUPRC    
    all_auprcs = []
    for fraction in fractions_json:
        for combine in combinations:
            all_auprcs.extend(results[celltype][fraction][combine]['AUPRC'])
    global_min_auprc = min(all_auprcs)
    global_max_auprc = max(all_auprcs)

    # Add padding
    auprc_range = global_max_auprc - global_min_auprc
    padding = auprc_range * 0.05
    global_min_auprc -= padding
    global_max_auprc += padding

    fig, axs = plt.subplots(len(fractions_json), 1, figsize=(6, 7))
    fig.suptitle(f'AUPRC Results for {celltype}', fontsize=16, y=0.98)

    for i, fraction in enumerate(fractions_json):
        ax = axs[i]

        median_auprcs = {combine: np.median(results[celltype][fraction][combine]['AUPRC']) for combine in combinations}
        sorted_combinations = sorted(combinations, key=lambda x: median_auprcs[x], reverse=False)

        data_to_plot = [results[celltype][fraction][combine]['AUPRC'] for combine in sorted_combinations]

        bplot = ax.boxplot(data_to_plot, patch_artist=True, vert=False)

        for patch, combine in zip(bplot['boxes'], sorted_combinations):
            patch.set_facecolor(combination_colors[combine])

        ax.set_title(f'Fraction: {fraction}')
        ax.set_yticklabels(sorted_combinations, ha='right')
        if i == len(fractions_json) - 1:
            ax.set_xlabel('AUPRC')

        ax.set_xlim(global_min_auprc, global_max_auprc)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
#        baseline_auprc_value = baseline_auprcs[float(fraction)]
#        ax.axvline(x=baseline_auprc_value, color='r', linestyle='--', linewidth=1.5, label='Baseline AUPRC')
    
    plt.tight_layout()
    fig.savefig(f"{celltype}_AUPRC_results.png", dpi=300, bbox_inches='tight')
    plt.show()


# In[1]:


def process_fractions_allmethods(celltype, fractions, combinations, x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, x_unlabeled_seq,
                                 dropout_rate, num_kernels, BATCH_SIZE,INIT_LR, early_stop_thresh, EPOCHS, NUM_ITERATIONS, results, input_dir, output_dir, 
                                 data_folder, device, random_state):
    
    final_sample_sizes_semi = []
    final_sample_sizes_semitruecroprev_v1 = []
    final_sample_sizes_semitruecroprev_v2 = []
    final_sample_size_increases_semi = []
    final_sample_size_increases_semitruecroprev_v1 = []
    final_sample_size_increases_semitruecroprev_v2 = []
    
    testData_indices_df = pd.DataFrame()

    for iteration in range(NUM_ITERATIONS):
        current_seed = random_state + iteration
        np.random.seed(int(current_seed))
        
        x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
            x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, test_size=0.2, seed=current_seed, verbose=1)
        
        iteration_df = pd.DataFrame(testData_indices.reshape(-1, 1), columns=[f'Iteration_{iteration + 1}'])
        testData_indices_df = pd.concat([testData_indices_df, iteration_df], axis=1)
        
        for fraction in fractions:
            print('######################### Downsample fraction:', fraction, '###########################')

            y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
                x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, seed=current_seed, fraction=fraction, verbose=(iteration == 0))

            #Random Forest codes            
            x_pos_downsample_sequence = onehot_to_seq(x_pos_seq_downsample)
            x_neg_downsample_sequence = onehot_to_seq(x_neg_seq_downsample)
            x_test_sequence = onehot_to_seq(x_test_noswap)
            x_train_sequence = np.concatenate((x_pos_downsample_sequence, x_neg_downsample_sequence), axis=0)
            y_train = np.concatenate((y_pos_downsample, y_neg_downsample), axis=0)

            import itertools
            x_train = sequence_to_kmer_features(x_train_sequence)
            x_test = sequence_to_kmer_features(x_test_sequence)

            # Train Random Forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(x_train, y_train)

            y_pred_proba = clf.predict_proba(x_test)[:, 1]

            auc_baseline = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC: {auc_baseline}")

            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc_baseline = auc(recall, precision)
            print(f"AUPRC: {auprc_baseline}")

######################## Semi-supervised learning using only 'true' data('Semi', 'Semi+truecroprev_v1')           
            # only use 'true' labeled data for semi-supervised learning
            start_time = time.time()
        
            trainData_semi, valData_semi = genTrainData(y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, current_seed, combine='true', verbose=(iteration == 0), train_fraction=0.75)
            
            original_truelabeled_samplesize = x_seq.shape[0]
                
            if iteration < 1:
                print('Before Semi-supervised Learning, the true labeled sample size: ', original_truelabeled_samplesize)
                
            x_unlabeled_seq_original = x_unlabeled_seq
            y_unlabeled_semi = np.zeros(x_unlabeled_seq_original.shape[0])
            x_unlabeled_seq_semi = np.swapaxes(x_unlabeled_seq_original, 2, 1)
            unlabeledData_seqs = TensorDataset(torch.from_numpy(x_unlabeled_seq_semi), torch.from_numpy(y_unlabeled_semi).long())
            
            combine = 'Semi'
            model_savename = combine + celltype + '.pth'
            model_semi = trainModel(trainData_semi, valData_semi, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                
            combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0], unlabeledData_seqs.tensors[0]])
            combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1], unlabeledData_seqs.tensors[1]])
            combinedData_seqs = TensorDataset(combined_x, combined_y)
                
            predsProb = testModel(model_semi, model_savename, combinedData_seqs, BATCH_SIZE, verbose=0, predonly=1)
            predsProb = np.exp(predsProb)
                
            true_probs = predsProb[:len(trainData_semi) + len(valData_semi)]
            quantile_90 = np.percentile(true_probs, 90)
            quantile_10 = np.percentile(true_probs, 10)
            unlabeled_probs = predsProb[len(trainData_semi) + len(valData_semi):]
            
            # set initial value to avoid error if early stop at iteration1
            newly_combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0]])
            newly_combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1]])
                
            MAX_ITERATIONS_semi = 20
            iteration_semi = 1
            while iteration_semi <= MAX_ITERATIONS_semi:
                #print(f"iteration_semi {iteration_semi}")
                    
                high_confidence_positive = np.where(unlabeled_probs > quantile_90)[0]
                high_confidence_negative = np.where(unlabeled_probs < quantile_10)[0]
                    
                if len(high_confidence_positive) == 0 and len(high_confidence_negative) == 0:
                    print(f'semi early stopped at iteration_semi: ',iteration_semi)
                    break
                        
                y_unlabeled_semi[high_confidence_positive] = 1
                y_unlabeled_semi[high_confidence_negative] = 0
                
                #print("trainData_semi and valData_semi shape: ",[trainData_semi.tensors[0].shape, valData_semi.tensors[0].shape])
                newly_labeled_x = x_unlabeled_seq_semi[np.concatenate([high_confidence_positive, high_confidence_negative])]
                newly_labeled_y = y_unlabeled_semi[np.concatenate([high_confidence_positive, high_confidence_negative])]
                newly_combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0], torch.from_numpy(newly_labeled_x)])
                newly_combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1], torch.from_numpy(newly_labeled_y)])
                newly_combinedData_seqs = TensorDataset(newly_combined_x, newly_combined_y)
                print(f'newly_combined_x.shape[0]: {newly_combined_x.shape[0]}')
                    
                high_conf_positive_data_x = torch.from_numpy(x_unlabeled_seq_semi[high_confidence_positive])
                high_conf_positive_data_y = torch.from_numpy(y_unlabeled_semi[high_confidence_positive]).long()

                high_conf_negative_data_x = torch.from_numpy(x_unlabeled_seq_semi[high_confidence_negative])
                high_conf_negative_data_y = torch.from_numpy(y_unlabeled_semi[high_confidence_negative]).long()

                mask_positive = np.ones(x_unlabeled_seq_semi.shape[0], dtype=bool)
                mask_negative = np.ones(x_unlabeled_seq_semi.shape[0], dtype=bool)

                mask_positive[high_confidence_positive] = False
                mask_negative[high_confidence_negative] = False

                remaining_unlabeled_data_x = torch.from_numpy(x_unlabeled_seq_semi[mask_positive & mask_negative])
                remaining_unlabeled_data_y = torch.from_numpy(y_unlabeled_semi[mask_positive & mask_negative]).long()
                    
                combined_x = torch.cat([trainData_semi.tensors[0], valData_semi.tensors[0], high_conf_positive_data_x, high_conf_negative_data_x, remaining_unlabeled_data_x])
                combined_y = torch.cat([trainData_semi.tensors[1], valData_semi.tensors[1], high_conf_positive_data_y, high_conf_negative_data_y, remaining_unlabeled_data_y])
                combinedData_seqs = TensorDataset(combined_x, combined_y)
                    
                x_seq_semiupdated, y_semiupdated = newly_combinedData_seqs.tensors
                x_seq_semiupdated = x_seq_semiupdated.numpy()
                y_semiupdated = y_semiupdated.numpy()

                train_features_np, val_features_np, train_labels_np, val_labels_np = train_test_split(x_seq_semiupdated, y_semiupdated, test_size=0.25, random_state=current_seed, stratify=y_semiupdated)
                    
                train_features = torch.from_numpy(train_features_np)
                train_labels = torch.from_numpy(train_labels_np).long()
                val_features = torch.from_numpy(val_features_np)
                val_labels = torch.from_numpy(val_labels_np).long()

                trainData_semi = TensorDataset(train_features, train_labels)
                valData_semi = TensorDataset(val_features, val_labels)

                model_semi = trainModel(trainData_semi, valData_semi, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
    
                predsProb = testModel(model_semi, model_savename, combinedData_seqs, BATCH_SIZE, verbose=0, predonly=1)
                predsProb = np.exp(predsProb)
    
                true_probs = predsProb[:len(newly_combinedData_seqs)]
                quantile_90 = np.percentile(true_probs, 90)
                quantile_10 = np.percentile(true_probs, 10)
                unlabeled_probs = predsProb[len(newly_combinedData_seqs):]
                
                unlabeledData_seqs = TensorDataset(remaining_unlabeled_data_x, remaining_unlabeled_data_y)
                y_unlabeled_semi = y_unlabeled_semi[mask_positive & mask_negative]
                x_unlabeled_seq_semi = x_unlabeled_seq_semi[mask_positive & mask_negative]
            
                iteration_semi += 1
            #print(f"Before Semi+truecroprev_v2 y_semiupdated:{y_semiupdated}")
            #print(f"Before Semi+truecroprev_v2 y_semiupdated shape:{y_semiupdated.shape}")
            elapsed_time = time.time() - start_time
            print(f'Elapsed time for Semi in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
########################
######################## Semi-supervised learning using 'truecroprev' data ('Semi+truecroprev_v2')
            start_time = time.time()
            newly_combined_x_v2, model_semitruecroprev_v2 = semi_supervised_learning(celltype, y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_unlabeled_seq, current_seed, BATCH_SIZE, INIT_LR,
                                                                                     early_stop_thresh, EPOCHS, num_kernels, dropout_rate, iteration, combine='Semi+truecroprev_v2')
            #print(f"After Semi+truecroprev_v2 y_semiupdated:{y_semiupdated}")
            #print(f"After Semi+truecroprev_v2 y_semiupdated shape:{y_semiupdated.shape}")
            elapsed_time = time.time() - start_time
            print(f'Elapsed time for Semi+truecroprev_v2 in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
########################
            for combine in combinations:
                if combine in ["true", "true+rev", "true+crop", "true+rev+crop"]:
                    start_time = time.time()
                    if iteration == 0:
                        print('######################### Naive method:', combine)
                    
                    if combine in ["true", "true+rev", "true+crop"]:
                        y_rev_naive = y_rev[0:x_seq.shape[0]]
                        x_seq_rev_naive = x_seq_rev[0:x_seq.shape[0]]
                    elif combine == "true+rev+crop":
                        y_rev_naive = y_rev
                        x_seq_rev_naive = x_seq_rev
                        
                    trainData_naive, valData_naive = genTrainData(y, x_seq, y_rev_naive, x_seq_rev_naive, y_crop, x_seq_crop, current_seed, combine=combine, verbose=(iteration == 0), train_fraction=0.75)
                    
                    if iteration == 0:
                        print(f'Before naive method {combine}, sample size: {x_seq.shape[0]}')
                        print(f'After naive method {combine}, sample size: {trainData_naive.tensors[0].shape[0]+valData_naive.tensors[0].shape[0]}')
                        pass
                    
                    model_savename = combine + celltype + '.pth'
                    model_naive = trainModel(trainData_naive, valData_naive, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_naive, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                    elapsed_time = time.time() - start_time
                    print(f'Elapsed time for {combine} in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
                
                elif combine =='VAE':
                    start_time = time.time()
                    
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=300, batch_size=1024, latent_dim=64, lr=2e-4)
                    #avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                    
                    if iteration < 1:
                        avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)

                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
#                    min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
#                    print(f'min_avg_combined_loss: {min_avg_combined_loss}')
#                    print(f'best_recon_loss: {best_recon_loss}')
#                    print(f'best_kl_loss: {best_kl_loss}')
#                    print(f'best_trimer_diff_loss: {best_trimer_diff_loss}')
            
                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                    elapsed_time = time.time() - start_time
                    print(f'Elapsed time for {combine} in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
                    
                elif combine =='VAE0.5':
                    start_time = time.time()
                    
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=300, batch_size=1024, latent_dim=64, lr=2e-4)
                    avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7*0.5, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                    
                    #if iteration < 15:

                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
#                    min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
#                    print(f'min_avg_combined_loss: {min_avg_combined_loss}')
#                    print(f'best_recon_loss: {best_recon_loss}')
#                    print(f'best_kl_loss: {best_kl_loss}')
#                    print(f'best_trimer_diff_loss: {best_trimer_diff_loss}')
            
                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                    elapsed_time = time.time() - start_time
                    print(f'Elapsed time for {combine} in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
                
                elif combine =='VAE0.1':
                    start_time = time.time()
                    
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=300, batch_size=1024, latent_dim=64, lr=2e-4)
                    avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7*0.1, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                    
                    #if iteration < 15:

                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
#                    min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
#                    print(f'min_avg_combined_loss: {min_avg_combined_loss}')
#                    print(f'best_recon_loss: {best_recon_loss}')
#                    print(f'best_kl_loss: {best_kl_loss}')
#                    print(f'best_trimer_diff_loss: {best_trimer_diff_loss}')
            
                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                    elapsed_time = time.time() - start_time
                    print(f'Elapsed time for {combine} in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
                
                elif combine =='VAE_notrimer':
                    start_time = time.time()
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE_notrimer Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=300, batch_size=1024, latent_dim=64, lr=2e-4)
                    avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=0, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                    
                    #if iteration < 15:
                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)

                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                    elapsed_time = time.time() - start_time
                    print(f'Elapsed time for {combine} in iteration {iteration}: {elapsed_time / 60:.2f} minutes')
                
                elif combine == 'Semi':
                    final_sample_sizes_semi.append(newly_combined_x.shape[0])
                    final_sample_size_increases_semi.append(newly_combined_x.shape[0] - original_truelabeled_samplesize)
                    model_savename = combine + celltype + '.pth'
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_semi, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine == 'Semi+truecroprev_v1':
                    y_semiupdated = np.concatenate((y_semiupdated, y_rev, y_crop), axis=0)
                    x_seq_semiupdated = np.concatenate((x_seq_semiupdated, x_seq_rev, x_seq_crop), axis=0)
                    final_sample_sizes_semitruecroprev_v1.append(x_seq_semiupdated.shape[0])
                    final_sample_size_increases_semitruecroprev_v1.append(x_seq_semiupdated.shape[0] - original_truelabeled_samplesize - x_seq_rev.shape[0]- x_seq_crop.shape[0])
                    
                    train_features_np, val_features_np, train_labels_np, val_labels_np = train_test_split(x_seq_semiupdated, y_semiupdated, test_size=0.25, random_state=current_seed, stratify=y_semiupdated)
                    train_features = torch.from_numpy(train_features_np)
                    train_labels = torch.from_numpy(train_labels_np).long()
                    val_features = torch.from_numpy(val_features_np)
                    val_labels = torch.from_numpy(val_labels_np).long()
                    trainData_semitruecroprev_v1 = TensorDataset(train_features, train_labels)
                    valData_semitruecroprev_v1 = TensorDataset(val_features, val_labels)
                    model_savename = combine + celltype + '.pth'
                    model_semitruecroprev_v1 = trainModel(trainData_semitruecroprev_v1, valData_semitruecroprev_v1, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_semitruecroprev_v1, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine == 'Semi+truecroprev_v2':
                    final_sample_sizes_semitruecroprev_v2.append(newly_combined_x_v2.shape[0])
                    final_sample_size_increases_semitruecroprev_v2.append(newly_combined_x_v2.shape[0] - original_truelabeled_samplesize - x_seq_rev.shape[0]- x_seq_crop.shape[0])
                    model_savename = combine + celltype + '.pth'
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_semitruecroprev_v2, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine == 'Random_forest':
                    auc_test = auc_baseline
                    auprc_test = auprc_baseline
                
                #print(f'preds: {preds}')
                
                results[celltype][fraction][combine]['Accuracy'].append(acc_test)
                results[celltype][fraction][combine]['AUC'].append(auc_test)
                results[celltype][fraction][combine]['AUPRC'].append(auprc_test)
                results[celltype][fraction][combine]['f1'].append(f1_test)
                results[celltype][fraction][combine]['precision'].append(precision_test)
                results[celltype][fraction][combine]['recall'].append(recall_test)
                results[celltype][fraction][combine]['R'].append(R_test)
                results[celltype][fraction][combine]['predsProb'].append(np.exp(predsProb))
                results[celltype][fraction][combine]['preds'].append(preds)
                results[celltype][fraction][combine]['y_test'].append(y_test)
        #print(results)
    
    #print(f'final_sample_sizes_semi: {final_sample_sizes_semi}')
    #print(f'final_sample_size_increases_semi: {final_sample_size_increases_semi}')
    average_final_sample_sizes_semi = calculate_average_sample_sizes_interlaced(final_sample_sizes_semi, NUM_ITERATIONS, fractions)
    average_final_sample_size_increases_semi = calculate_average_sample_sizes_interlaced(final_sample_size_increases_semi, NUM_ITERATIONS, fractions)
    print(f'average_final_sample_sizes_semi: {average_final_sample_sizes_semi}')
    print(f'average_final_sample_size_increases_semi: {average_final_sample_size_increases_semi}')
    
    average_final_sample_sizes_semitruecroprev_v1 = calculate_average_sample_sizes_interlaced(final_sample_sizes_semitruecroprev_v1, NUM_ITERATIONS, fractions)
    average_final_sample_size_increases_semitruecroprev_v1 = calculate_average_sample_sizes_interlaced(final_sample_size_increases_semitruecroprev_v1, NUM_ITERATIONS, fractions)
    print(f'average_final_sample_sizes_semitruecroprev_v1: {average_final_sample_sizes_semitruecroprev_v1}')
    print(f'average_final_sample_size_increases_semitruecroprev_v1: {average_final_sample_size_increases_semitruecroprev_v1}')
    
    average_final_sample_sizes_semitruecroprev_v2 = calculate_average_sample_sizes_interlaced(final_sample_sizes_semitruecroprev_v2, NUM_ITERATIONS, fractions)
    average_final_sample_size_increases_semitruecroprev_v2 = calculate_average_sample_sizes_interlaced(final_sample_size_increases_semitruecroprev_v2, NUM_ITERATIONS, fractions)
    print(f'average_final_sample_sizes_semitruecroprev_v2: {average_final_sample_sizes_semitruecroprev_v2}')
    print(f'average_final_sample_size_increases_semitruecroprev_v2: {average_final_sample_size_increases_semitruecroprev_v2}')
    
    testData_indices_df.to_csv(f'{output_dir}/{celltype}_testData_indices.csv', index=False)
    
    print('#######################################################################################')
    return results


# In[ ]:


def process_fractions_allmethods_nounlabeled(celltype, fractions, combinations, x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop,
                                             dropout_rate, num_kernels, BATCH_SIZE,INIT_LR, early_stop_thresh, EPOCHS, NUM_ITERATIONS, results, input_dir, output_dir, 
                                             data_folder, device, random_state):
    
    
    testData_indices_df = pd.DataFrame()

    for iteration in range(NUM_ITERATIONS):
        start_time = time.time()
        
        current_seed = random_state + iteration
        np.random.seed(int(current_seed))
        
        x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
            x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, test_size=0.2, seed=current_seed, verbose=1)
        
        iteration_df = pd.DataFrame(testData_indices.reshape(-1, 1), columns=[f'Iteration_{iteration + 1}'])
        testData_indices_df = pd.concat([testData_indices_df, iteration_df], axis=1)
        
        for fraction in fractions:
            print('######################### Downsample fraction:', fraction, '###########################')

            y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
                x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, seed=current_seed, fraction=fraction, verbose=(iteration == 0))

            #Random Forest codes            
            x_pos_downsample_sequence = onehot_to_seq(x_pos_seq_downsample)
            x_neg_downsample_sequence = onehot_to_seq(x_neg_seq_downsample)
            x_test_sequence = onehot_to_seq(x_test_noswap)
            x_train_sequence = np.concatenate((x_pos_downsample_sequence, x_neg_downsample_sequence), axis=0)
            y_train = np.concatenate((y_pos_downsample, y_neg_downsample), axis=0)

            import itertools
            x_train = sequence_to_kmer_features(x_train_sequence)
            x_test = sequence_to_kmer_features(x_test_sequence)

            # Train Random Forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(x_train, y_train)

            y_pred_proba = clf.predict_proba(x_test)[:, 1]

            auc_baseline = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC: {auc_baseline}")

            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc_baseline = auc(recall, precision)
            print(f"AUPRC: {auprc_baseline}")

########################
            for combine in combinations:
                if combine in ["true", "true+rev", "true+crop", "true+rev+crop"]:
                    if iteration == 0:
                        print('######################### Naive method:', combine)
                    
                    if combine in ["true", "true+rev", "true+crop"]:
                        y_rev_naive = y_rev[0:x_seq.shape[0]]
                        x_seq_rev_naive = x_seq_rev[0:x_seq.shape[0]]
                    elif combine == "true+rev+crop":
                        y_rev_naive = y_rev
                        x_seq_rev_naive = x_seq_rev
                        
                    trainData_naive, valData_naive = genTrainData(y, x_seq, y_rev_naive, x_seq_rev_naive, y_crop, x_seq_crop, current_seed, combine=combine, verbose=(iteration == 0), train_fraction=0.75)
                    
                    if iteration == 0:
                        print(f'Before naive method {combine}, sample size: {x_seq.shape[0]}')
                        print(f'After naive method {combine}, sample size: {trainData_naive.tensors[0].shape[0]+valData_naive.tensors[0].shape[0]}')
                        pass
                    
                    model_savename = combine + celltype + '.pth'
                    model_naive = trainModel(trainData_naive, valData_naive, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_naive, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                
                elif combine =='VAE':
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=300, batch_size=1024, latent_dim=64, lr=2e-4)
                    #avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
                    
                    if iteration < 1:
                        avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                        
                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)

                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine =='VAE_notrimer':
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE_notrimer Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=300, batch_size=1024, latent_dim=64, lr=2e-4)
                    avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=0, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                    
                    #if iteration < 15:
                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
                    #min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)

                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                
                elif combine == 'Random_forest':
                    auc_test = auc_baseline
                    auprc_test = auprc_baseline
                
                #print(f'preds: {preds}')
                
                results[celltype][fraction][combine]['Accuracy'].append(acc_test)
                results[celltype][fraction][combine]['AUC'].append(auc_test)
                results[celltype][fraction][combine]['AUPRC'].append(auprc_test)
                results[celltype][fraction][combine]['f1'].append(f1_test)
                results[celltype][fraction][combine]['precision'].append(precision_test)
                results[celltype][fraction][combine]['recall'].append(recall_test)
                results[celltype][fraction][combine]['R'].append(R_test)
                results[celltype][fraction][combine]['predsProb'].append(np.exp(predsProb))
                results[celltype][fraction][combine]['preds'].append(preds)
                results[celltype][fraction][combine]['y_test'].append(y_test)
        #print(results)
        
        elapsed_time = time.time() - start_time
        print(f'Elapsed time for iteration {iteration}: {elapsed_time / 60:.2f} minutes')
    
    testData_indices_df.to_csv(f'{output_dir}/{celltype}_testData_indices.csv', index=False)
    
    print('#######################################################################################')
    return results


# In[ ]:


def process_fractions_allmethods_nounlabeled_old(celltype, fractions, combinations, x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop,
                                             dropout_rate, num_kernels, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, NUM_ITERATIONS, results, input_dir, output_dir, 
                                             data_folder, device, random_state):

    for iteration in range(NUM_ITERATIONS):
        current_seed = random_state + iteration
        np.random.seed(int(current_seed))
        
        start_time = time.time()
        
        x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
                x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, test_size=0.2, seed=current_seed, verbose=1) 

        for fraction in fractions:
            print('######################### Downsample fraction:', fraction, '###########################')
            y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
                x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, seed=current_seed, fraction=fraction, verbose=(iteration == 0))
            
            for combine in combinations:
                
                if combine in ["true", "true+rev", "true+crop", "true+rev+crop"]:
                    if iteration == 0:
                        print('######################### Naive method:', combine)
                    
                    if combine in ["true", "true+rev", "true+crop"]:
                        y_rev_naive = y_rev[0:x_seq.shape[0]]
                        x_seq_rev_naive = x_seq_rev[0:x_seq.shape[0]]
                    elif combine == "true+rev+crop":
                        y_rev_naive = y_rev
                        x_seq_rev_naive = x_seq_rev
                        
                    trainData, valData = genTrainData(y, x_seq, y_rev_naive, x_seq_rev_naive, y_crop, x_seq_crop, random_state, combine=combine, verbose=(iteration == 0), train_fraction=0.75)
                    
                    if iteration == 0:
                        print(f'Before naive method {combine}, sample size: {x_seq.shape[0]}')
                        print(f'After naive method {combine}, sample size: {trainData.tensors[0].shape[0]+valData.tensors[0].shape[0]}')
                        pass
                    
                    model_savename = combine + celltype + '.pth'
                    model_naive = trainModel(trainData, valData, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_naive, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                
                elif combine =='VAE':
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
                    min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
            
                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, random_state)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)                
                
                results[celltype][fraction][combine]['AUC'].append(auc_test)
                #results[celltype][fraction][combine]['R'].append(R_test)
        #print(results)
        
        elapsed_time = time.time() - start_time
        print(f'Elapsed time for iteration {iteration}: {elapsed_time / 60:.2f} minutes')
    print('#######################################################################################')
    return results


# In[ ]:


def testprocess_fractions_allmethods(celltype, fractions, combinations, x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, x_unlabeled_seq,
                                 dropout_rate, num_kernels, BATCH_SIZE,INIT_LR, early_stop_thresh, EPOCHS, NUM_ITERATIONS, results, input_dir, output_dir, 
                                 data_folder, device, random_state):
    
    final_sample_sizes_semi = []
    final_sample_sizes_semitruecroprev_v1 = []
    final_sample_sizes_semitruecroprev_v2 = []
    final_sample_size_increases_semi = []
    final_sample_size_increases_semitruecroprev_v1 = []
    final_sample_size_increases_semitruecroprev_v2 = []
    
    testData_indices_df = pd.DataFrame()

    for iteration in range(NUM_ITERATIONS):
        start_time = time.time()
        
        current_seed = random_state + iteration
        np.random.seed(int(current_seed))
        
        x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
            x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, test_size=0.2, seed=current_seed, verbose=1)
        
        iteration_df = pd.DataFrame(testData_indices.reshape(-1, 1), columns=[f'Iteration_{iteration + 1}'])
        testData_indices_df = pd.concat([testData_indices_df, iteration_df], axis=1)
        
        for fraction in fractions:
            print('######################### Downsample fraction:', fraction, '###########################')

            y, x_seq, y_rev, x_seq_rev, y_crop, x_seq_crop, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
                x_pos_seq_trainval, x_neg_seq_trainval, x_pos_seq_rev_trainval, x_neg_seq_rev_trainval, x_pos_seq_crop_trainval, x_neg_seq_crop_trainval, seed=current_seed, fraction=fraction, verbose=(iteration == 0))

########################
            for combine in combinations:
                if combine in ["true", "true+rev", "true+crop", "true+rev+crop"]:
                    if iteration == 0:
                        print('######################### Naive method:', combine)
                    
                    if combine in ["true", "true+rev", "true+crop"]:
                        y_rev_naive = y_rev[0:x_seq.shape[0]]
                        x_seq_rev_naive = x_seq_rev[0:x_seq.shape[0]]
                    elif combine == "true+rev+crop":
                        y_rev_naive = y_rev
                        x_seq_rev_naive = x_seq_rev
                        
                    trainData_naive, valData_naive = genTrainData(y, x_seq, y_rev_naive, x_seq_rev_naive, y_crop, x_seq_crop, current_seed, combine=combine, verbose=(iteration == 0), train_fraction=0.75)
                    
                    if iteration == 0:
                        print(f'Before naive method {combine}, sample size: {x_seq.shape[0]}')
                        print(f'After naive method {combine}, sample size: {trainData_naive.tensors[0].shape[0]+valData_naive.tensors[0].shape[0]}')
                        pass
                    
                    model_savename = combine + celltype + '.pth'
                    model_naive = trainModel(trainData_naive, valData_naive, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_naive, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                
                elif combine =='VAE':
                    if iteration == 0:
                        print('######################### VAE method:', combine)
                    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
                    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
                    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
                    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)
                    if iteration == 0:
                        #print(f'y_downsampletrue sahpe: {y_downsampletrue.shape}')
                        #print(f'x_seq_downsampletrue sahpe: {x_seq_downsampletrue.shape}')
                        print("Sample Size of True Data Used for VAE Seqs Generation:", x_seq_downsampletrue.shape[0])
                        #print("Sample Size of TestData for final testModel:", testData.tensors[0].shape[0])
                           
                    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
                    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

                    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.pos.fasta", output_dir=input_dir)
                    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{idata}.{celltype}.neg.fasta", output_dir=input_dir)
                    
                    avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
                    #if iteration < 15:

                    #this step includes torch.save(model.state_dict(), f'cvae.{celltype}.pth')
#                    min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss = train_model_for_celltype(idata, celltype, input_dir, output_dir, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4)
#                    print(f'min_avg_combined_loss: {min_avg_combined_loss}')
#                    print(f'best_recon_loss: {best_recon_loss}')
#                    print(f'best_kl_loss: {best_kl_loss}')
#                    print(f'best_trimer_diff_loss: {best_trimer_diff_loss}')
            
                    latent_dim = 64
                    model = cVAE(latent_dim).to(device)
                    lr = 2e-4
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(idata, celltype, input_dir, output_dir)
                    generate_and_save_sequences_for_celltype(idata, celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
                    seq_pos_file_vae='seq.vae.'+idata+'.'+celltype+'.pos.fasta'
                    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
                    seq_neg_file_vae='seq.vae.'+idata+'.'+celltype+'.neg.fasta'
                    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
                    if iteration == 0:
                        print('x_pos_seq_vae shape:', x_pos_seq_vae.shape)
                        print('x_neg_seq_vae shape:', x_neg_seq_vae.shape)

                    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
                    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
                    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
                    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
                    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
                    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

                    #print('x_seq_vae shape:', x_seq_vae.shape)

                    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                
                    if iteration == 0:
                        print('After VAE, the Total Sample Size:', trainData_vae.tensors[0].shape[0]+valData_vae.tensors[0].shape[0])

                    model_savename = combine + celltype + '.pth'
                    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_vae, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                
                elif combine == 'Semi':
                    final_sample_sizes_semi.append(newly_combined_x.shape[0])
                    final_sample_size_increases_semi.append(newly_combined_x.shape[0] - original_truelabeled_samplesize)
                    model_savename = combine + celltype + '.pth'
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_semi, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine == 'Semi+truecroprev_v1':
                    y_semiupdated = np.concatenate((y_semiupdated, y_rev, y_crop), axis=0)
                    x_seq_semiupdated = np.concatenate((x_seq_semiupdated, x_seq_rev, x_seq_crop), axis=0)
                    final_sample_sizes_semitruecroprev_v1.append(x_seq_semiupdated.shape[0])
                    final_sample_size_increases_semitruecroprev_v1.append(x_seq_semiupdated.shape[0] - original_truelabeled_samplesize - x_seq_rev.shape[0]- x_seq_crop.shape[0])
                    
                    train_features_np, val_features_np, train_labels_np, val_labels_np = train_test_split(x_seq_semiupdated, y_semiupdated, test_size=0.25, random_state=current_seed, stratify=y_semiupdated)
                    train_features = torch.from_numpy(train_features_np)
                    train_labels = torch.from_numpy(train_labels_np).long()
                    val_features = torch.from_numpy(val_features_np)
                    val_labels = torch.from_numpy(val_labels_np).long()
                    trainData_semitruecroprev_v1 = TensorDataset(train_features, train_labels)
                    valData_semitruecroprev_v1 = TensorDataset(val_features, val_labels)
                    model_savename = combine + celltype + '.pth'
                    model_semitruecroprev_v1 = trainModel(trainData_semitruecroprev_v1, valData_semitruecroprev_v1, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, verbose=0, num_kernels=num_kernels, dropout_rate=dropout_rate)
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_semitruecroprev_v1, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine == 'Semi+truecroprev_v2':
                    final_sample_sizes_semitruecroprev_v2.append(newly_combined_x_v2.shape[0])
                    final_sample_size_increases_semitruecroprev_v2.append(newly_combined_x_v2.shape[0] - original_truelabeled_samplesize - x_seq_rev.shape[0]- x_seq_crop.shape[0])
                    model_savename = combine + celltype + '.pth'
                    acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = testModel(model_semitruecroprev_v2, model_savename, testData, BATCH_SIZE, verbose=0, predonly=0)
                    
                elif combine == 'Random_forest':
                    auc_test = auc_baseline
                    auprc_test = auprc_baseline
                
                #print(f'preds: {preds}')
                
                results[celltype][fraction][combine]['Accuracy'].append(acc_test)
                results[celltype][fraction][combine]['AUC'].append(auc_test)
                results[celltype][fraction][combine]['AUPRC'].append(auprc_test)
                results[celltype][fraction][combine]['f1'].append(f1_test)
                results[celltype][fraction][combine]['precision'].append(precision_test)
                results[celltype][fraction][combine]['recall'].append(recall_test)
                results[celltype][fraction][combine]['R'].append(R_test)
                results[celltype][fraction][combine]['predsProb'].append(np.exp(predsProb))
                results[celltype][fraction][combine]['preds'].append(preds)
                results[celltype][fraction][combine]['y_test'].append(y_test)
        #print(results)
        
        elapsed_time = time.time() - start_time
        print(f'Elapsed time for iteration {iteration}: {elapsed_time / 60:.2f} minutes')
    
    #print(f'final_sample_sizes_semi: {final_sample_sizes_semi}')
    #print(f'final_sample_size_increases_semi: {final_sample_size_increases_semi}')
    average_final_sample_sizes_semi = calculate_average_sample_sizes_interlaced(final_sample_sizes_semi, NUM_ITERATIONS, fractions)
    average_final_sample_size_increases_semi = calculate_average_sample_sizes_interlaced(final_sample_size_increases_semi, NUM_ITERATIONS, fractions)
    print(f'average_final_sample_sizes_semi: {average_final_sample_sizes_semi}')
    print(f'average_final_sample_size_increases_semi: {average_final_sample_size_increases_semi}')
    
    average_final_sample_sizes_semitruecroprev_v1 = calculate_average_sample_sizes_interlaced(final_sample_sizes_semitruecroprev_v1, NUM_ITERATIONS, fractions)
    average_final_sample_size_increases_semitruecroprev_v1 = calculate_average_sample_sizes_interlaced(final_sample_size_increases_semitruecroprev_v1, NUM_ITERATIONS, fractions)
    print(f'average_final_sample_sizes_semitruecroprev_v1: {average_final_sample_sizes_semitruecroprev_v1}')
    print(f'average_final_sample_size_increases_semitruecroprev_v1: {average_final_sample_size_increases_semitruecroprev_v1}')
    
    average_final_sample_sizes_semitruecroprev_v2 = calculate_average_sample_sizes_interlaced(final_sample_sizes_semitruecroprev_v2, NUM_ITERATIONS, fractions)
    average_final_sample_size_increases_semitruecroprev_v2 = calculate_average_sample_sizes_interlaced(final_sample_size_increases_semitruecroprev_v2, NUM_ITERATIONS, fractions)
    print(f'average_final_sample_sizes_semitruecroprev_v2: {average_final_sample_sizes_semitruecroprev_v2}')
    print(f'average_final_sample_size_increases_semitruecroprev_v2: {average_final_sample_size_increases_semitruecroprev_v2}')
    
    testData_indices_df.to_csv(f'{output_dir}/{celltype}_testData_indices.csv', index=False)
    
    print('#######################################################################################')
    return results

