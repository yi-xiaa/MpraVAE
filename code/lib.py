#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Bio import SeqIO
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
torch.manual_seed(42)
import torch.nn as nn
torch.cuda.manual_seed_all(42)
from tqdm import tqdm
from torch.nn import functional as F
import torch.nn.functional as F
import warnings
import os
import math

# In[2]:


import numpy as np
from numpy import array
from random import sample,seed
import time
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import h5py
import seaborn as sns
from scipy.stats import wilcoxon,pearsonr
from re import search
import sys
from collections import Counter
import itertools
import statistics

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



# In[3]:
def onehot(fafile):
    x = []
    for seq_record in SeqIO.parse(fafile, "fasta"):
        seq_array = np.array(list(seq_record.seq))

        if len(seq_array) != 1001:
            print(f"Warning: Sequence length {len(seq_array)} is not 1001")
            continue

        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

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



# In[4]:
def readData(celltype):
    # true data
    seq_pos_file='seq.'+'.'+celltype+'.pos.fasta'
    x_pos_seq=onehot(data_folder/seq_pos_file)
    seq_neg_file='seq.'+'.'+celltype+'.neg.fasta'
    x_neg_seq=onehot(data_folder/seq_neg_file)

    print('true, pos and neg: ',[x_pos_seq.shape,x_neg_seq.shape])

    return x_pos_seq,x_neg_seq


# In[ ]:

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
    
    pos_trainval_indices = np.setdiff1d(np.arange(x_pos_seq.shape[0]), pos_indices)
    neg_trainval_indices = np.setdiff1d(np.arange(x_neg_seq.shape[0]), neg_indices)

    x_pos_seq_trainval = x_pos_seq[pos_trainval_indices]
    x_neg_seq_trainval = x_neg_seq[neg_trainval_indices]
    
    y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)
    x_test_noswap = np.concatenate((x_pos_seq_test, x_neg_seq_test), axis=0)
    x_test = np.swapaxes(x_test_noswap,2,1)
    testData_seqs = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test).long())
    
    testData_indices = np.concatenate((pos_indices, original_len_pos + neg_indices), axis=0)

    return x_pos_seq_trainval, x_neg_seq_trainval, testData_seqs, testData_indices, x_test_noswap, y_test


# In[5]:
def shuffleXY(x_seq,y):
    indices = np.arange(len(y))
    indices = np.random.permutation(indices)
    y=y[indices]
    x_seq=x_seq[indices]
    return x_seq,y

# In[6]:

def genData(x_pos_seq,x_neg_seq,seed=1234):
    
    y_pos=np.ones(x_pos_seq.shape[0])
    y_neg=np.zeros(x_neg_seq.shape[0])
    y=np.concatenate((y_pos,y_neg),axis=0)
    x_seq=np.concatenate((x_pos_seq,x_neg_seq),axis=0)
    x_seq=np.swapaxes(x_seq,2,1)  

    np.random.seed(int(seed))
    x_seq,y=shuffleXY(x_seq,y)
    
    return y, x_seq


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

    y = np.concatenate((y_pos_downsample, y_neg_downsample), axis=0)
    x_seq = np.concatenate((x_pos_seq_downsample, x_neg_seq_downsample), axis=0)
    x_seq = np.swapaxes(x_seq,2,1)
    
    return y, x_seq, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample

# In[ ]:
def genTrainData(y, x_seq, random_state, combine='true', verbose=0, train_fraction=0.75):
    
    # for true
    trainData_seq, valData_seq, ytrain, yval = train_test_split(x_seq, y, stratify=y, test_size= 1-train_fraction, random_state=random_state)
    
    if combine in ["true"]:
        trainData_seqs=trainData_seq
        valData_seqs=valData_seq
        ytrains=ytrain
        yvals=yval
        
    trainData = TensorDataset(torch.from_numpy(trainData_seqs), torch.from_numpy(ytrains).long())
    valData = TensorDataset(torch.from_numpy(valData_seqs), torch.from_numpy(yvals).long())

    return trainData, valData

# In[ ]:

def genTrainData_vae(y, x_seq, y_vae, x_seq_vae, random_state, verbose=0):
    
    # for true
    trainData_seq, valData_seq, ytrain, yval = train_test_split(x_seq, y, stratify=y, test_size=0.25, random_state=random_state)

    # for vae
    trainData_seq_vae, valData_seq_vae, ytrain_vae, yval_vae = train_test_split(x_seq_vae, y_vae, stratify=y_vae, test_size=0.25, random_state=random_state)
    
    trainData_seqs=np.concatenate([trainData_seq,trainData_seq_vae],axis=0)
    ytrains=np.concatenate([ytrain,ytrain_vae])
    valData_seqs=np.concatenate([valData_seq,valData_seq_vae],axis=0)
    yvals=np.concatenate([yval,yval_vae])      
    
    trainData_seqs,ytrains=shuffleXY(trainData_seqs,ytrains)
    valData_seqs,yvals=shuffleXY(valData_seqs,yvals)

    trainData_seqs = TensorDataset(torch.from_numpy(trainData_seqs), torch.from_numpy(ytrains).long())
    valData_seqs = TensorDataset(torch.from_numpy(valData_seqs), torch.from_numpy(yvals).long())
    
    return trainData_seqs, valData_seqs

# In[ ]:


def calculate_average_sample_sizes_interlaced(sample_sizes, num_iterations, fractions):
    fraction_averages = {}

    for i, fraction in enumerate(fractions):
        fraction_samples = sample_sizes[i::len(fractions)]
        average_size = sum(fraction_samples) / num_iterations
        fraction_averages[fraction] = average_size

    return fraction_averages

# In[ ]:
def MpraVAE(celltype,x_pos_seq, x_neg_seq, dropout_rate, num_kernels, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, input_dir, output_dir, 
                                             data_folder, device, random_state):

    x_pos_seq_trainval, x_neg_seq_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
            x_pos_seq, x_neg_seq, test_size=0, seed=current_seed, verbose=1)

    y, x_seq, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
                x_pos_seq_trainval, x_neg_seq_trainval, seed=current_seed, verbose=0)

    y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
    y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
    y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
    x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)

    vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
    vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

    save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{celltype}.pos.fasta", output_dir=input_dir)
    save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{celltype}.neg.fasta", output_dir=input_dir)
                    
    avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                        
    latent_dim = 64
    model = cVAE(latent_dim).to(device)
    lr = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(celltype, input_dir, output_dir)
    generate_and_save_sequences_for_celltype(celltype, input_dir= input_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, verbose=(iteration == 0))
                
    seq_pos_file_vae='seq.vae.'+'.'+celltype+'.pos.fasta'
    x_pos_seq_vae=onehot(data_folder/seq_pos_file_vae)
    seq_neg_file_vae='seq.vae.'+'.'+celltype+'.neg.fasta'
    x_neg_seq_vae=onehot(data_folder/seq_neg_file_vae)
                
    y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
    y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
    y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
    x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
    x_seq_vae=np.swapaxes(x_seq_vae,2,1)
    x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

    trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, current_seed)
                

    model_savename = 'VAE_' + celltype + '.pth'
    model_vae = trainModel(trainData_vae, valData_vae, model_savename, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, num_kernels=num_kernels, dropout_rate=dropout_rate)


# In[1]:

def read_fasta(file_name):
    sequences = []
    cnt = 0
    for record in SeqIO.parse(file_name, "fasta"):
        cnt +=1
        seq = str(record.seq)[:]
        if all(base in 'ATGC' for base in seq): 
            sequences.append(seq)
            
        if cnt == 50000:
            break
    return sequences


# In[ ]:


def indices_to_sequence(indices_data):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = [''.join([mapping[idx.item()] for idx in sequence]) for sequence in indices_data]
    return sequences


# In[ ]:


def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping[base] for base in sequence], dtype=int)


# In[ ]:


def get_trimer_frequencies(sequences):
    trimer_freq = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 2):
            trimer = seq[i:i+3]
            trimer_freq[trimer] += 1
            
    total_trimers = sum(trimer_freq.values())
    for trimer, freq in trimer_freq.items():
        trimer_freq[trimer] = freq / total_trimers
        
    return trimer_freq


# In[ ]:

def vae_loss(recon_x, x, mu, log_var):
    # BCE loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss #L1


# In[ ]:

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0

    for batch_idx, (data, conditions) in enumerate(train_loader):
        data, conditions = data.to(device), conditions.to(device)

        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data, conditions)
        
        loss = vae_loss(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(train_loader.dataset)

# In[ ]:


def vae_loss(recon_x, x, mu, log_var, kl_weight=1.0):
    recon_loss = F.cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss, kl_weight * kl_loss



def recon_to_sequence(recon):
    recon_max = torch.argmax(recon,dim=1)
    seq_batch = recon_max.cpu().numpy()
 
    nucleotide_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
 
    mapped_list = [
        ''.join([nucleotide_mapping[value] for value in row])
        for row in seq_batch
    ]
    return mapped_list
 
def trimer_frequency(seq_batch):
    mapped_list = recon_to_sequence(seq_batch)
    trimer_count = defaultdict(int)
    total_trimers = 0
 
    for seq in mapped_list:
        for i in range(len(seq) - 2):
            trimer = seq[i:i+3]
            trimer_count[trimer] += 1
            total_trimers += 1
 
    for key in trimer_count:
        trimer_count[key] /= total_trimers
 
    return trimer_count
 

def loss_with_trimer_difference(recon_batch, data, mu, log_var, lambda1, lambda2):
    N = mu.shape[0]

    recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var)
    recon_loss = 4000*recon_loss / (4 * 1001)

    if lambda1 == 0:
        trimer_diff_loss = 0
    else:
        trimer_freq_gen = trimer_frequency(recon_batch)
        trimer_freq_orig = trimer_frequency(data)
        trimer_diff_loss = sum(abs(trimer_freq_gen[trimer] - trimer_freq_orig.get(trimer, 0)) for trimer in trimer_freq_gen) / 64
    
    combined_loss = recon_loss + lambda1 * trimer_diff_loss + lambda2 * kl_loss

    combined_loss /= N
    recon_loss /= N
    trimer_diff_loss /= N
    kl_loss /= N
    return combined_loss, recon_loss, lambda1 * trimer_diff_loss, lambda2 * kl_loss



def train_epoch(epoch, model, train_loader, optimizer, device, lambda1=1e7, lambda2=0.5):
    model.train()
    total_combined_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_trimer_diff_loss = 0

    for batch_idx, (data, conditions) in enumerate(train_loader):
        data, conditions = data.to(device), conditions.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data, conditions)

        combined_loss, recon_loss, trimer_diff_loss, kl_loss = loss_with_trimer_difference(recon_batch, data, mu, log_var, lambda1, lambda2)
        combined_loss.backward()
        optimizer.step()

        total_combined_loss += combined_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_trimer_diff_loss += trimer_diff_loss
    
    return combined_loss.item(), recon_loss.item(), kl_loss.item(), trimer_diff_loss


def process_sequences(celltype, sequence_type, input_dir, output_dir):
    file_path = os.path.join(input_dir, f'seq.vaedownsampletrue.{celltype}.{sequence_type}.fasta')
    
    sequences = read_fasta(file_path)
    encoded_seqs = [one_hot_encode(seq) for seq in sequences]

    np.save(os.path.join(output_dir, f'{celltype}_{sequence_type}_encoded_data.npy'), encoded_seqs)

    trimer_freq = get_trimer_frequencies(sequences)

    with open(os.path.join(output_dir, f'{celltype}_{sequence_type}_trimer_freq.pkl'), 'wb') as file:
        pickle.dump(trimer_freq, file)

def train_model_for_celltype(celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4):

    for seq_type in ['neg', 'pos']:
        process_sequences(celltype, seq_type, input_dir, output_dir)
        
    pos_encoded_data = np.load(f'{celltype}_pos_encoded_data.npy', allow_pickle=True)
    with open(f'{celltype}_pos_trimer_freq.pkl', 'rb') as file:
        pos_trimer_freq = pickle.load(file)
    
    neg_encoded_data = np.load(f'{celltype}_neg_encoded_data.npy', allow_pickle=True)
    with open(f'{celltype}_neg_trimer_freq.pkl', 'rb') as file:
        neg_trimer_freq = pickle.load(file)

    neg_encoded_data = neg_encoded_data.transpose(0, 2, 1)
    pos_encoded_data = pos_encoded_data.transpose(0, 2, 1)

    pos_labels = np.ones((pos_encoded_data.shape[0], 1))
    neg_labels = np.zeros((neg_encoded_data.shape[0], 1))

    all_data = np.concatenate([pos_encoded_data, neg_encoded_data], axis=0)
    all_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    
    tensor_data = torch.tensor(all_data, dtype=torch.float32)
    tensor_labels = torch.tensor(all_labels, dtype=torch.float32)
    data_loader = DataLoader(TensorDataset(tensor_data, tensor_labels), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    avg_combined_loss_history = []
    avg_recon_loss_history = []
    avg_kl_loss_history = []
    avg_trimer_diff_loss_history = []
    
    highest_kl_loss = float('inf')
    epochs_since_klincrease = 0
    
    for epoch in range(num_epochs):
        avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_epoch(epoch, model, data_loader, optimizer, device, lambda1, lambda2)
        
        if epoch == 0 or epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch}: combined_loss history: {avg_combined_loss}')
            print(f'Epoch {epoch}: recon_loss history: {avg_recon_loss}')
            print(f'Epoch {epoch}: kl_loss history: {avg_kl_loss}')
            print(f'Epoch {epoch}: trimer_diff_loss history: {avg_trimer_diff_loss}')
        
        avg_combined_loss_history.append(avg_combined_loss)
        avg_recon_loss_history.append(avg_recon_loss)
        avg_kl_loss_history.append(avg_kl_loss)
        avg_trimer_diff_loss_history.append(avg_trimer_diff_loss)

    torch.save(model.state_dict(), f'test_cvae.{celltype}.pth')
    
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].plot(avg_combined_loss_history, label='combined_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('combined_loss')
    axs[0].set_title('Combined Loss')

    axs[1].plot(avg_recon_loss_history, label='recon_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('recon_loss')
    axs[1].set_title('Reconstruction Loss')

    axs[2].plot(avg_kl_loss_history, label='kl_loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('kl_loss')
    axs[2].set_title('KL Loss')

    axs[3].plot(avg_trimer_diff_loss_history, label='trimer_diff_loss')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('trimer_diff_loss')
    axs[3].set_title('Trimer Diff Loss')

    plt.tight_layout()
    fig.savefig(f"{celltype}_VAE_loss_lambda1{lambda1}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss


# In[2]:


def save_to_fasta(sequence_list, filename, output_dir, header_prefix="seq"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        for idx, seq in enumerate(sequence_list):
            f.write(f">{header_prefix}{idx}\n")
            f.write(seq + "\n")

def process_sequences_for_celltype(celltype, input_dir, output_dir):
    def read_and_process(sequence_type):
        file_path = os.path.join(input_dir, f'seq.vaedownsampletrue.{celltype}.{sequence_type}.fasta')
        
        sequences = read_fasta(file_path)
        encoded_seqs = [one_hot_encode(seq) for seq in sequences]
        
        np.save(os.path.join(output_dir, f'{celltype}_{sequence_type}_encoded_data.npy'), encoded_seqs)

        trimer_freq = get_trimer_frequencies(sequences)
        
        with open(os.path.join(output_dir, f'{celltype}_{sequence_type}_trimer_freq.pkl'), 'wb') as file:
            pickle.dump(trimer_freq, file)
        
        return trimer_freq

    pos_trimer_freq = read_and_process('pos')
    neg_trimer_freq = read_and_process('neg')
    
    return pos_trimer_freq, neg_trimer_freq


def generate_and_save_sequences_for_celltype(celltype, input_dir, output_dir, pos_trimer_freq, neg_trimer_freq, verbose=0):
    model_path = os.path.join(input_dir, f'test_cvae.{celltype}.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    seq_pos_file='seq.vaedownsampletrue.'+'.'+celltype+'.pos.fasta'
    x_pos_seq=onehot(data_folder/seq_pos_file)
    seq_neg_file='seq.vaedownsampletrue.'+'.'+celltype+'.neg.fasta'
    x_neg_seq=onehot(data_folder/seq_neg_file)
    
    with torch.no_grad():
        pos_generated_samples = []
        condition_pos = torch.ones(50, 1).to(device)
        for _ in range(math.ceil((5 * x_pos_seq.shape[0]) / 50)):
            z = torch.randn(50, latent_dim).to(device)
            samples = model.decoder(z, condition_pos)
            pos_generated_samples.append(samples)
        pos_generated_samples = torch.cat(pos_generated_samples, 0)[:5 * x_pos_seq.shape[0]]

        neg_generated_samples = []
        condition_neg = torch.zeros(50, 1).to(device)
        for _ in range(math.ceil((5 * x_neg_seq.shape[0]) / 50)):
            z = torch.randn(50, latent_dim).to(device)
            samples = model.decoder(z, condition_neg)
            neg_generated_samples.append(samples)
        neg_generated_samples = torch.cat(neg_generated_samples, 0)[:5 * x_neg_seq.shape[0]]


    _, pos_max_indices = torch.max(pos_generated_samples, dim=1)
    _, neg_max_indices = torch.max(neg_generated_samples, dim=1)

    pos_generated_sequences = indices_to_sequence(pos_max_indices)
    pos_trimer_freq_generated = get_trimer_frequencies(pos_generated_sequences)

    neg_generated_sequences = indices_to_sequence(neg_max_indices)
    neg_trimer_freq_generated = get_trimer_frequencies(neg_generated_sequences)


    pos_mse = sum(abs(pos_trimer_freq_generated[key] - pos_trimer_freq[key]) for key in pos_trimer_freq) / 64
    neg_mse = sum(abs(neg_trimer_freq_generated[key] - neg_trimer_freq[key]) for key in neg_trimer_freq) / 64
    avg_mse = (pos_mse + neg_mse) / 2

    
    save_to_fasta(pos_generated_sequences, f"seq.vae.{celltype}.pos.fasta", output_dir=output_dir)
    save_to_fasta(neg_generated_sequences, f"seq.vae.{celltype}.neg.fasta", output_dir=output_dir)
    
    
    
    trimer_keys = list(pos_trimer_freq.keys())
    data = {
        "Pos Original Seq": [pos_trimer_freq[key] for key in trimer_keys],
        "Pos Generated Seq": [pos_trimer_freq_generated.get(key, 0) for key in trimer_keys],
        "Pos Abs Diff": [abs(pos_trimer_freq[key] - pos_trimer_freq_generated.get(key, 0)) for key in trimer_keys],
        "Neg Original Seq": [neg_trimer_freq[key] for key in trimer_keys],
        "Neg Generated Seq": [neg_trimer_freq_generated.get(key, 0) for key in trimer_keys],
        "Neg Abs Diff": [abs(neg_trimer_freq[key] - neg_trimer_freq_generated.get(key, 0)) for key in trimer_keys]
    }
    trimer_df = pd.DataFrame(data, index=trimer_keys)

    csv_filename = os.path.join(output_dir, f'{celltype}_VAE_trimer_frequencies.csv')
    trimer_df.to_csv(csv_filename)


# In[ ]:




