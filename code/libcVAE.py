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


# In[1]:


def read_fasta(file_name):
    sequences = []
    cnt = 0
    for record in SeqIO.parse(file_name, "fasta"):
        cnt +=1
        seq = str(record.seq)[:]
        if all(base in 'ATGC' for base in seq):  # Check if all characters are valid
            sequences.append(seq)
            
        if cnt == 50000:
            break
    return sequences


# In[ ]:


def indices_to_sequence(indices_data):
    # Map indices to characters
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
    model.train()  # Set the model to training mode
    train_loss = 0

    for batch_idx, (data, conditions) in enumerate(train_loader):
        data, conditions = data.to(device), conditions.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, log_var = model(data, conditions)
        
        # Calculate loss
        loss = vae_loss(recon_batch, data, mu, log_var)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(train_loader.dataset)


# In[2]:


import matplotlib.pyplot as plt

def plot_trimer_frequencies(original_freq, generated_freq, title="", save_filename=None):
    trimers = sorted(original_freq.keys())  # Get a sorted list of trimers
    
    original_values = [original_freq[trimer] for trimer in trimers]
    generated_values = [generated_freq.get(trimer, 0) for trimer in trimers]  # Use 0 as a default value if trimer not in generated_freq
    
    bar_width = 0.35
    r1 = range(len(trimers))
    r2 = [x + bar_width for x in r1]

    plt.figure(figsize=(15,7))

    plt.bar(r1, original_values, width=bar_width, color='blue', label='Original')
    plt.bar(r2, generated_values, width=bar_width, color='red', label='Generated')

    plt.xlabel('Trimer', fontweight='bold')
    
    nth_label = 2  # Show every 2nd label
    plt.xticks([r + bar_width for r in range(len(trimers)) if r % nth_label == 0], 
              [trimers[i] for i in range(len(trimers)) if i % nth_label == 0], 
              rotation=45, fontsize=10)
    
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(title)
    plt.legend()
    
    if save_filename:  # Save the plot if a filename is provided
        plt.savefig(save_filename)
    else:
        plt.show()


# In[ ]:





# # Added functions

# In[ ]:


def vae_loss(recon_x, x, mu, log_var, kl_weight=1.0):
    # Categorical cross-entropy loss
    # Note: recon_x should contain raw scores (logits) for each class
    #       x should contain the class indices
    
#    recon_max1 = torch.argmax(recon_x,dim=1)
#    seq_batch1 = recon_max1.cpu().numpy()
#    seq_batch1
#    recon_max2 = torch.argmax(x,dim=1)
#    seq_batch2 = recon_max2.cpu().numpy()
#    recon_loss = sum(np.sum(seq_batch2!=seq_batch1,axis=1))*2
    
    recon_loss = F.cross_entropy(recon_x, x, reduction='sum')
#    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss, kl_weight * kl_loss


def trimer_frequency_old(seq_batch):
    trimer_count = defaultdict(int)
    total_trimers = 0

    for seq in seq_batch:
        for i in range(len(seq) - 2):
            trimer = seq[i:i+3]
            trimer_count[trimer] += 1
            total_trimers += 1

    for key in trimer_count:
        trimer_count[key] /= total_trimers

    return trimer_count

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
    N = mu.shape[0] # get the minibatch size
#    print(f'mu.shape[0]: {mu.shape[0]}')
    
    recon_loss, kl_loss = vae_loss(recon_batch, data, mu, log_var)
    recon_loss = 4000*recon_loss / (4 * 1001)

    if lambda1 == 0:
        trimer_diff_loss = 0
    else:
        trimer_freq_gen = trimer_frequency(recon_batch)
        trimer_freq_orig = trimer_frequency(data)
        trimer_diff_loss = sum(abs(trimer_freq_gen[trimer] - trimer_freq_orig.get(trimer, 0)) for trimer in trimer_freq_gen) / 64
    
    combined_loss = recon_loss + lambda1 * trimer_diff_loss + lambda2 * kl_loss
#    print(f'trimer_diff_loss from loss_with_trimer_difference(): {trimer_diff_loss}')
    
#    for trimer in trimer_freq_gen:
#        print(f'trimer in trimer_freq_gen: {trimer}, trimer_freq_gen[trimer]: {trimer_freq_gen[trimer]}, trimer_freq_orig.get(trimer, 0):{trimer_freq_orig.get(trimer, 0)}' ) 
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
#        print(f'recon_batch: {recon_batch}')
        combined_loss, recon_loss, trimer_diff_loss, kl_loss = loss_with_trimer_difference(recon_batch, data, mu, log_var, lambda1, lambda2)
        combined_loss.backward()
        optimizer.step()

        total_combined_loss += combined_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_trimer_diff_loss += trimer_diff_loss

#    print(f'len(train_loader.dataset):{len(train_loader.dataset)}')
#    avg_combined_loss = total_combined_loss / len(train_loader.dataset)
#    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
#    avg_kl_loss = total_kl_loss / len(train_loader.dataset)
#    avg_trimer_diff_loss = total_trimer_diff_loss / len(train_loader.dataset)
#    return avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss
    
    return combined_loss.item(), recon_loss.item(), kl_loss.item(), trimer_diff_loss


def process_sequences(idata, celltype, sequence_type, input_dir, output_dir):
    file_path = os.path.join(input_dir, f'seq.vaedownsampletrue.{idata}.{celltype}.{sequence_type}.fasta')
    
    sequences = read_fasta(file_path)
    encoded_seqs = [one_hot_encode(seq) for seq in sequences]

    np.save(os.path.join(output_dir, f'{idata}_{celltype}_{sequence_type}_encoded_data.npy'), encoded_seqs)

    trimer_freq = get_trimer_frequencies(sequences)
#    print(f"--- {sequence_type.upper()} SEQUENCES ---")
#    for trimer, freq in trimer_freq.items():
#        print(f'{trimer}: {freq}')
#    print("\n")

    with open(os.path.join(output_dir, f'{idata}_{celltype}_{sequence_type}_trimer_freq.pkl'), 'wb') as file:
        pickle.dump(trimer_freq, file)

def train_model_for_celltype(idata, celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=1000, batch_size=1024, latent_dim=64, lr=2e-4):

    for seq_type in ['neg', 'pos']:
        process_sequences(idata, celltype, seq_type, input_dir, output_dir)
        
    pos_encoded_data = np.load(f'{idata}_{celltype}_pos_encoded_data.npy', allow_pickle=True)
    with open(f'{idata}_{celltype}_pos_trimer_freq.pkl', 'rb') as file:
        pos_trimer_freq = pickle.load(file)
    
    neg_encoded_data = np.load(f'{idata}_{celltype}_neg_encoded_data.npy', allow_pickle=True)
    with open(f'{idata}_{celltype}_neg_trimer_freq.pkl', 'rb') as file:
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
#    print(f'tensor_data: {tensor_data}')
#    print(f'tensor_labels: {tensor_labels}')
    data_loader = DataLoader(TensorDataset(tensor_data, tensor_labels), batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#    min_avg_combined_loss = float('inf')
#    best_recon_loss = 0
#    best_kl_loss = 0
#    best_trimer_diff_loss = 0
    
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
        
#        if avg_kl_loss < highest_kl_loss:
#            highest_kl_loss = avg_kl_loss
#            epochs_since_klincrease = 0
#        else:
#            epochs_since_klincrease += 1
#        
#        if epochs_since_klincrease >= 100:
#            print(f"Stopping early at epoch {epoch} due to no increase in KL loss.")
#            break

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
    
#    return min_avg_combined_loss, best_recon_loss, best_kl_loss, best_trimer_diff_loss


# In[2]:


def save_to_fasta(sequence_list, filename, output_dir, header_prefix="seq"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        for idx, seq in enumerate(sequence_list):
            f.write(f">{header_prefix}{idx}\n")
            f.write(seq + "\n")

def process_sequences_for_celltype(idata, celltype, input_dir, output_dir):
    def read_and_process(sequence_type):
        # Construct full path to the input file
        file_path = os.path.join(input_dir, f'seq.vaedownsampletrue.{idata}.{celltype}.{sequence_type}.fasta')
        
        sequences = read_fasta(file_path)
        encoded_seqs = [one_hot_encode(seq) for seq in sequences]
        
        np.save(os.path.join(output_dir, f'{idata}_{celltype}_{sequence_type}_encoded_data.npy'), encoded_seqs)

        trimer_freq = get_trimer_frequencies(sequences)
#        print(f"--- {sequence_type.upper()} Original SEQUENCES ---")
#        for trimer, freq in trimer_freq.items():
#            print(f'{trimer}: {freq}')
#        print("\n")
        
        with open(os.path.join(output_dir, f'{idata}_{celltype}_{sequence_type}_trimer_freq.pkl'), 'wb') as file:
            pickle.dump(trimer_freq, file)
        
        return trimer_freq

    pos_trimer_freq = read_and_process('pos')
    neg_trimer_freq = read_and_process('neg')
    
    return pos_trimer_freq, neg_trimer_freq


def generate_and_save_sequences_for_celltype(idata, celltype, input_dir, output_dir, pos_trimer_freq, neg_trimer_freq, verbose=0):
    model_path = os.path.join(input_dir, f'test_cvae.{celltype}.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    seq_pos_file='seq.vaedownsampletrue.'+idata+'.'+celltype+'.pos.fasta'
    x_pos_seq=onehot(data_folder/seq_pos_file)
    seq_neg_file='seq.vaedownsampletrue.'+idata+'.'+celltype+'.neg.fasta'
    x_neg_seq=onehot(data_folder/seq_neg_file)
    
    #print('x_pos_seq shape in generate_and_save_sequences_for_celltype():', x_pos_seq.shape)
    #print('x_neg_seq shape in generate_and_save_sequences_for_celltype():', x_neg_seq.shape)
    
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

        
    #print(f'pos_generated_samples: {pos_generated_samples}')    
    _, pos_max_indices = torch.max(pos_generated_samples, dim=1)
    _, neg_max_indices = torch.max(neg_generated_samples, dim=1)

    pos_generated_sequences = indices_to_sequence(pos_max_indices)
    pos_trimer_freq_generated = get_trimer_frequencies(pos_generated_sequences)
#    print(f"--- Pos Generated SEQUENCES ---")
#    for trimer, freq in pos_trimer_freq_generated.items():
#        print(f'{trimer}: {freq}')
#    print("\n")

    neg_generated_sequences = indices_to_sequence(neg_max_indices)
    neg_trimer_freq_generated = get_trimer_frequencies(neg_generated_sequences)
#    print(f"--- Neg Generated SEQUENCES ---")
#    for trimer, freq in neg_trimer_freq_generated.items():
#        print(f'{trimer}: {freq}')
#    print("\n")

    #plot_trimer_frequencies(pos_trimer_freq, pos_trimer_freq_generated, title=f"Positive Trimer Frequencies for {celltype}", save_filename=f"pos_trimer_{celltype}.png")
    #plot_trimer_frequencies(neg_trimer_freq, neg_trimer_freq_generated, title=f"Negative Trimer Frequencies for {celltype}", save_filename=f"neg_trimer_{celltype}.png")

    pos_mse = sum(abs(pos_trimer_freq_generated[key] - pos_trimer_freq[key]) for key in pos_trimer_freq) / 64
    neg_mse = sum(abs(neg_trimer_freq_generated[key] - neg_trimer_freq[key]) for key in neg_trimer_freq) / 64
    avg_mse = (pos_mse + neg_mse) / 2
    
#    for key in pos_trimer_freq:
#        print(f'key in pos_trimer_freq: {key}, pos_trimer_freq[key]: {pos_trimer_freq[key]}, pos_trimer_freq_generated[key]: {pos_trimer_freq_generated[key]}')
#    print("\n")
#    for key in neg_trimer_freq:
#        print(f'key in neg_trimer_freq: {key}, neg_trimer_freq[key]: {neg_trimer_freq[key]}, neg_trimer_freq_generated[key]: {neg_trimer_freq_generated[key]}')

    if verbose==1:
        print('Before VAE, the pos and neg Sample Size: ',[x_pos_seq.shape[0],x_neg_seq.shape[0]])
        print('Before VAE, the Total Sample Size: ', x_pos_seq.shape[0] + x_neg_seq.shape[0])
        print(f'Average MSE for {celltype}:', avg_mse)
    
    save_to_fasta(pos_generated_sequences, f"seq.vae.{idata}.{celltype}.pos.fasta", output_dir=output_dir)
    save_to_fasta(neg_generated_sequences, f"seq.vae.{idata}.{celltype}.neg.fasta", output_dir=output_dir)
    
    
    
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




