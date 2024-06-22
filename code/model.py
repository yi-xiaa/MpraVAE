#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


# In[2]:


class MyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
        
    def __getitem__(self, index):
        seq = self.seqs[index]
        return seq
    
    def __len__(self):
        return len(self.seqs)


# In[3]:


class MyDataset2(Dataset):
    def __init__(self, seqs, reads):
        self.seqs = seqs
        self.reads = reads
        
    def __getitem__(self, index):
        seq = self.seqs[index]
        read = self.reads[index]
        return seq, read
    
    def __len__(self):
        return len(self.seqs)


# In[4]:
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
    


# In[5]:


class DeepSEAlight_single(nn.Module):
    def __init__(self, ):
        super(DeepSEAlight_single, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop = nn.Dropout(p=0.1)
        self.Linear1 = nn.Linear(53*256, 256)
        self.Linear2 = nn.Linear(256, 128)
        self.Linear3 = nn.Linear(128, 2)
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
        
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        x = x.view(-1, 53*256)
                
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        x = self.Linear3(x)
        
        x = self.logSoftmax(x)
        return x


# In[ ]:

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv1d(5, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)  # BatchNorm layer after conv1
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)  # BatchNorm layer after conv2
        
        self.fc_mu = nn.Linear(128*250, latent_dim)
        self.fc_log_var = nn.Linear(128*250, latent_dim)

    def forward(self, x, condition):
        condition = condition.view(condition.size(0), 1, 1).expand(-1, -1, x.size(2))
        x = torch.cat([x, condition], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim + 1, 128*250)
        self.bn_fc = nn.BatchNorm1d(128*250)
        
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn_deconv1 = nn.BatchNorm1d(64) 
        
        self.deconv2 = nn.ConvTranspose1d(64, 4, kernel_size=4, stride=2, padding=1, output_padding=1)

    def forward(self, z, condition):
        z = torch.cat([z, condition], dim=1)
        x = F.relu(self.bn_fc(self.fc(z)))
        x = x.view(x.size(0), 128, 250)
        x = F.relu(self.bn_deconv1(self.deconv1(x)))
        x = F.softmax(self.deconv2(x), dim=1)
        return x


class cVAE(nn.Module):
    def __init__(self, latent_dim):
        super(cVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, condition):
        mu, log_var = self.encoder(x, condition)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, condition)
        return recon_x, mu, log_var


# In[ ]:








