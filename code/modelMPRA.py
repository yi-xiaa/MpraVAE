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


#Lout=L( (Lin+2*padding-dilation*(kernelsize-1)-1)/stride)+1 )
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





# In[ ]:




