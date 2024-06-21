#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn


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

#class Decoder(nn.Module):
#    def __init__(self, latent_dim):
#        super(Decoder, self).__init__()
#        
#        self.fc = nn.Linear(latent_dim + 1, 128*250)
#        self.bn_fc = nn.BatchNorm1d(128*250)  # BatchNorm layer after fc
#        
#        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
#        self.bn_deconv1 = nn.BatchNorm1d(64)  # BatchNorm layer after deconv1
#        
#        self.deconv2 = nn.ConvTranspose1d(64, 4, kernel_size=4, stride=2, padding=1)
#        # No BatchNorm after the final layer since we apply softmax activation
#
#    def forward(self, z, condition):
#        z = torch.cat([z, condition], dim=1)
#        x = F.relu(self.bn_fc(self.fc(z)))
#        x = x.view(x.size(0), 128, 250)
#        x = F.relu(self.bn_deconv1(self.deconv1(x)))
#        x = F.softmax(self.deconv2(x), dim=1)  # Apply softmax over the channels dimension
#        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim + 1, 128*250)
        self.bn_fc = nn.BatchNorm1d(128*250)  # BatchNorm layer after fc
        
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn_deconv1 = nn.BatchNorm1d(64)  # BatchNorm layer after deconv1
        
        self.deconv2 = nn.ConvTranspose1d(64, 4, kernel_size=4, stride=2, padding=1, output_padding=1)
        # Added output_padding to the last deconv layer to ensure that after the transposed convolution operation, the size of the output tensor matches the size of the target tensor.

    def forward(self, z, condition):
        z = torch.cat([z, condition], dim=1)
        x = F.relu(self.bn_fc(self.fc(z)))
        x = x.view(x.size(0), 128, 250)
        x = F.relu(self.bn_deconv1(self.deconv1(x)))
        x = F.softmax(self.deconv2(x), dim=1)  # Apply softmax over the channels dimension
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




