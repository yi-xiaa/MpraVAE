from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import torch
import warnings
from lib import *
from model import *


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




parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument("--input_file", type=str, required=True, help="Path to input data file")
parser.add_argument("--model_file", type=str, required=True, help="Path to MpraVAE model file")
args = parser.parse_args()

input_file = Path(args.input_file)
input_dir = input_file.parent
model_file = args.model_file

torch.cuda.is_available()
torch.cuda.device_count()

print(torch.__version__)
print(torch.version.cuda)

warnings.filterwarnings("ignore")

BATCH_SIZE = 64
INIT_LR = 1e-4
early_stop_thresh = 10
EPOCHS=50
random_state=1234
seed=1234

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

latent_dim = 64
model = cVAE(latent_dim).to(device)
#print(model)

lr = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

x_pos_seq, x_neg_seq = read_h5_file(input_file)

x_pos_seq_trainval, x_neg_seq_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
    x_pos_seq, x_neg_seq, test_size=0, seed=seed, verbose=1)

y, x_seq, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
    x_pos_seq_trainval, x_neg_seq_trainval, seed=seed, fraction=1.0, verbose=0)

y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)

vae_x_pos_downsample = onehot_to_seq(x_pos_seq_downsample)
vae_x_neg_downsample = onehot_to_seq(x_neg_seq_downsample)

save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.pos.fasta", input_dir)
save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.neg.fasta", input_dir)
                    
avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_VAEmodel(input_dir, model_file, lambda1=1e7, lambda2=0.5, num_epochs=4, batch_size=1024, latent_dim=64, lr=2e-4)
                        






