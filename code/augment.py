from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import torch
import warnings

with open("/path/to/lib.py") as f:
    exec(f.read())
with open("/path/to/model.py") as f:
    exec(f.read())
with open("/path/to/train.py") as f:
    exec(f.read())

parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument("celltype", type=str, help="Cell type to process")
args = parser.parse_args()
celltype = args.celltype

data_folder = Path("/path/to/Data")
input_dir = '/path/to/input_data_folder'
output_dir = '/path/to/output_folder'
fasta_output_dir = '/path/to/fasta_output_folder'


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

lr = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('####################################')
print('########### Celltype:', celltype, '##########')
print('####################################')
dropout_rate = 0.1
num_kernels = (128, 256)

x_pos_seq, x_neg_seq= readData(celltype)
    
MpraVAE(celltype, x_pos_seq, x_neg_seq, dropout_rate, num_kernels, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, input_dir, output_dir, 
                                           data_folder, device, random_state)
    







