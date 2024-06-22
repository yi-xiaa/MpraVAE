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
# celltypes = ["Jurkat"]
# diseases = ["autoimmune_disease"]

idata='data10'
fractions = [0.2, 0.5, 1.0]
fractions_json = ['0.2', '0.5', '1.0']

NUM_ITERATIONS = 1

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

print(celltype)
x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, x_unlabeled_seq = readData(idata, celltype)

combinations = ['VAE']

results = {celltype: {fraction: {combine: {'Accuracy': [],'AUC': [],'AUPRC': [],'f1': [],'precision': [],'recall': [],'R': [],'predsProb': [],'preds': [],'y_test': []} for combine in combinations} for fraction in fractions}}

print('####################################')
print('########### Celltype:', celltype, '##########')
print('####################################')
# Retrieve dropout_rate and num_kernels for the current celltype
dropout_rate = 0.1
num_kernels = (128, 256)

x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, x_unlabeled_seq = readData(idata, celltype)
    
results = process_fractions_allmethods(celltype, fractions, combinations, x_pos_seq, x_neg_seq, x_pos_seq_rev, x_neg_seq_rev, x_pos_seq_crop, x_neg_seq_crop, x_unlabeled_seq,
                                           dropout_rate, num_kernels, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, NUM_ITERATIONS, results, input_dir, output_dir, 
                                           data_folder, device, random_state)
    







