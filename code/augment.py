from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import torch
import warnings


parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument("celltype", type=str, help="Cell type to process")
parser.add_argument("--lib_path", type=str, required=True, help="Path to lib.py")
parser.add_argument("--model_path", type=str, required=True, help="Path to model.py")
parser.add_argument("--data_folder", type=str, required=True, help="Path to Data folder")
parser.add_argument("--model_dir", type=str, required=True, help="Path to MpraVAE model folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
parser.add_argument("--multiplier", type=int, default=5, help="Multiplier for generating sequences (default: 5)")
args = parser.parse_args()

celltype = args.celltype
lib_path = args.lib_path
model_path = args.model_path
data_folder = Path(args.data_folder)
model_dir = args.model_dir
output_dir = args.output_dir
multiplier = args.multiplier

with open(lib_path) as f:
    exec(f.read())
with open(model_path) as f:
    exec(f.read())

torch.cuda.is_available()
torch.cuda.device_count()

print(torch.__version__)
print(torch.version.cuda)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

print('####################################')
print('########### Celltype:', celltype, '##########')
print('####################################')

latent_dim = 64
model = cVAE(latent_dim).to(device)
lr = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(celltype, input_dir, output_dir)
generate_and_save_sequences_for_celltype(celltype, input_dir= model_dir, output_dir= output_dir, pos_trimer_freq=pos_trimer_freq, neg_trimer_freq=neg_trimer_freq, multiplier=multiplier, verbose=0)
                

