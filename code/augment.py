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

parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument("--model_file", type=str, required=True, help="Path to MpraVAE model folder")
parser.add_argument("--multiplier", type=int, default=5, help="Multiplier for generating sequences (default: 5)")
parser.add_argument("--input_file", type=str, required=True, help="Path to input file")
parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
args = parser.parse_args()

model_file = Path(args.model_file)
model_dir = model_file.parent
input_file = Path(args.input_file)
input_dir = input_file.parent
output_file = Path(args.output_file)
output_dir = output_file.parent

multiplier = args.multiplier


torch.cuda.is_available()
torch.cuda.device_count()

print(torch.__version__)
print(torch.version.cuda)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


latent_dim = 64
model = cVAE(latent_dim).to(device)
lr = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pos_trimer_freq, neg_trimer_freq = process_sequences_for_celltype(input_dir)
generate_and_save_sequences_for_celltype(model_dir, input_dir, output_dir, pos_trimer_freq, neg_trimer_freq, multiplier, verbose=0)
                


    


                

