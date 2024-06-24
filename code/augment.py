from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import torch
import warnings

#with open("/path/to/lib.py") as f:
#    exec(f.read())
#with open("/path/to/model.py") as f:
#    exec(f.read())
#with open("/path/to/train.py") as f:
#    exec(f.read())

#parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
#parser.add_argument("celltype", type=str, help="Cell type to process")
#args = parser.parse_args()
#celltype = args.celltype

#data_folder = Path("/path/to/Data")
#input_dir = '/path/to/input_data_folder'
#output_dir = '/path/to/output_folder'
#fasta_output_dir = '/path/to/fasta_output_folder'


parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument("celltype", type=str, help="Cell type to process")
parser.add_argument("--lib_path", type=str, required=True, help="Path to lib.py")
parser.add_argument("--model_path", type=str, required=True, help="Path to model.py")
parser.add_argument("--train_path", type=str, required=True, help="Path to train.py")
parser.add_argument("--data_folder", type=str, required=True, help="Path to Data folder")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input data folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
parser.add_argument("--fasta_output_dir", type=str, required=True, help="Path to fasta output folder")
args = parser.parse_args()

celltype = args.celltype
lib_path = args.lib_path
model_path = args.model_path
train_path = args.train_path
data_folder = Path(args.data_folder)
input_dir = args.input_dir
output_dir = args.output_dir
fasta_output_dir = args.fasta_output_dir

with open(lib_path) as f:
    exec(f.read())
with open(model_path) as f:
    exec(f.read())
with open(train_path) as f:
    exec(f.read())







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
    







