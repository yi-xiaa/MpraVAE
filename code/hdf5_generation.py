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
parser.add_argument("--data_folder", type=str, required=True, help="Path to Data folder")
args = parser.parse_args()

celltype = args.celltype
lib_path = args.lib_path
data_folder = Path(args.data_folder)

with open(lib_path) as f:
    exec(f.read())

torch.cuda.is_available()
torch.cuda.device_count()

print(torch.__version__)
print(torch.version.cuda)

warnings.filterwarnings("ignore")

random_state=1234
seed=1234

print('####################################')
print('########### Celltype:', celltype, '##########')
print('####################################')

x_pos_seq, x_neg_seq, x_test_pos_seq, x_test_neg_seq= readData(celltype, data_folder)







