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

parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument("celltype", type=str, help="Cell type to process")
parser.add_argument("--data_folder", type=str, required=True, help="Path to Data folder")
args = parser.parse_args()

celltype = args.celltype
data_folder = Path(args.data_folder)


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







