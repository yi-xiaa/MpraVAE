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
from train import *

parser = argparse.ArgumentParser(description="Run analysis for a given cell type")
parser.add_argument('--input_files', type=str, required=True, help='Comma-separated paths to the input files')
parser.add_argument("--model_file", type=str, required=True, help="Path to CNN model file")
args = parser.parse_args()

input_files = args.input_files.split(',')
for input_file in input_files:
    input_file = Path(input_file)
    input_dir = input_file.parent
    
model_file = Path(args.model_file)


torch.cuda.is_available()
torch.cuda.device_count()

print(torch.__version__)
print(torch.version.cuda)

warnings.filterwarnings("ignore")

BATCH_SIZE = 64
INIT_LR = 1e-4
early_stop_thresh = 10
EPOCHS=50
seed=1234

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

latent_dim = 64
model = cVAE(latent_dim).to(device)

lr = 2e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dropout_rate = 0.1
num_kernels = (128, 256)

input_file = os.path.join(input_dir, 'sequences.h5')
x_pos_seq, x_neg_seq = read_h5_file(input_file)


x_pos_seq_trainval, x_neg_seq_trainval, testData, testData_indices, x_test_noswap, y_test=split_testdata(
    x_pos_seq, x_neg_seq, test_size=0, seed=seed, verbose=1)

y, x_seq, x_pos_seq_downsample, y_pos_downsample, x_neg_seq_downsample, y_neg_downsample = genData_downsample(
    x_pos_seq_trainval, x_neg_seq_trainval, seed=seed, fraction=1.0, verbose=0)

y_pos_downsample_vae = np.ones(x_pos_seq_downsample.shape[0])
y_neg_downsample_vae = np.zeros(x_neg_seq_downsample.shape[0])
y_downsampletrue = np.concatenate((y_pos_downsample_vae, y_neg_downsample_vae), axis=0)
x_seq_downsampletrue = np.concatenate((np.swapaxes(x_pos_seq_downsample, 2, 1), np.swapaxes(x_neg_seq_downsample, 2, 1)), axis=0)


input_file = os.path.join(input_dir, 'mpravae_synthetic_sequences.h5')
x_pos_seq_vae, x_neg_seq_vae = read_h5_file(input_file)


y_pos_vae=np.ones(x_pos_seq_vae.shape[0])
y_neg_vae=np.zeros(x_neg_seq_vae.shape[0])
y_vae=np.concatenate((y_pos_vae,y_neg_vae),axis=0)
x_seq_vae=np.concatenate((x_pos_seq_vae,x_neg_seq_vae),axis=0)
x_seq_vae=np.swapaxes(x_seq_vae,2,1)
x_seq_vae,y_vae=shuffleXY(x_seq_vae,y_vae)

trainData_vae, valData_vae = genTrainData_vae(y_downsampletrue, x_seq_downsampletrue, y_vae, x_seq_vae, seed)

model_vae = trainModel(trainData_vae, valData_vae, model_file, BATCH_SIZE, INIT_LR, early_stop_thresh, EPOCHS, num_kernels=num_kernels, dropout_rate=dropout_rate)


