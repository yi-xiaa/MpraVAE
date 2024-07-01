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
parser.add_argument("--train_path", type=str, required=True, help="Path to train.py")
parser.add_argument("--data_folder", type=str, required=True, help="Path to Data folder")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input data folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
args = parser.parse_args()

celltype = args.celltype
lib_path = args.lib_path
model_path = args.model_path
train_path = args.train_path
data_folder = Path(args.data_folder)
input_dir = args.input_dir
output_dir = args.output_dir

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

x_pos_seq, x_neg_seq, x_test_pos_seq, x_test_neg_seq = readEncodedData(celltype, data_folder)
    
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

save_to_fastafile(vae_x_pos_downsample, f"seq.vaedownsampletrue.{celltype}.pos.fasta", output_dir=input_dir)
save_to_fastafile(vae_x_neg_downsample, f"seq.vaedownsampletrue.{celltype}.neg.fasta", output_dir=input_dir)
                    
avg_combined_loss, avg_recon_loss, avg_kl_loss, avg_trimer_diff_loss = train_model_for_celltype(celltype, input_dir, output_dir, lambda1=1e7, lambda2=0.5, num_epochs=600, batch_size=1024, latent_dim=64, lr=2e-4)
                        






