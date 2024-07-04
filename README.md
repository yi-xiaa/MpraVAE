# MpraVAE
A deep generative model to augment genetic variants sample size, enhancing the prediction of regulatory variants in non-coding regions built on top of [Pytorch](https://pytorch.org/).

## Introduction
Predicting the functional impact of genetic variants in non-coding regions is challenging. Massively parallel reporter assays (MPRAs) can test thousands of variants for allele-specific regulatory activity, but typically only identify a few hundred labelled variants, limiting their use for genome-wide prediction. 

MpraVAE, a deep generative model, addresses this limitation by augmenting the training sample size of labelled variants. Benchmarking on multiple MPRA datasets shows that MpraVAE significantly improves the prediction of regulatory variants compared to conventional data augmentation methods and existing scoring techniques. Taking autoimmune diseases as one example, MpraVAE enabled genome-wide de novo prediction of regulatory variants, revealing their enrichment in enhancers, active histone marks, immune-related cell types, and key regulatory sites. These variants also facilitated the discovery of immune-related genes through integration with PCHi-C and DNase-seq data, highlighting MpraVAE's importance in genetic and gene discovery for complex traits.

![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/Figure_1.png)

## MpraVAE Webserver
We implement a R shinny webserver to predict the regulatory effects of genetic variants in GWAS loci, eQTLs and various genomic features. The webserver can be accessed from [link](https://mpravae.rc.ufl.edu/).

## Requirements and Installation

MpraVAE is implemented by Python3.
- Python >= 3.10.12
- numpy >= 1.25.2
- pytorch >= 2.0.0
- biopython >= 1.81

Download MpraVAE:
```Python
git clone https://github.com/yi-xiaa/MpraVAE
```

- Python
```Python
pip3 install -r requirements --user
```

- R
```R
install.packages(c("optparse", "dplyr", "data.table"))

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("metap", "BSgenome.Hsapiens.UCSC.hg38", "BSgenome.Hsapiens.UCSC.hg19", 
                       "Matrix", "SummarizedExperiment", "TFBSTools", "JASPAR2020"))
```

```command
Rscript -e 'install.packages(c("optparse", "dplyr", "data.table"))'
Rscript -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")'
Rscript -e 'BiocManager::install(c("metap", "BSgenome.Hsapiens.UCSC.hg38", "BSgenome.Hsapiens.UCSC.hg19", "Matrix", "SummarizedExperiment", "TFBSTools", "JASPAR2020"))'
```


## Usage
- Take summary data input.csv as input and output fasta files, the output files are pos.fasta, neg.fasta.
```command
Rscript fasta_generation.R  --input_file train.csv --output_dir train_data
Rscript fasta_generation.R  --input_file test.csv --output_dir test_data

Arguments:
  -i, --input_file            Path to the input file
  -o, --output_dir            Output directory
```

- Convert the fasta files into hdf5 format, the output would be sequences.h5.
```command
python hdf5_generation.py  --input_dir train_data  --output_dir train_data
python hdf5_generation.py  --input_dir test_data   --output_dir test_data

Arguments:
  --input_dir            Path to input directory
  --output_dir           Path to output directory
```

- Train MpraVAE model for synthetic data generation using sequences.h5 in train_data folder, the output would be MpraVAE.pth in model folder
```command
python MpraVAE_train.py  --input_file train.h5  --model_file MpraVAE.pth

Arguments:
  --input_file           Path to input data file
  --model_file           Path to MpraVAE model file
```

- Generate synthetic data using the MpraVAE model, specify the multiplier for the synthetic data sample size relative to the observed data. The output will be mpravae_generated_sequences.h5.
```command
python augment.py --model_file MpraVAE.pth --multiplier 5  --input_file  train.h5 --output_file mpravae_synthetic_sequences.h5

Arguments:
  --model_file           Path to MpraVAE model file
  --multiplier           Multiplier for generating sequences (default: 5)
  --input_file           Path to input file
  --output_file          Path to output file
```

- Train CNN classifier using both observed and MpraVAE synthetic data, the output would be CNN.pth
```command
python CNN_train.py  --input_files  train.h5,mpravae_synthetic_sequences.h5 --model_file CNN.pth

Arguments:
  --input_files          Comma-separated paths to the input files(1 observed, 1 synthetic)
  --model_file           Path to CNN model file
```

- Use the CNN classifier to give prediction for the test data, and append the output as a column in prediction.csv.
```command
python predict.py --model_file CNN.pth --input_file test.h5  --output_file prediction.csv

Arguments:
  --model_file           Path to CNN model
  --input_file           Path to the input h5 file
  --output_file          Path to output file
```



## Example
- The initial MPRA variants summary table should have chr, pos, log2FC, fdr information, such as [MPRA_autoimmune](https://github.com/yi-xiaa/MpraVAE/blob/main/data/MPRA_autoimmune.csv).
![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/pic1.png)

```command
git clone https://github.com/yi-xiaa/MpraVAE

module load R
cd .../MpraVAE/

Rscript code/fasta_generation.R --input_file data/MPRA_autoimmune_train.csv --output_dir data/train_data
Rscript code/fasta_generation.R --input_file data/MPRA_autoimmune_test.csv --output_dir data/test_data

module load conda
conda activate your_environment_name

python code/hdf5_generation.py  --input_dir data/train_data  --output_dir data/train_data
python code/hdf5_generation.py  --input_dir data/test_data   --output_dir data/test_data

python code/MpraVAE_train.py  --input_file data/train_data/sequences.h5  --model_file model/MpraVAE.pth

python code/augment.py --model_file model/MpraVAE.pth --multiplier 5  --input_file  data/train_data/sequences.h5 --output_file data/train_data/mpravae_synthetic_sequences.h5

python code/CNN_train.py  --input_files  data/train_data/sequences.h5,data/train_data/mpravae_synthetic_sequences.h5 --model_file model/CNN.pth

python code/predict.py --model_file model/CNN.pth --input_file data/test_data/sequences.h5  --output_file prediction/prediction.csv
```

## Reference
If you use `MpraVAE`, please cite:

[Jin, W., Xia, Y., Sai, S., Liu, Y., & Chen, L. (2024). In silico generation and augmentation of regulatory variants from massively parallel reporter assay using conditional variational autoencoder. *bioRxiv*.](https://doi.org/10.1101/2024.06.25.600715)


[lichen-lab](https://github.com/lichen-lab "https://github.com/lichen-lab")