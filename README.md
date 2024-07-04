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
Rscript fasta_generation.R  --input_file data/train.csv --output_dir data/train_data
Rscript fasta_generation.R  --input_file data/test.csv --output_dir data/test_data
```

- Convert the fasta files into hdf5 format, the output would be sequences.h5.
```command
python hdf5_generation.py  --input_dir data/train_data  --output_dir data/train_data
python hdf5_generation.py  --input_dir data/test_data   --output_dir data/test_data
```

- Train MpraVAE model for synthetic data generation using sequences.h5 in train_data folder, the output would be MpraVAE.pth in model folder
```command
python MpraVAE_train.py  --input_file data/train_data/train.h5  --model_dir model/

python MpraVAE_train.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --data_folder /path/to/Data --input_dir /path/to/input_data_folder --output_dir /path/to/output_folder
```





- Take summary data input.csv as input and output fasta files, the output files are train.[celltype/disease].pos.fasta, train.[celltype/disease].neg.fasta, test.[celltype/disease].pos.fasta and test.[celltype/disease].neg.fasta
```command
Rscript fasta_generation.R  --data /data/input.csv --output /path/to/output_folder --test_size 0.2
```

- Convert the fasta files into hdf5 format, the output would be train.[celltype/disease].pos.h5, train.[celltype/disease].neg.h5, test.[celltype/disease].pos.h5 and test.[celltype/disease].neg.h5.
```command
python hdf5_generation.py celltype_name/disease_name --lib_path /path/to/lib.py --data_folder /path/to/Data
```

- Train MpraVAE model for synthetic data generation using train.[celltype/disease].pos.fasta and train.[celltype/disease].neg.fasta in data_folder, the output would be MpraVAE.{celltype/disease}.pth
```command
python MpraVAE_train.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --data_folder /path/to/Data --input_dir /path/to/input_data_folder --output_dir /path/to/output_folder
```

- Generate synthetic data using the MpraVAE model, specify the multiplier for the synthetic data sample size relative to the observed data. The output will be mpravae_generated.{celltype/disease}.pos.h5 and mpravae_generated.{celltype/disease}.neg.h5 containing both observed and synthetic data.
```command
python augment.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --data_folder /path/to/Data --model_dir /path/to/MpraVAE_model_folder --output_dir /path/to/output_folder --multiplier 5
```

- Train CNN classifier using MpraVAE synthetic data, the output would be CNN.[celltype/disease].pth
```command
python CNN_train.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --train_path /path/to/train.py --data_folder /path/to/Data --input_dir /path/to/input_data_folder --output_dir /path/to/output_folder
```

- Use the CNN classifier to give prediction for the test data, the output is one column append as column for test_prediction.csv
```command
python predict.py --modelname "path/to/your_CNN_model.pth" --seq_input_path "/path/to/your/testdata.fasta" --outfolder "your_output_folder/"
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

    (awaiting for formal link)


[lichen-lab](https://github.com/lichen-lab "https://github.com/lichen-lab")