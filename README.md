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
- Take summary data input.csv as input and output fasta files, the output files are train.[celltype/disease].pos.fasta, train.[celltype/disease].neg.fasta, test.[celltype/disease].pos.fasta and test.[celltype/disease].neg.fasta
```command
Rscript /path/to/fasta_generation.R  --data /data/input.csv --output /path/to/output_folder --test_size 0.2
```

- Convert the fasta files into hdf5 format, the output would be train.[celltype/disease].pos.h5, train.[celltype/disease].neg.h5, test.[celltype/disease].pos.h5 and test.[celltype/disease].neg.h5.
```command
python /path/to/hdf5_generation.py celltype_name/disease_name --lib_path /path/to/lib.py --data_folder /path/to/Data
```

- Train MpraVAE model for synthetic data generation using train.[celltype/disease].pos.fasta and train.[celltype/disease].neg.fasta in data_folder, the output would be MpraVAE.{celltype/disease}.pth
```command
python /path/to/MpraVAE_train.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --data_folder /path/to/Data --input_dir /path/to/input_data_folder --output_dir /path/to/output_folder
```

- Generate synthetic data using the MpraVAE model, specify the multiplier for the synthetic data sample size relative to the observed data. The output will be mpravae_generated.{celltype/disease}.pos.h5 and mpravae_generated.{celltype/disease}.neg.h5 containing both observed and synthetic data.
```command
python /path/to/augment.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --data_folder /path/to/Data --model_dir /path/to/MpraVAE_model_folder --output_dir /path/to/output_folder --multiplier 5
```

- Train CNN classifier using MpraVAE synthetic data, the output would be CNN.[celltype/disease].pth
```command
python /path/to/CNN_train.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --train_path /path/to/train.py --data_folder /path/to/Data --input_dir /path/to/input_data_folder --output_dir /path/to/output_folder
```

- Use the CNN classifier to give prediction for the test data, the output is one column append as column for test_prediction.csv
```command
python /path/to/predict.py --modelname "path/to/your_CNN_model.pth" --seq_input_path "/path/to/your/testdata.fasta" --outfolder "your_output_folder/"
```



## Example
- The initial MPRA variants summary table should have chr, pos, log2FC, fdr information, such as [MPRA_autoimmune](https://github.com/yi-xiaa/MpraVAE/blob/main/data/MPRA_autoimmune.csv).
![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/pic1.png)

```command
module load R
cd .../MpraVAE/

Rscript code/fasta_generation.R --data data/MPRA_autoimmune.csv --output data/ --test_size 0.2

module load conda
conda activate your_environment_name

python code/hdf5_generation.py autoimmune_disease --lib_path code/lib.py --data_folder data/

python code/MpraVAE_train.py autoimmune_disease --lib_path code/lib.py --model_path code/model.py --data_folder data/ --input_dir data/ --output_dir data/

python code/augment.py autoimmune_disease --lib_path code/lib.py --model_path code/model.py --data_folder data/ --model_dir model/ --output_dir /path/to/output_folder --multiplier 5

python code/CNN_train.py autoimmune_disease --lib_path code/lib.py --model_path code/model.py --train_path code/train.py --data_folder data/ --input_dir data/ --output_dir data/

python code/predict.py --modelname "model/CNN.autoimmune_disease.pth" --seq_input_path "data/test.autoimmune_disease.pos.fasta" --outfolder "result/"
python code/predict.py --modelname "model/CNN.autoimmune_disease.pth" --seq_input_path "data/test.autoimmune_disease.neg.fasta" --outfolder "result/"
```

## Reference
If you use `MpraVAE`, please cite:

    (awaiting for formal link)


[lichen-lab](https://github.com/lichen-lab "https://github.com/lichen-lab")