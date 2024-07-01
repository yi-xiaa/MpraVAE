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
- Command line to take summary data input.csv as input and output fasta files, the output files are train.[celltype/disease].pos.fasta, train.[celltype/disease].neg.fasta, test.[celltype/disease].pos.fasta and test.[celltype/disease].neg.fasta
```command
Rscript /path/to/fasta_generation.R  --data /data/input.csv --output /path/to/output_folder --test_size 0.2
```

- Command line to convert the fasta files into hdf5 format, the output would be train.[celltype/disease].pos.h5, train.[celltype/disease].neg.h5, test.[celltype/disease].pos.h5 and test.[celltype/disease].neg.h5.
```command
python /path/to/hdf5_generation.py celltype_name/disease_name --lib_path /path/to/lib.py --data_folder /path/to/Data
```

- Command line to train MpraVAE model for synthetic data generation using train.[celltype/disease].pos.fasta and train.[celltype/disease].neg.fasta in data_folder, the output would be MpraVAE.{celltype/disease}.pth
```command
python /path/to/MpraVAE_train.py celltype_name/disease_name --lib_path /path/to/lib.py --model_path /path/to/model.py --train_path /path/to/train.py --data_folder /path/to/Data --input_dir /path/to/input_data_folder --output_dir /path/to/output_folder
```

- Command line to generate synthetic data using the MpraVAE model, specify the multiplier for the sample size relative to the observed data. The output will be synthetic_data.h5 containing both observed and synthetic data.
```command
python /path/to/augment.py /path/to/MpraVAE.{celltype}.pth train_data.h5 --multiplier 6
```

- Command line to train CNN classifier using MpraVAE synthetic data, the output is CNN.pth
```command
python CNN_train.py synthetic_data.h5
```

- Command line to use the CNN classifier to give prediction for the test data, the output is CNN.pth, the output is one column append as column for test_prediction.vcf
```command
Python predict.py CNN.pth test_data.h5
```



## Example
- The initial MPRA variants summary table should have chr, pos, log2FC, fdr information, such as [MPRA_autoimmune](https://github.com/yi-xiaa/MpraVAE/blob/main/data/MPRA_autoimmune.csv).
![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/pic1.png)

- R command line to take the summary table as input, then output the fasta files for deep learning.
```command
module load R
cd .../MpraVAE/
Rscript code/fasta_generation.R --data data/MPRA_autoimmune.csv --output data/ --test_size 0.2
```

- Python command line to get the MpraVAE classifier, here we use Mpra autoimmune as example.
```command
module load conda
conda activate your_environment_name
python code/augment.py autoimmune_disease --lib_path code/lib.py --model_path code/model.py --train_path code/train.py --data_folder data/ --input_dir data/ --output_dir data/ --fasta_output_dir data/
```

- Python command line for classifier to give prediction of variants in example.fasta, the prediction probability would be save as an probs_out.csv in the result folder.
```command
python code/predict.py --modelname "model/VAE_autoimmune_disease.pth" --seq_input_path "data/example.fasta" --outfolder "result/"
```

## Reference
If you use `MpraVAE`, please cite:

    (awaiting for formal link)


[lichen-lab](https://github.com/lichen-lab "https://github.com/lichen-lab")