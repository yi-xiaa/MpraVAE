# MpraVAE
A deep generative model to augment genetic variants sample size, enhancing the prediction of regulatory variants in non-coding regions built on top of [Pytorch](https://pytorch.org/).

## Introduction
Predicting the functional impact of genetic variants in non-coding regions is challenging. Massively parallel reporter assays (MPRAs) can test thousands of variants for allele-specific regulatory activity, but typically only identify a few hundred labelled variants, limiting their use for genome-wide prediction. 

MpraVAE, a deep generative model, addresses this limitation by augmenting the training sample size of labelled variants. Benchmarking on multiple MPRA datasets shows that MpraVAE significantly improves the prediction of regulatory variants compared to conventional data augmentation methods and existing scoring techniques. Taking autoimmune diseases as one example, MpraVAE enabled genome-wide de novo prediction of regulatory variants, revealing their enrichment in enhancers, active histone marks, immune-related cell types, and key regulatory sites. These variants also facilitated the discovery of immune-related genes through integration with PCHi-C and DNase-seq data, highlighting MpraVAE's importance in genetic and gene discovery for complex traits.

![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/Figure1.png)

## MpraVAE Webserver
We implement a R shinny webserver to predict the regulatory effects of genetic variants in GWAS loci, eQTLs and various genomic features. The webserver can be accessed from [link](https://mpravae.rc.ufl.edu/).

## Requirements and Installation

- R
```R
install.packages(c("dplyr", "data.table", "randomForest", "cvTools", "ROCR"))

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("metap", "GenomicRanges", "ChIPpeakAnno", "EnsDb.Hsapiens.v75", 
                       "EnsDb.Hsapiens.v86", "SNPlocs.Hsapiens.dbSNP144.GRCh38", 
                       "SNPlocs.Hsapiens.dbSNP144.GRCh37", "BSgenome.Hsapiens.UCSC.hg19", 
                       "BSgenome.Hsapiens.UCSC.hg38", "TxDb.Hsapiens.UCSC.hg19.knownGene", 
                       "TxDb.Hsapiens.UCSC.hg38.knownGene", "org.Hs.eg.db", 
                       "motifmatchr", "Matrix", "SummarizedExperiment", 
                       "TFBSTools", "JASPAR2020"))
```

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


## Usage

## Example
- The initial MPRA variants summary table should have chr, pos, log2FC, fdr information, such as
![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/pic1.png)

- R command line to take the summary table as input, then output the input file for deep learning. This step will output several .fasta files.
```command
Rscript /path/to/main.R --code /path/to/code/folder --data data/input.csv --gene_ref /path/to/gene.ref.rda --motif /path/to/human.motif.rda --output /path/to/output
```

- Python command line to get the MpraVAE classifier, here we use Mpra autoimmune as example.
```command
python /path/to/main.py autoimmune_disease
```

- Python command line for classifier to predict the label of variants in input.fasta
```command
python /path/to/predict.py --modelname "path/to/your_model_name.pth" --seq_input_path "/path/to/your/input.fasta" --outfolder "your_output_folder/"
```

## Documentation
We provide several tutorials and user guide. If you find our tool useful for your research, please consider citing the MpraVAE manuscript.

<table>
  <tr>
    <td><a href="URL_TO_USER_GUIDE">User guide</a></td>
    <td><a href="URL_TO_PBMCS_TUTORIAL">Data Preprocess</a></td>
    <td><a href="URL_TO_GRN_BENCHMARK">Data Augmentaion</a></td>
  </tr>
</table>

## Reference
If you use `MpraVAE`, please cite:

    (awaiting for formal link)



[lichen-lab](https://github.com/lichen-lab "https://github.com/lichen-lab")