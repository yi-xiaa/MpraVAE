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