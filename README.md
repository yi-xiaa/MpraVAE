# MpraVAE
A deep generative model to augment genetic variants sample size, enhancing the prediction of regulatory variants in non-coding regions built on top of [Pytorch](https://pytorch.org/).

## Introduction
Predicting the functional impact of genetic variants in non-coding regions is challenging. Massively parallel reporter assays (MPRAs) can test thousands of variants for allele-specific regulatory activity, but typically only identify a few hundred labelled variants, limiting their use for genome-wide prediction. 

MpraVAE, a deep generative model, addresses this limitation by augmenting the training sample size of labelled variants. Benchmarking on multiple MPRA datasets shows that MpraVAE significantly improves the prediction of regulatory variants compared to conventional data augmentation methods and existing scoring techniques. Taking autoimmune diseases as one example, MpraVAE enabled genome-wide de novo prediction of regulatory variants, revealing their enrichment in enhancers, active histone marks, immune-related cell types, and key regulatory sites. These variants also facilitated the discovery of immune-related genes through integration with PCHi-C and DNase-seq data, highlighting MpraVAE's importance in genetic and gene discovery for complex traits.


![](https://github.com/yi-xiaa/MpraVAE/blob/main/doc/Figure1.png)

## Requirements and Installation
- R
```R
BiocManager::install(c("biomformat","edgeR","DESeq2"))
install.packages(c('ggplot2','gridExtra','lattice','reshape2','MASS','dirmult','nonnest2'))
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
  </tr>
  <tr>
    <td><a href="URL_TO_GRN_BENCHMARK">Data Augmentaion</a></td>
    <td><a href="URL_TO_GRN_BENCHMARK">Benchmark</a></td>
  </tr>
</table>

## Reference
If you use `MpraVAE`, please cite:

    Ye Wang, Li Chen, DeepPerVar: a multi-modal deep learning framework for functional interpretation of genetic variants in personal genome, Bioinformatics, Volume 38, Issue 24, 15 December 2022, Pages 5340–5351, [https://doi.org/10.1093/bioinformatics/btac696](https://doi.org/10.1093/bioinformatics/btac696).



[lichen-lab](https://github.com/lichen-lab "https://github.com/lichen-lab")