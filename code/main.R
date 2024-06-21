# main.R

library(optparse)
library(dplyr)
library(data.table)
library(metap)

option_list = list(
  make_option(c("-d", "--code"), type="character", default=NULL, help="Path to the dependence code", metavar="character"),
  make_option(c("-d", "--data"), type="character", default=NULL, help="Path to the input data", metavar="character"),
  make_option(c("-g", "--gene_ref"), type="character", default=NULL, help="Path to the gene reference file", metavar="character"),
  make_option(c("-m", "--motif"), type="character", default=NULL, help="Path to the human motif file", metavar="character"),
  make_option(c("-o", "--output"), type="character", default="./", help="Output directory", metavar="character"),
  make_option(c("-i", "--idata"), type="integer", default=10, help="idata value", metavar="integer")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

setwd(dirname(opt$code))
source('prediction.lib.R')
source('genetics_lib.R')


load(opt$gene_ref)
load(opt$motif)

setwd(opt$output)
dat = fread(opt$data)
print(dim(dat))
print(summary(dat$fdr))
print(summary(dat$log2FC))

fdr.up = 0.1
fdr.down = 0.8
log2FC.thres = 0.1

dat$fdr[is.na(dat$fdr)] = 1
dat$log2FC[is.na(dat$log2FC)] = 0
dat = dat[!is.na(dat$fdr) & !is.na(dat$log2FC), ]
print(dim(dat))

dat$disease[dat$disease == 'autoimmune disease'] = 'autoimmune_disease'
diseases = unique(dat$disease)
print(diseases)

for (idisease in diseases) {
  message(idisease)
  dat1 = dat[dat$disease == idisease, ]
  id.pos = dat1$fdr < fdr.up & abs(dat1$log2FC) > log2FC.thres
  id.neg = dat1$fdr > fdr.down
  
  message(sum(id.pos), ' ', sum(id.neg))
  
  neg_indices = which(id.neg)
  sampled_neg_indices = sample(neg_indices, min(10 * sum(id.pos), length(neg_indices)))
  id.neg = rep(FALSE, length(id.neg))
  id.neg[sampled_neg_indices] = TRUE
  
  message(sum(id.pos), ' ', sum(id.neg))
  
  evalPerf(dat1, id.pos, id.neg, motifs, feature.type = '3mer',
           name.export = paste(paste0('data', opt$idata), idisease, sep = '.'), outpath = opt$output, is.output = TRUE)
  
  evalPerf(dat1, id.pos, id.neg, motifs, feature.type = 'motif',
           name.export = paste(paste0('data', opt$idata), idisease, sep = '.'), outpath = opt$output, is.output = FALSE)
  
  createRevAndCropSeq(dat1, id.pos, id.neg, ext.crop = 1000, ntimes = 2,
                      paste(paste0('data', opt$idata), idisease, sep = '.'), opt$output)
}
