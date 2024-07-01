library(optparse)
library(dplyr)
library(data.table)
library(metap)
library(BSgenome.Hsapiens.UCSC.hg38)
library(BSgenome.Hsapiens.UCSC.hg19)
library(Matrix)
library(SummarizedExperiment)
library(TFBSTools)
library(JASPAR2020)
library(BSgenome)
library(GenomicRanges)
library(Biostrings)

extendReg<-function(gr,ext=500){
  start(gr)=start(gr)-ext
  end(gr)=end(gr)+ext
  gr
}

fasta_generation<-function(dat,id.pos,id.neg,name.export,outpath, test_size){
  if(unique(dat$genome)=='hg38'){
    genome=BSgenome.Hsapiens.UCSC.hg38
  }else{
    genome=BSgenome.Hsapiens.UCSC.hg19
  }
  
  tmp=dat[,c('chr','pos','pos')]
  colnames(tmp)=c('chr','start','end')
  gr=makeGRangesFromDataFrame(tmp)
  
  gr.ext=extendReg(gr,ext=500)
  seq.ext=getSeq(genome,gr.ext)
  seq.ext.pos=seq.ext[id.pos]
  seq.ext.neg=seq.ext[id.neg]
  
  names(seq.ext.pos)=paste0('pos',1:length(seq.ext.pos))
  names(seq.ext.neg)=paste0('neg',1:length(seq.ext.neg))

  num_test_pos <- ceiling(length(seq.ext.pos) * test_size)
  num_test_neg <- ceiling(length(seq.ext.neg) * test_size)
  
  test_pos_indices <- sample(seq_along(seq.ext.pos), num_test_pos)
  test_neg_indices <- sample(seq_along(seq.ext.neg), num_test_neg)
  
  train_pos_indices <- setdiff(seq_along(seq.ext.pos), test_pos_indices)
  train_neg_indices <- setdiff(seq_along(seq.ext.neg), test_neg_indices)
  
  seq.train.pos <- seq.ext.pos[train_pos_indices]
  seq.train.neg <- seq.ext.neg[train_neg_indices]
  seq.test.pos <- seq.ext.pos[test_pos_indices]
  seq.test.neg <- seq.ext.neg[test_neg_indices]
  
  writeXStringSet(seq.train.pos, file = file.path(outpath, paste0('train.', name.export, '.pos.fasta')))
  writeXStringSet(seq.train.neg, file = file.path(outpath, paste0('train.', name.export, '.neg.fasta')))
  writeXStringSet(seq.test.pos, file = file.path(outpath, paste0('test.', name.export, '.pos.fasta')))
  writeXStringSet(seq.test.neg, file = file.path(outpath, paste0('test.', name.export, '.neg.fasta')))
}

option_list = list(
  make_option(c("-d", "--data"), type="character", default="./", help="Path to the input data", metavar="character"),
  make_option(c("-o", "--output"), type="character", default="./", help="Output directory", metavar="character"),
  make_option(c("-t", "--test_size"), type="numeric", default=0.2, help="Proportion of data to be used for testing", metavar="numeric")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

#setwd(opt$output)
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

# by cell types
celltypes=names(table(dat$celltype))
celltypes

for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  fasta_generation(dat1,id.pos,id.neg,paste(celltypes[icelltype],sep='.'), opt$output, opt$test_size)
}

# by disease
diseases = unique(dat$disease)
print(diseases)

for (idisease in diseases) {
  message(idisease)
  dat1 = dat[dat$disease == idisease, ]
  id.pos = dat1$fdr < fdr.up & abs(dat1$log2FC) > log2FC.thres
  id.neg = dat1$fdr > fdr.down
  
  message(sum(id.pos), ' ', sum(id.neg))
  
  fasta_generation(dat1, id.pos, id.neg, paste(idisease, sep = '.'), opt$output, opt$test_size)
}




