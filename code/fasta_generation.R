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

extendReg<-function(gr,ext=500){
  start(gr)=start(gr)-ext
  end(gr)=end(gr)+ext
  gr
}

fasta_generation<-function(dat,id.pos,id.neg,name.export,outpath){
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
  writeXStringSet(seq.ext.pos,file=file.path(outpath,paste0('seq.',name.export,'.pos.fasta')))
  writeXStringSet(seq.ext.neg,file=file.path(outpath,paste0('seq.',name.export,'.neg.fasta')))
}


option_list = list(
  make_option(c("-d", "--data"), type="character", default=NULL, help="Path to the input data", metavar="character"),
  make_option(c("-o", "--output"), type="character", default="./", help="Output directory", metavar="character")
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
  
  createRevAndCropSeq(dat1,id.pos,id.neg,paste(celltypes[icelltype],sep='.'), opt$output)
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
  
  fasta_generation(dat1, id.pos, id.neg, paste(idisease, sep = '.'), opt$output)
}




