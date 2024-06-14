setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data')

source('prediction.lib.R')
source('genetics_lib.R')


#datapath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data'

load('gene.ref.rda')
load('human.motif.rda')

library(dplyr)
library(metap)

######################################################################################################
### paper 1: Functional regulatory variants implicate distinct transcriptional networks in dementia
setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data1')
outpath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data1'

idata=1
dat=fread('Functional regulatory variants implicate distinct transcriptional networks in dementia.xls')
dim(dat)

fdr.up=0.1
fdr.down=0.8
log2FC.thres=0.1

summary(dat$fdr)
summary(dat$log2FC)


dat$fdr[is.na(dat$fdr)]=1
dat$log2FC[is.na(dat$log2FC)]=0


dat=dat[!is.na(dat$fdr) & !is.na(dat$log2FC),]
dim(dat)


### by cell types
celltypes=names(table(dat$celltype))
celltypes

for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath)
}


### by disease
diseases=names(table(dat$disease))
diseases

for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1=dat[dat$disease==diseases[idisease],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),diseases[idisease],sep='.'),outpath)
}



######################################################################################################################
### paper 4  Genome-wide functional screen of 30 UTR variants uncovers causal variants for human disease and evolution
setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data4')
outpath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data4'

idata=4
dat=fread('Genome-wide functional screen of 30 UTR variants uncovers causal variants for human disease and evolution.xls')
dim(dat)



# for each celltype delete duplicates
# Remove duplicate entries by selecting the one with the smallest 'fdr' value for each chr+pos combination
dat <- dat[order(chr, pos, fdr), .SD[1], by = .(chr, pos)]
dim(dat)
head(dat)

summary_fdr_by_celltype <- dat[, .(
  Min_FDR = min(fdr, na.rm = TRUE),
  Max_FDR = max(fdr, na.rm = TRUE),
  Mean_FDR = mean(fdr, na.rm = TRUE),
  Median_FDR = median(fdr, na.rm = TRUE),
  Std_Dev_FDR = sd(fdr, na.rm = TRUE)
), by = celltype]
print(summary_fdr_by_celltype)




fdr.up=0.1
fdr.down=0.8
log2FC.thres=0.1


summary(dat$fdr)
summary(dat$log2FC)


dat$fdr[is.na(dat$fdr)]=1
dat$log2FC[is.na(dat$log2FC)]=0

dat=dat[!is.na(dat$fdr) & !is.na(dat$log2FC),]
dim(dat)


celltypes=names(table(dat$celltype))
celltypes

for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath)
}


### by disease
diseases=names(table(dat$disease))
diseases

for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1=dat[dat$disease==diseases[idisease],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),diseases[idisease],sep='.'),outpath)
}



######################################################################################################
### Paper 6 Transcriptional-regulatory convergence across functional MDD risk variants identified by massively parallel reporter assays 
setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data6')
outpath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data6'

idata=6
dat=fread('Transcriptional-regulatory convergence across functional MDD risk variants identified by massively parallel reporter assays.xls')
dim(dat)


fdr.up=0.1
fdr.down=0.5
log2FC.thres=0.1

summary(dat$fdr)
summary(dat$log2FC)


dat$fdr[is.na(dat$fdr)]=1
dat$log2FC[is.na(dat$log2FC)]=0

dat=dat[!is.na(dat$fdr) & !is.na(dat$log2FC),]
dim(dat)


### by cell types
celltypes=names(table(dat$celltype))
celltypes


for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath)
}

### by disease
dat$disease[dat$disease=='major depressive disorder']='major_depressive_disorder'
diseases=names(table(dat$disease))
diseases

for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1=dat[dat$disease==diseases[idisease],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),diseases[idisease],sep='.'),outpath)
}


#############################################################################################################################################################################
### paper 7: Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution
setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data7')
outpath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data7'

idata=7
dat=fread('Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution.GRCh37_ALL.xls')
dim(dat)
head(dat)
sum(dat$celltype=="SK-MEL-28")



combine_pvalues <- function(pvalues) {
  pvalues <- pmax(pvalues, 1e-10) # Add a small constant to avoid zero p-values
  if (length(pvalues) < 2) { 
    return(ifelse(length(pvalues) == 1, pvalues, NA))
  } else {
    return(sumlog(pvalues)$p)
  }
}

combined_df <- dat %>%
  group_by(chr, pos, celltype) %>%
  summarize(
    ref = first(ref),
    alt = first(alt),
    log2FC = mean(log2FC, na.rm = TRUE),
    #combined_pvalue = sumlog(pvalue)$p,
    combined_pvalue = combine_pvalues(pvalue),
    gene_name = first(gene_name),
    fdr = first(fdr),
    gene_start = first(gene_start),
    gene_end = first(gene_end),
    disease = first(disease),
    strand = first(strand),
    genome = first(genome),
    GWASorQTL = first(GWASorQTL),
    insideGene = first(insideGene),
    distancetoGene = first(distancetoGene),
    rsid = first(rsid)
  ) %>%
  ungroup()

dim(combined_df)
head(combined_df)
table(combined_df$celltype)

sum(combined_df$celltype=="SK-MEL-28")
head(combined_df[combined_df$celltype=="SK-MEL-28",])

combined_df <- combined_df %>%
  group_by(celltype) %>%
  mutate(adjusted_pvalue = p.adjust(combined_pvalue, method = "fdr")) %>%
  ungroup()

sum(is.na(combined_df$combined_pvalue))

# testdf <- combined_df[combined_df$celltype=="SK-MEL-28",]
# sum(testdf$adjusted_pvalue<0.1)
# sum(testdf$adjusted_pvalue>0.8)

dat <- combined_df
dat$fdr <- dat$adjusted_pvalue



fdr.up=0.1
fdr.down=0.8
log2FC.thres=0.1

summary(dat$fdr)
summary(dat$log2FC)
dat$celltype[dat$celltype=='NIH/3T3']='NIH-3T3'


dat$fdr[is.na(dat$fdr)]=1
dat$log2FC[is.na(dat$log2FC)]=0

dat=dat[!is.na(dat$fdr) & !is.na(dat$log2FC),]
dim(dat)


### by cell types
celltypes=names(table(dat$celltype))
celltypes


for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  #id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>0.1
  id.pos=dat1$fdr<fdr.up
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath)
}


### by disease
dat$disease[dat$disease=='Bernard-Soulier Syndrome']='Bernard-Soulier_Syndrome'
dat$disease[dat$disease=='Cleft lip']='Cleft_lip'
dat$disease[dat$disease=='Familial hypercholesterol emia']='Familial_hypercholesterol_emia'
dat$disease[dat$disease=='Hemophilia B']='Hemophilia_B'
dat$disease[dat$disease=='Hereditary persistence of fetal hemoglobin']='Hereditary_persistence_of_fetal_hemoglobin'
dat$disease[dat$disease=='Human pigmentation']='Human_pigmentation'
dat$disease[dat$disease=='Limb malformations']='Limb_malformations'
dat$disease[dat$disease=='Maturity-onset diabetes of the young (MODY)']='Maturity-onset_diabetes_of_the_young_MODY'
dat$disease[dat$disease=='Plasma low-density lipoprotein cholesterol & myocardial infarction']='Plasma_low-density_lipoprotein_cholesterol_n_myocardial_infarction'
dat$disease[dat$disease=='Prostate cancer']='Prostate_cancer'
dat$disease[dat$disease=='Pyruvate kinase deficiency']='Pyruvate_kinase_deficiency'
dat$disease[dat$disease=='Sickle cell disease']='Sickle_cell_disease'
dat$disease[dat$disease=='Thyroid cancer']='Thyroid_cancer'
dat$disease[dat$disease=='Type 2 diabetes']='Type2_diabetes'
dat$disease[dat$disease=='Various types of cancer']='Various_types_of_cancer'
diseases=names(table(dat$disease))
diseases

for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1=dat[dat$disease==diseases[idisease],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),diseases[idisease],sep='.'),outpath)
}



#######################################################################################################################################
### paper 10:  Prioritization of autoimmune disease-associated genetic variants that perturb regulatory element activity in T cells
setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data10')
outpath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data10'

idata=10
dat=fread('Prioritization of autoimmune disease-associated genetic variants that perturb regulatory element activity in T cells(preprocessed).csv')
dim(dat)
summary(dat$fdr)
summary(dat$log2FC)

fdr.up=0.1
fdr.down=0.8
log2FC.thres=0.1


dat$fdr[is.na(dat$fdr)]=1
dat$log2FC[is.na(dat$log2FC)]=0
summary(dat$fdr)
summary(dat$log2FC)


dat=dat[!is.na(dat$fdr) & !is.na(dat$log2FC),]
dim(dat)

### Remove sequences with invalid characters (any character other than A, C, G, T)
# tmp=dat[,c('chr','pos','pos')]
# colnames(tmp)=c('chr','start','end')
# gr=makeGRangesFromDataFrame(tmp)
# gr.ext = extendReg(gr, ext = 1000)
# 
# if(unique(dat$genome)=='hg38'){
#   genome=BSgenome.Hsapiens.UCSC.hg38
# }else{
#   genome=BSgenome.Hsapiens.UCSC.hg19
# }
# seq.ext = getSeq(genome, gr.ext)
# 
# invalid_seqs <- sapply(seq.ext, function(seq) any(grepl("[^ACGT]", seq)))
# print(paste0("invalid_row_numbers: ", which(invalid_seqs)))
# dat <- dat[!invalid_seqs, ]
# dim(dat)
#############################################################

### by cell types
celltypes=names(table(dat$celltype))
celltypes


for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  # very imbalanced case how to carefully selected negative set besides statistical signifcance??? (should make the data balanced!!!)
  neg_indices = which(id.neg) # Get indices where id.neg is TRUE
  sampled_neg_indices = sample(neg_indices, min(10 * sum(id.pos), length(neg_indices)))
  id.neg = rep(FALSE, length(id.neg))
  id.neg[sampled_neg_indices] = TRUE
  
  message(sum(id.pos), ' ', sum(id.neg))
  
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath)
}


### by disease
dat$disease[dat$disease=='autoimmune disease']='autoimmune_disease'
diseases=names(table(dat$disease))
diseases



############ Method 1: only keep one variants if multiple entries closer than 1kb
filtered_dat <- dat[order(dat$disease, dat$chr, dat$pos), ]
keep <- rep(TRUE, nrow(filtered_dat))
for (i in 2:nrow(filtered_dat)) {
  if (filtered_dat$disease[i] == filtered_dat$disease[i-1] && filtered_dat$chr[i] == filtered_dat$chr[i-1]) {
    if (filtered_dat$pos[i] - filtered_dat$pos[i-1] < 1000) {
      keep[i] <- FALSE
    }
  }
}
filtered_dat <- filtered_dat[keep, ]
print(filtered_dat)
dim(filtered_dat)
dat <- filtered_dat
##################################

############# Method 2: split by chromasome
for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1 = dat[dat$disease == diseases[idisease], ]
  
  unique_chrs <- unique(dat1$chr)
  for(ichr in 1:length(unique_chrs)){
    message(unique_chrs[ichr])
    dat_chr = dat1[dat1$chr == unique_chrs[ichr], ]
    id.pos = dat_chr$fdr < fdr.up & abs(dat_chr$log2FC) > log2FC.thres
    id.neg = dat_chr$fdr > fdr.down
    
    message(sum(id.pos), ' ', sum(id.neg))
    
    neg_indices = which(id.neg)
    sampled_neg_indices = sample(neg_indices, min(10 * sum(id.pos), length(neg_indices)))
    id.neg = rep(FALSE, length(id.neg))
    id.neg[sampled_neg_indices] = TRUE
    
    message(sum(id.pos), ' ', sum(id.neg))
    
    # # Evaluate performance and create sequences for each chromosome within each disease
    # evalPerf(dat_chr, id.pos, id.neg, motifs, feature.type='3mer',
    #          name.export=paste(paste0('data', idata), diseases[idisease], chr, sep='.'), outpath, is.output=T)
    # 
    # evalPerf(dat_chr, id.pos, id.neg, motifs, feature.type='motif',
    #          name.export=paste(paste0('data', idata), diseases[idisease], chr, sep='.'), outpath, is.output=F)
    # 
    # createRevAndCropSeq(dat_chr, id.pos, id.neg, ext.crop=1000, ntimes=2, 
    #                     paste(paste0('data', idata), diseases[idisease], chr, sep='.'), outpath)
  }
}


###
subset1 <- data.frame()
subset2 <- data.frame()

subset1_chromosomes <- c()
subset2_chromosomes <- c()

for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1 = dat[dat$disease == diseases[idisease], ]
  
  unique_chrs <- unique(dat1$chr)
  
  chr_pos_counts <- sapply(unique_chrs, function(chr) {
    dat_chr <- dat1[dat1$chr == chr, ]
    sum(dat_chr$fdr < fdr.up & abs(dat_chr$log2FC) > log2FC.thres)
  })
  
  chr_pos_counts_df <- data.frame(chr = unique_chrs, pos_count = chr_pos_counts)
  chr_pos_counts_df <- chr_pos_counts_df[order(chr_pos_counts_df$pos_count, decreasing = TRUE), ]
  
  total_pos <- sum(chr_pos_counts_df$pos_count)
  target_pos_80 <- 0.8 * total_pos
  
  accumulated_pos <- 0
  for(i in 1:nrow(chr_pos_counts_df)){
    if(accumulated_pos < target_pos_80){
      subset1 <- rbind(subset1, dat1[dat1$chr == chr_pos_counts_df$chr[i], ])
      subset1_chromosomes <- c(subset1_chromosomes, chr_pos_counts_df$chr[i])
      accumulated_pos <- accumulated_pos + chr_pos_counts_df$pos_count[i]
    } else {
      subset2 <- rbind(subset2, dat1[dat1$chr == chr_pos_counts_df$chr[i], ])
      subset2_chromosomes <- c(subset2_chromosomes, chr_pos_counts_df$chr[i])
    }
  }
}

subset1_pos_count <- sum(subset1$fdr < fdr.up & abs(subset1$log2FC) > log2FC.thres)
subset2_pos_count <- sum(subset2$fdr < fdr.up & abs(subset2$log2FC) > log2FC.thres)

message("Subset 1 sum(id.pos): ", subset1_pos_count)
message("Subset 2 sum(id.pos): ", subset2_pos_count)

message("Chromosomes in Subset 1: ", paste(subset1_chromosomes, collapse = ", "))
message("Chromosomes in Subset 2: ", paste(subset2_chromosomes, collapse = ", "))

message("Ratio: ", subset1_pos_count / (subset1_pos_count + subset2_pos_count), " : ", subset2_pos_count / (subset1_pos_count + subset2_pos_count))
##########################



for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1=dat[dat$disease==diseases[idisease],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  # very imbalanced case how to carefully selected negative set besides statistical signifcance??? (should make the data balanced!!!)
  neg_indices = which(id.neg) # Get indices where id.neg is TRUE
  sampled_neg_indices = sample(neg_indices, min(10 * sum(id.pos), length(neg_indices)))
  id.neg = rep(FALSE, length(id.neg))
  id.neg[sampled_neg_indices] = TRUE
  
  message(sum(id.pos), ' ', sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),diseases[idisease],sep='.'),outpath)
}



######################################################################################################
### paper 11: Massively parallel reporter assays and variant scoring identified functional variants and target genes for melanoma loci and highlighted cell-type specificity 
setwd('/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data11')
outpath='/blue/li.chen1/yi.xia/Projects/MPRA/Additional_data/Data11'

idata=11
dat=fread('Massively parallel reporter assays and variant scoring identified functional variants and target genes for melanoma loci and highlighted cell-type specificity.xls')
dim(dat)
summary(dat$fdr)
summary(dat$log2FC)

fdr.up=0.1
fdr.down=0.8
log2FC.thres=0.1


dat$fdr[is.na(dat$fdr)]=1
dat$log2FC[is.na(dat$log2FC)]=0
summary(dat$fdr)
summary(dat$log2FC)

dat$celltype[dat$celltype=='malignant melanoma']='malignant_melanoma'
dat$celltype[dat$celltype=='normal melanocyte']='normal_melanocyte'

dat=dat[!is.na(dat$fdr) & !is.na(dat$log2FC),]
dim(dat)


### by cell types
celltypes=names(table(dat$celltype))
celltypes


for(icelltype in 1:length(celltypes)){
  message(celltypes[icelltype])
  dat1=dat[dat$celltype==celltypes[icelltype],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),celltypes[icelltype],sep='.'),outpath)
}


### by disease
diseases=names(table(dat$disease))
diseases




############# Method 2: split by chromasome
for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1 = dat[dat$disease == diseases[idisease], ]
  
  unique_chrs <- unique(dat1$chr)
  for(ichr in 1:length(unique_chrs)){
    message(unique_chrs[ichr])
    dat_chr = dat1[dat1$chr == unique_chrs[ichr], ]
    id.pos = dat_chr$fdr < fdr.up & abs(dat_chr$log2FC) > log2FC.thres
    id.neg = dat_chr$fdr > fdr.down
    
    message(sum(id.pos), ' ', sum(id.neg))
    
    neg_indices = which(id.neg)
    sampled_neg_indices = sample(neg_indices, min(10 * sum(id.pos), length(neg_indices)))
    id.neg = rep(FALSE, length(id.neg))
    id.neg[sampled_neg_indices] = TRUE
    
    message(sum(id.pos), ' ', sum(id.neg))
    
    # # Evaluate performance and create sequences for each chromosome within each disease
    # evalPerf(dat_chr, id.pos, id.neg, motifs, feature.type='3mer',
    #          name.export=paste(paste0('data', idata), diseases[idisease], chr, sep='.'), outpath, is.output=T)
    # 
    # evalPerf(dat_chr, id.pos, id.neg, motifs, feature.type='motif',
    #          name.export=paste(paste0('data', idata), diseases[idisease], chr, sep='.'), outpath, is.output=F)
    # 
    # createRevAndCropSeq(dat_chr, id.pos, id.neg, ext.crop=1000, ntimes=2, 
    #                     paste(paste0('data', idata), diseases[idisease], chr, sep='.'), outpath)
  }
}


###
subset1 <- data.frame()
subset2 <- data.frame()

subset1_chromosomes <- c()
subset2_chromosomes <- c()

for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1 = dat[dat$disease == diseases[idisease], ]
  
  unique_chrs <- unique(dat1$chr)
  
  chr_pos_counts <- sapply(unique_chrs, function(chr) {
    dat_chr <- dat1[dat1$chr == chr, ]
    sum(dat_chr$fdr < fdr.up & abs(dat_chr$log2FC) > log2FC.thres)
  })
  
  chr_pos_counts_df <- data.frame(chr = unique_chrs, pos_count = chr_pos_counts)
  chr_pos_counts_df <- chr_pos_counts_df[order(chr_pos_counts_df$pos_count, decreasing = TRUE), ]
  
  total_pos <- sum(chr_pos_counts_df$pos_count)
  target_pos_80 <- 0.8 * total_pos
  
  accumulated_pos <- 0
  for(i in 1:nrow(chr_pos_counts_df)){
    if(accumulated_pos < target_pos_80){
      subset1 <- rbind(subset1, dat1[dat1$chr == chr_pos_counts_df$chr[i], ])
      subset1_chromosomes <- c(subset1_chromosomes, chr_pos_counts_df$chr[i])
      accumulated_pos <- accumulated_pos + chr_pos_counts_df$pos_count[i]
    } else {
      subset2 <- rbind(subset2, dat1[dat1$chr == chr_pos_counts_df$chr[i], ])
      subset2_chromosomes <- c(subset2_chromosomes, chr_pos_counts_df$chr[i])
    }
  }
}

subset1_pos_count <- sum(subset1$fdr < fdr.up & abs(subset1$log2FC) > log2FC.thres)
subset2_pos_count <- sum(subset2$fdr < fdr.up & abs(subset2$log2FC) > log2FC.thres)

message("Subset 1 sum(id.pos): ", subset1_pos_count)
message("Subset 2 sum(id.pos): ", subset2_pos_count)

message("Chromosomes in Subset 1: ", paste(subset1_chromosomes, collapse = ", "))
message("Chromosomes in Subset 2: ", paste(subset2_chromosomes, collapse = ", "))

message("Ratio: ", subset1_pos_count / (subset1_pos_count + subset2_pos_count), " : ", subset2_pos_count / (subset1_pos_count + subset2_pos_count))
##########################







for(idisease in 1:length(diseases)){
  message(diseases[idisease])
  dat1=dat[dat$disease==diseases[idisease],]
  id.pos=dat1$fdr<fdr.up & abs(dat1$log2FC)>log2FC.thres
  id.neg=dat1$fdr>fdr.down
  
  message(sum(id.pos),' ',sum(id.neg))
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='3mer',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=T)
  
  evalPerf(dat1,id.pos,id.neg,motifs,feature.type='motif',
           name.export=paste(paste0('data',idata),diseases[idisease],sep='.'),outpath,is.output=F)
  
  createRevAndCropSeq(dat1,id.pos,id.neg,ext.crop=1000,ntimes=2,paste(paste0('data',idata),diseases[idisease],sep='.'),outpath)
}




