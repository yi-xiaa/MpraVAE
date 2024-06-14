
#library('GenomicRanges')
library(data.table)
library(ChIPpeakAnno)
library(EnsDb.Hsapiens.v75) ##(hg19)
library(EnsDb.Hsapiens.v86) ##(hg38)

library(SNPlocs.Hsapiens.dbSNP144.GRCh38)
library(SNPlocs.Hsapiens.dbSNP144.GRCh37)

library(BSgenome.Hsapiens.UCSC.hg19)
library(BSgenome.Hsapiens.UCSC.hg38)


library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)


library(org.Hs.eg.db)


#setwd('~/Dropbox (UFL)/IU/reading/genetics/MPRA/genetics_intermediate')

#source('genetics_lib.R')


#indir='~/Dropbox (UFL)/IU/reading/genetics/MPRA/genetics_intermediate'

#outdir='~/Dropbox (UFL)/IU/reading/genetics/MPRA/genetics_processed_clean'



#library(diffloop)


# obtain nearest gene of each variant

# Select annotations for intersection with regions
# Note inclusion of custom annotation, and use of shortcuts


#annots = c('hg19_basicgenes', 'hg19_genes_intergenic')


# Build the annotations (a single GRanges object)
#annotations = build_annotations(genome = 'hg19', annotations = annots)


removeDup<-function(df){
  
  library(dplyr)
  #df$key=paste(df$chr,df$pos)
  df$key=df$rsid
  
  result_df <- df %>%
    group_by(key) %>%
    slice_max(abs(log2FC),n=1)
   # slice_min(fdr)
  
  result_df=as.data.frame(result_df)
  result_df=result_df[,-ncol(result_df)]
  result_df=data.table(result_df)
  result_df
  
}


annotVar<-function(gr,genome='hg19'){

  ## create annotation file from EnsDb or TxDb
  if(genome=='hg19'){
    annoData <- toGRanges(EnsDb.Hsapiens.v75, feature="gene")
  }else if(genome=='hg38'){
    annoData <- toGRanges(EnsDb.Hsapiens.v86, feature="gene")
  }
  
  gr.anno <- annotatePeakInBatch(gr, AnnotationData=annoData,
                                       output="both")
  gr.anno$gene_name <-
    annoData$gene_name[match(gr.anno$feature,
                             names(annoData))]
  #gr.anno=unique(gr.anno)
  
  names(gr.anno)=NULL
  
  gr.anno
  
}


# https://support.bioconductor.org/p/133105/#133121


annotSNP<-function(gr.anno,genome='hg19'){
  
  seqlevelsStyle(gr.anno) <- "NCBI"

  if(genome=='hg19'){
    snpdb=SNPlocs.Hsapiens.dbSNP144.GRCh37
  }else if(genome=='hg38'){
    snpdb=SNPlocs.Hsapiens.dbSNP144.GRCh38
  }
  annot=snpsByOverlaps(snpdb, gr.anno)
  id=match(annot,gr.anno)
  rsids=numeric(length(gr.anno))
  rsids[id]=annot$RefSNP_id
  rsids[-id]='NA'
  gr.anno$rsid=rsids
  seqlevelsStyle(gr.anno) <- "UCSC"
  gr.anno
  
}


annotSNP2<-function(rsid,genome='hg19'){
  if(genome=='hg19'){
    snpdb=SNPlocs.Hsapiens.dbSNP144.GRCh37
    genome.db=BSgenome.Hsapiens.UCSC.hg19
  }else if(genome=='hg38'){
    snpdb=SNPlocs.Hsapiens.dbSNP144.GRCh38
    genome.db=BSgenome.Hsapiens.UCSC.hg38
  }
  
  gpos=snpsById(snpdb, rsid,ifnotfound="drop")
  
  seqlevelsStyle(gpos) <- "UCSC"
  
  z=inferRefAndAltAlleles(gpos, genome.db)
  z=z[,c(2,3)]
  colnames(z)=c('ref','alt')
  mcols(gpos) <- cbind(mcols(gpos), z)
  gpos$alleles_as_ambig=NULL
  
  gpos$alt=paste(gpos$alt,collapse = ',')
  
  
  gpos
}



getMeta<-function(gr.anno){
  tmp=as.data.frame(elementMetadata(gr.anno))
  
  tmp=data.table(tmp)
  
  tmp=tmp[,-c('peak','feature','shortestDistance','fromOverlappingOrNearest')]
  
  names.ch=c('start_position','end_position','feature_strand','insideFeature','distancetoFeature')
  
  id.ch=match(names.ch,colnames(tmp))
  
  colnames(tmp)[id.ch]=c('gene_start','gene_end','gene_strand','insideGene','distancetoGene')
  
  tmp
  
}




getGene<-function(genome){
  
  if(genome=='hg19'){
    gene=genes(TxDb.Hsapiens.UCSC.hg19.knownGene)
  }else if(genome=='hg38'){
    gene=genes(TxDb.Hsapiens.UCSC.hg38.knownGene)
  }
  gene.anno=select(org.Hs.eg.db,keys=gene$gene_id,keytype='ENTREZID',columns=c('SYMBOL','GENENAME','ENTREZID'))
   
  id=match(gene$gene_id,gene.anno$ENTREZID)
  
  mcols(gene)=cbind(mcols(gene),(gene.anno)[id,])
  
  gene
  
}


# gene.hg19=getGene(genome='hg19')
# 
# gene.hg38=getGene(genome='hg38')
# 
# save(gene.hg19,gene.hg38,file='gene.ref.rda')












