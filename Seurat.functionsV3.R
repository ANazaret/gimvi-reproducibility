library("Seurat")

hvg_CCA <- function(data,ndim=10,plotting=F,filter_genes = FALSE,ngenes=1000){
    if (filter_genes ==TRUE){
        genes.use <- c()
		for (i in 1:length(data)) {
		  genes.use <- c(genes.use, head(rownames(HVFInfo(object = data[[i]])), ngenes))
		}
		if(length(data)==2){n_shared=0}else{n_shared=2}
		genes.use <- names(which(table(genes.use) > n_shared))
		for (i in 1:length(data)) {
		  genes.use <- genes.use[genes.use %in% rownames(GetAssayData(object = data[[i]], slot = "scale.data"))]
		}
    }
    else{
        genes.use = rownames(data[[1]]@data)
    }
    
    for (i in 1:length(x = data)) {
    	data[[i]] <- NormalizeData(object = data[[i]], verbose = FALSE)
	    data[[i]] <- FindVariableFeatures(object = data[[i]], selection.method = "vst", 
        	nfeatures = 900, verbose = FALSE)
	}
    
    anchors <- FindIntegrationAnchors(object.list = data, dims = 1:ndim)
    integrated <- IntegrateData(anchorset = anchors, dims = 1:ndim)
    
    # combined <- RunCCA(object = data[[1]], object2 = data[[2]], genes.use = genes.use,num.cc = ndim)
	# combined <- CalcVarExpRatio(object = combined, reduction.type = "pca", grouping.var = "batch",
    # dims.use = 1:ndim)
	# combined <- AlignSubspace(object = combined, reduction.type = "cca", grouping.var = "batch",
	#    dims.align = 1:ndim)
    # latent <- GetDimReduction(object = combined,
    #    reduction.type = "cca.aligned",
    #    slot = "cell.embeddings"
    #)
    DefaultAssay(object = integrated) <- "integrated"
    integrated <- ScaleData(object = integrated)
    integrated <- RunPCA(object = integrated, npcs = ndim)

	latent <- integrated[["pca"]][[]]

    cells = do.call(c,lapply(data,function(X){colnames(X)}))
    batch <- sapply(strsplit(cells,'_'),function(X){X[1]})
    cells = sapply(strsplit(cells,'_'),function(X){X[2]})
    
    
    return(list(latent,genes.use,batch,cells))
}


impute <- function(data_seq, data_fish){
  data_seq <- NormalizeData(object = data_seq, verbose = FALSE)
  data_seq <- FindVariableFeatures(object = data_seq, selection.method = "vst", nfeatures = 900, verbose = FALSE)
  
  data_fish <- NormalizeData(object = data_fish, verbose = FALSE)
  data_fish <- FindVariableFeatures(object = data_fish, selection.method = "vst", nfeatures = 900, verbose = FALSE)
  

  anchors <- FindTransferAnchors(reference = data_seq, query = data_fish)
  to_predict <- GetAssayData(object = data_seq, slot = "counts")
  predictions <- TransferData(anchorset = anchors, refdata = to_predict)
  imputed <- GetAssayData(object = predictions, slot = "data")
  
  return(imputed)
}

SeuratPreproc <- function(X,label,batchname,zero_cells,genenames=NA){
	X = as.matrix(X)
	if(is.na(genenames[1])==T){
		genenames = paste('gene',c(1:length(X[,1])),sep='_')
	}
	rownames(X) = genenames
	colnames(X) = paste(batchname,c(1:length(X[1,])),sep='_')
	X <- CreateSeuratObject(counts = X,min.features = 0, min.cells = 0)
	X <- NormalizeData(object = X)
    X <- FindVariableFeatures(X, do.plot = F, display.progress = F)
	X <- ScaleData(object = X)
	X@meta.data[, "batch"] <- batchname
	X@meta.data[,"labels"] <- label
	return(X)
}
