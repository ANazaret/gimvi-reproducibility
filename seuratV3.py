import numpy as np
import rpy2.robjects as ro
import warnings
from rpy2.rinterface import RRuntimeWarning
import rpy2.robjects.numpy2ri as numpy2ri
from scipy.io import mmwrite, mmread
from scipy.sparse import coo_matrix, csr_matrix


class SEURAT:
    def __init__(self, dataname):
        self.dataname = dataname
        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        numpy2ri.activate()
        r_source = ro.r['source']
        r_source("Seurat.functionsV3.R")
        ro.r["library"]("Matrix")
        ro.r["library"]("RcppCNPy")
        ro.r["library"]("reticulate")

    def create_seurat(self, fish, seq):
        self._create_seurat(fish, 1)
        self._create_seurat(seq, 2)

    def _create_seurat(self, dataset, batchname):
        genenames = dataset.gene_names
        genenames, uniq = np.unique(genenames, return_index=True)
        labels = []
        matrix = coo_matrix(dataset.X[:, uniq]).tocsr()
        mmwrite('temp.mtx', matrix.T)
        ro.r('X <- readMM("temp.mtx")')
        ro.r.assign("batchname", batchname)
        ro.r.assign("genenames", ro.StrVector(genenames))
        ro.r.assign("labels", ro.StrVector(labels))
        ro.r('seurat' + str(batchname) + '<- SeuratPreproc(X,labels,batchname,genenames)')
        return 1

    def get_pcs(self):
        ro.r('seurat1 <- RunPCA(seurat1, pc.genes = seurat1@var.genes, do.print = FALSE)')
        ro.r('seurat2 <- RunPCA(seurat2, pc.genes = seurat2@var.genes, do.print = FALSE)')
        pc1 = ro.r('GetDimReduction(object = seurat1, reduction.type = "pca", slot = "cell.embeddings")[,c(1:10)]')
        pc2 = ro.r('GetDimReduction(object = seurat2, reduction.type = "pca", slot = "cell.embeddings")[,c(1:10)]')
        return pc1, pc2

    def get_cca(self, filter_genes=True, n_latent=8):
        if filter_genes == True:
            ro.r('combined <- hvg_CCA(list(seurat1,seurat2), filter_genes = TRUE, ndim=%d)' % n_latent)
        else:
            ro.r('combined <- hvg_CCA(list(seurat1,seurat2), ndim=%d)' % n_latent)
        latent = ro.r('combined[[1]]')
        genes = ro.r('combined[[2]]')
        batch_indices = ro.r('combined[[3]]')
        cells = ro.r('combined[[4]]')
        return latent, batch_indices, genes, cells

    def impute(self):
        ro.r('imputed <- impute(seurat2, seurat1)')
        ro.r('writeMM(imputed, "tmp.mtx")')
        res = mmread('tmp.mtx')
        return csr_matrix(res.T)
