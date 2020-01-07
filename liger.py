import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import warnings
from rpy2.rinterface import RRuntimeWarning


class LIGER:

    def __init__(self):

        warnings.filterwarnings("ignore", category=RRuntimeWarning)
        rpy2.robjects.numpy2ri.activate()
        ro.r["library"]("Matrix")
        ro.r["library"]("matrixStats")
        ro.r["library"]("liger")

    def create_liger(self, dataset1, dataset2, name1, name2):
        # transferring all dataset info to the R matrix format
        #
        gene1 = [gene.lower() for gene in dataset1.gene_names]
        nc1, nr1 = dataset1.X.shape
        data1 = dataset1.X.T
        Rdata1 = ro.r.matrix(data1, nrow=nr1, ncol=nc1)
        Rdata1.rownames = ro.StrVector(gene1)
        Rdata1.colnames = ro.StrVector(np.arange(nc1).astype(str))

        gene2 = [gene.lower() for gene in dataset2.gene_names]
        nc2, nr2 = dataset2.X.shape
        data2 = dataset2.X.T
        Rdata2 = ro.r.matrix(data2, nrow=nr2, ncol=nc2)
        Rdata2.rownames = ro.StrVector(gene2)
        Rdata2.colnames = ro.StrVector(np.arange(nc1, nc1 + nc2).astype(str))
        ro.r.assign('size_1', nc1)

        ro.r.assign("name1", name1)
        ro.r.assign("name2", name2)
        ro.r.assign('Rdata1', Rdata1)
        ro.r.assign('Rdata2', Rdata2)
        ro.r('liger <- createLiger(list(name1 = Rdata1, name2 = Rdata2), remove.missing = F)')

    def run_factorization(self, k=20, l=5.0, suggest_k=False, suggest_l=False):
        ro.r('liger <- normalize(liger)')
        ro.r('liger <- selectGenes(liger, var.thresh = 0.001, do.plot = F)')
        ro.r('liger <- scaleNotCenter(liger, remove.missing = F)')
        ro.r.assign('k', k)
        ro.r.assign('l', l)
        if suggest_k:
            ro.r('k <- suggestK(liger, gen.new = T, return.results = T, plot.log2 = F)')
        if suggest_l:
            ro.r('l <- suggestLambda(liger, k, return.results = T)')
        print(k)
        ro.r('liger <- optimizeALS(liger, k=k, lambda=l, thresh = 5e-5, nrep = 3)')
        ro.r('liger <- quantileAlignSNF(liger)')
        clusters = np.array(ro.r('liger@clusters'))
        latent_factors = np.array(ro.r('liger@H.norm'))
        cells_idx_1 = np.array(ro.r('colnames(liger@norm.data[[1]])')).astype(int)
        cells_idx_2 = np.array(ro.r('colnames(liger@norm.data[[2]])')).astype(int) - ro.r('size_1')[0]
        return clusters, latent_factors, cells_idx_1, cells_idx_2

    def plot_tsne_clusters(self):
        ro.r('liger <- runTSNE(liger)')
        ro.r('p_a <- plotByDatasetAndCluster(liger, return.plots = T)')
        ro.r('print(p_a[[1]])')
        ro.r('print(p_a[[2]])')
