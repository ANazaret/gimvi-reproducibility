import pickle
from seuratV3 import SEURAT as SEURATV3
from liger import LIGER
from sklearn.decomposition import PCA, NMF

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from metrics import Metrics
from scipy.stats import spearmanr

import numpy as np
import copy
from collections import namedtuple
import time
import abc

import sys
#sys.path.append("../../")
#sys.path.append("../scVI/")
from scvi.dataset import GeneExpressionDataset
from scvi.models import JVAE, Classifier, VAE
from scvi.inference import JVAETrainer, UnsupervisedTrainer

Data = namedtuple('Data', ['data_seq', 'data_fish', 'data_fish_partial', 'train_indices', 'test_indices'])

class Benchmarkable(abc.ABC):
    def __init__(self, data: Data, name):
        self.data = data
        self.name = name
        
        # Computed variables
        self.latent_both_seq = None
        self.latent_both_fish = None
        self.latent_only_seq = None
        self.latent_only_fish = None
        self.imputed_full = None
        self.imputed = None
        
        # Benchmark attributes
        self.knn_purity = None
        self.entropy_batch_mixing = None
        self.imputation = None
        
        # Train time of the joint model
        self.train_time = None

    @staticmethod
    def load(filename, save_path='dumps/'):
        with open(save_path + filename, "rb") as f:
            return pickle.load(f)

    def save(self, save_path='dumps/'):
        with open(save_path + self.name, "wb") as f:
            pickle.dump(self, f)

    @abc.abstractmethod
    def train_seq(self):
        pass

    @abc.abstractmethod
    def train_fish(self):
        pass

    @abc.abstractmethod
    def train_both(self):
        pass

    def train(self):
        self.train_seq()
        self.train_fish()
        starting_time = time.time()
        self.train_both()
        self.train_time = time.time() - starting_time

    @abc.abstractmethod
    def compute_latent(self):
        # Compute latent spaces: Return latent_both_fish, latent_both_seq, latent_only_fish, latent_only_seq
        pass
    
    @abc.abstractmethod
    def compute_imputed_values(self):
        # Compute imputed values for the missing genes
        pass
    
    def compute(self):
        self.compute_latent()
        self.compute_imputed_values()
    
    def benchmark(self):
        self.benchmark_knn_purity()
        self.benchmark_entropy_batch_mixing()
        self.benchmark_imputation()    

    def benchmark_knn_purity(self):
        self.knn_purity = Metrics.knn_purity(
            self.latent_both_seq,
            self.latent_both_fish,
            self.latent_only_seq,
            self.latent_only_fish,
        )
        return self.knn_purity

    def benchmark_entropy_batch_mixing(self):
        latent = np.concatenate((self.latent_both_fish, self.latent_both_seq))
        batch_indices = np.concatenate(
            (
                np.zeros(self.latent_both_fish.shape[0]),
                np.ones(self.latent_both_seq.shape[0]),
            )
        )
        self.entropy_batch_mixing = Metrics.entropy_batch_mixing(latent, batch_indices)
        return self.entropy_batch_mixing

    
    def benchmark_imputation(self):
        imputed = self.imputed
        reality = self.data.data_fish.X[:,self.data.test_indices]
        self.imputation = Metrics.imputation_metrics(reality, imputed)
        
        
        
        
class SeuratV3(Benchmarkable):
    def __init__(self, data, name, n_latent=8):
        super().__init__(data, name)
        self.r_bridge = SEURATV3("")
        self.n_latent = n_latent
        self.r_bridge.create_seurat(self.data.data_fish_partial, self.data.data_seq)

    def train_both(self):
        latent_both, _, _, _ = self.r_bridge.get_cca(n_latent=self.n_latent)
        self.latent_both = np.array(latent_both)

    def train_fish(self):
        dataset = self.data.data_fish
        pca = PCA(n_components=self.n_latent)
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        normalized_matrix = np.log(1 + 1e4 * normalized_matrix)
        self.latent_only_fish = pca.fit_transform(normalized_matrix)
        self.model_fish = pca

    def train_seq(self):
        dataset = self.data.data_seq
        pca = PCA(n_components=self.n_latent)
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        normalized_matrix = np.log(1 + 1e4 * normalized_matrix)
        self.latent_only_seq = pca.fit_transform(normalized_matrix)
        self.model_seq = pca

    def compute_latent(self):
        """ Return latent_both_fish, latent_both_seq, latent_only_fish, latent_only_seq
        """
        self.latent_both_fish = self.latent_both[
            : self.data.data_fish_partial.X.shape[0], :
        ]
        self.latent_both_seq = self.latent_both[
            self.data.data_fish_partial.X.shape[0] :, :
        ]

        return (
            self.latent_both_fish,
            self.latent_both_seq,
            self.latent_only_fish,
            self.latent_only_seq,
        )

    def compute_imputed_values_old(self, k=10):
        dataset = self.data.data_seq
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        knn = KNeighborsRegressor(k, weights="distance")
        predicted = knn.fit(
            self.latent_both_seq, normalized_matrix
        ).predict(self.latent_both_fish)
        self.imputed_full = predicted * self.data.data_fish_partial.X.sum(axis=1).reshape(-1, 1)
        self.imputed = self.imputed_full[:, self.data.test_indices]
        return self.imputed
    
    def compute_imputed_values(self):
        res = self.r_bridge.impute()
        self.imputed_full = res.todense()        
        self.imputed = self.imputed_full[:, self.data.test_indices]
        return self.imputed
        
        
class Liger(Benchmarkable):
    def __init__(self, data, name, n_latent=20):
        super().__init__(data, name)
        self.r_bridge = LIGER()
        self.n_latent = n_latent 

    def train_both(self):
        self.r_bridge.create_liger(self.data.data_fish_partial, self.data.data_seq, 'spatial', 'seq')
        clusters_liger, latent_liger, _, _ = self.r_bridge.run_factorization(k=self.n_latent)
        self.latent_both = latent_liger
        
    def train_fish(self):
        dataset = self.data.data_fish
        nmf = NMF(n_components=self.n_latent)
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        normalized_matrix = np.log(1 + 1e4 * normalized_matrix)
        self.latent_only_fish = nmf.fit_transform(normalized_matrix)
        self.model_fish = nmf

    def train_seq(self):
        dataset = self.data.data_seq
        nmf = NMF(n_components=self.n_latent)
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        normalized_matrix = np.log(1 + 1e4 * normalized_matrix)
        self.latent_only_seq = nmf.fit_transform(normalized_matrix)
        self.model_seq = nmf

    def compute_latent(self):
        """ Return latent_both_fish, latent_both_seq, latent_only_fish, latent_only_seq
        """
        self.latent_both_fish = self.latent_both[
            : self.data.data_fish_partial.X.shape[0], :
        ]
        self.latent_both_seq = self.latent_both[
            self.data.data_fish_partial.X.shape[0] :, :
        ]

        return (
            self.latent_both_fish,
            self.latent_both_seq,
            self.latent_only_fish,
            self.latent_only_seq,
        )

    def compute_imputed_values(self, k=10):
        dataset = self.data.data_seq
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        knn = KNeighborsRegressor(k, weights="distance")
        predicted = knn.fit(
            self.latent_both_seq, normalized_matrix
        ).predict(self.latent_both_fish)
        self.imputed_full = predicted * self.data.data_fish_partial.X.sum(axis=1).reshape(-1, 1)
        self.imputed = self.imputed_full[:, self.data.test_indices]
        return self.imputed
        
        
import sklearn.neighbors
import scipy

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, n_neighbors=10):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        self.clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
        self.clf.fit(Xs_new, Ys)
        self.y_pred = self.clf.predict(Xt)
        # acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return self.y_pred



    
    
class Coral(Benchmarkable):
    def __init__(self, data, name, n_neighbors=10):
        super().__init__(data, name)
        self.n_neighbors = n_neighbors
        self.coral = CORAL()
        
    def train_both(self):
        Xs = self.data.data_seq.X[:, self.data.train_indices]
        Ys = self.data.data_seq.X[:, self.data.test_indices]

        Xt = self.data.data_fish.X[:, self.data.train_indices]
        Yt = self.data.data_fish.X[:, self.data.test_indices]
        
        self.imputed = self.coral.fit_predict(Xs, Ys, Xt, self.n_neighbors)
        
        

    def train_fish(self):
        pass

    def train_seq(self):
        pass

    def compute_latent(self):
        pass
    
    def compute_imputed_values(self, k=10):
        pass
        
        
        
        
class Base_scVI(Benchmarkable):
    def __init__(self, data, name, n_latent=10):
        super().__init__(data, name)
        self.n_latent = n_latent
        self.USE_CUDA = False
        
    
    def train(self, n_epochs=20):
        self.train_seq(n_epochs)
        self.train_fish(n_epochs)
        starting_time = time.time()
        self.train_both(n_epochs)
        self.train_time = time.time() - starting_time


    def train_fish(self, n_epochs=20):
        dataset = self.data.data_fish
        vae = VAE(
            dataset.nb_genes,
            n_batch=dataset.n_batches,
            dispersion="gene-batch",
            n_latent=self.n_latent,
            reconstruction_loss="nb",
        )
        self.trainer_fish = UnsupervisedTrainer(
            vae, dataset, train_size=0.95, use_cuda=self.USE_CUDA
        )
        self.trainer_fish.train(n_epochs=n_epochs, lr=0.001)

    def train_seq(self, n_epochs=20, reconstruction_seq='nb'):
        dataset = self.data.data_seq
        vae = VAE(
            dataset.nb_genes,
            dispersion="gene",
            n_latent=self.n_latent,
            reconstruction_loss=reconstruction_seq,
        )
        self.trainer_seq = UnsupervisedTrainer(
            vae, dataset, train_size=0.95, use_cuda=self.USE_CUDA
        )
        self.trainer_seq.train(n_epochs=n_epochs, lr=0.001)


class scVI(Base_scVI):
    def __init__(self, data, name, n_latent=10, reconstruction_seq='zinb'):
        super().__init__(data, name, n_latent)
        
        self.full_dataset = GeneExpressionDataset()

        self.full_dataset.populate_from_datasets([copy.deepcopy(data.data_fish_partial), copy.deepcopy(data.data_seq)])
        self.full_dataset.compute_library_size_batch()
        self.reconstruction_seq = reconstruction_seq
        
    def train_both(self, n_epochs=20):
        vae_both = VAE(
            self.full_dataset.nb_genes,
            n_latent=self.n_latent,
            n_batch=self.full_dataset.n_batches,
            dispersion="gene-batch",
            reconstruction_loss=self.reconstruction_seq,
        )
        self.trainer_both = UnsupervisedTrainer(
            vae_both,
            self.full_dataset,
            train_size=0.95,
            use_cuda=self.USE_CUDA,
            frequency=1,
        )
        self.trainer_both.train(n_epochs=n_epochs, lr=0.001)
        # self.posterior_both = self.trainer_both.create_posterior()
    
    def compute_latent(self):
        """ Return latent_both_fish, latent_both_seq, latent_only_fish, latent_only_seq
        """

        both = self.trainer_both.create_posterior().get_latent()[0]
        self.latent_both = both

        self.latent_both_fish = self.latent_both[
            : self.data.data_fish_partial.X.shape[0], :
        ]
        self.latent_both_seq = self.latent_both[
            self.data.data_fish_partial.X.shape[0] :, :
        ]

        fish = self.trainer_fish.create_posterior().get_latent()[0]
        self.latent_only_fish = fish

        seq = self.trainer_seq.create_posterior().get_latent()[0]
        self.latent_only_seq = seq

        return (
            self.latent_both_fish,
            self.latent_both_seq,
            self.latent_only_fish,
            self.latent_only_seq,
        )    
    
    def compute_imputed_values(self, k=10):
        dataset = self.data.data_seq
        normalized_matrix = dataset.X / np.sum(dataset.X, axis=1)[:, np.newaxis]
        knn = KNeighborsRegressor(k, weights="distance")
        predicted = knn.fit(
            self.latent_both_seq, normalized_matrix
        ).predict(self.latent_both_fish)
        self.imputed_full = predicted * self.data.data_fish_partial.X.sum(axis=1).reshape(-1, 1)
        self.imputed = self.imputed_full[:, self.data.test_indices]
        return self.imputed
