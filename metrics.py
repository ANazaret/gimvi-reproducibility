import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def kl_bin(p, q):
    resp = np.log(p / q)
    resp[np.isinf(resp)] = 0
    resp *= p

    resq = np.log((1 - p) / (1 - q))
    resq[np.isinf(resq)] = 0
    resq *= (1 - p)
    res = resp + resq
    return res


def jaccards_unions(ab):
    a, b = ab
    # union = bitfield.Bitfield()
    union = set()
    union_size = []
    for k, (x, y) in enumerate(zip(a, b)):
        union.add(x)
        union.add(y)
        c = len(union)
        # union_size.append((2*k+2-c)/c)  # Jaccard
        union_size.append(2 - c / (k + 1))  # Accuracy
    return union_size


class Metrics:
    QUANTILES = [0.1, 0.2, 0.5, 0.8, 0.9]

    def __init__(self):
        pass

    @staticmethod
    def _knn_purity(latent_both, latent_only, n_samples=2000, n_neighbors=5000):
        nne_both = NearestNeighbors(metric="euclidean").fit(latent_both)
        nne_only = NearestNeighbors(metric="euclidean").fit(latent_only)
        max_neighbors = min(n_neighbors, len(latent_both))

        np.random.seed(0)
        indices = np.random.choice(np.arange(latent_both.shape[0]), size=n_samples, replace=False, )
        _, neigh_both = nne_both.kneighbors(latent_both[indices], n_neighbors=max_neighbors)
        _, neigh_only = nne_only.kneighbors(latent_only[indices], n_neighbors=max_neighbors)

        scores = np.array(list(map(jaccards_unions, zip(neigh_both, neigh_only))))

        res = dict()
        res['k'] = np.array(range(1, max_neighbors + 1))
        res['mean'] = scores.mean(axis=0)
        for q in Metrics.QUANTILES:
            res[q] = np.quantile(scores, q, axis=0)
        res['samples'] = indices
        return res

    @staticmethod
    def knn_purity(latent_both_seq, latent_both_fish, latent_only_seq, latent_only_fish, n_samples=2000,
                   n_neighbors=5000):
        seq = Metrics._knn_purity(latent_both_seq, latent_only_seq)
        fish = Metrics._knn_purity(latent_both_fish, latent_only_fish)

        res = dict()
        res['seq'] = seq
        res['fish'] = fish
        return res

    @staticmethod
    def entropy_batch_mixing(
            latent_both, batch_ids, n_samples=2000, min_neighbors=50, max_neighbors=5000, quantiles=False
    ):
        freq = batch_ids.mean()
        nne = NearestNeighbors(metric="euclidean")
        nne.fit(latent_both)

        np.random.seed(12)
        indices = np.random.choice(np.arange(latent_both.shape[0]), size=n_samples, replace=False, )
        _, res = nne.kneighbors(latent_both[indices], n_neighbors=max_neighbors)
        res_batch = batch_ids[res]
        prop = np.cumsum(res_batch, axis=1) / np.arange(1, max_neighbors + 1)

        kls = kl_bin(prop, freq)[:, min_neighbors:]

        res = dict()
        res['k'] = np.array(range(min_neighbors + 1, max_neighbors + 1))
        res['mean'] = kls.mean(axis=0)
        for q in Metrics.QUANTILES:
            res[q] = np.quantile(kls, q, axis=0)

        res['samples'] = indices
        return res

    @staticmethod
    def imputation_metrics(original, imputed):
        absolute_error = np.abs(original - imputed)
        relative_error = absolute_error / np.maximum(
            np.abs(original), np.ones_like(original)
        )
        spearman_gene = []
        for g in range(imputed.shape[1]):
            if np.all(imputed[:, g] == 0):
                correlation = 0
            else:
                correlation = spearmanr(original[:, g], imputed[:, g])[0]
            spearman_gene.append(correlation)

        return {
            "median_absolute_error_per_gene": np.median(absolute_error, axis=0),
            "mean_absolute_error_per_gene": np.mean(absolute_error, axis=0),
            "mean_relative_error": np.mean(relative_error, axis=1),
            "median_relative_error": np.median(relative_error, axis=1),
            "spearman_per_gene": np.array(spearman_gene),

            # Metric we report in the GimVI paper:
            "median_spearman_per_gene": np.median(spearman_gene),
        }
