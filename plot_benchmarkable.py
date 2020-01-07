import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from benchmarkable import Benchmarkable
import scipy.integrate as it
import numpy as np
import umap


class PlotBenchmarkable:

    def __init__(self, models: list):
        self.benchmarkables = models
        self.COLORS = sns.color_palette(n_colors=20)
        self.MAX_KNN = 1000

    def compute_metrics(self):
        for b in self.benchmarkables:
            b.benchmark()

    @staticmethod
    def smooth(y, w=7):
        return pd.Series(y).rolling(w, min_periods=1).mean().values

    def plot_all(self):
        plt.subplot(2, 3, 1)
        self.plot_knn_purity()
        plt.subplot(2, 3, 2)
        self.plot_entropy_batch_mixing()
        plt.subplot(2, 3, 3)
        self.plot_density('spearman_per_gene')
        plt.subplot(2, 3, 4)
        self.plot_density('spearman_per_cell')
        plt.subplot(2, 3, 5)
        self.plot_density('median_absolute_error_per_gene')
        plt.subplot(2, 3, 6)
        self.plot_density('median_absolute_error_per_cell')

    def plot(self):
        plt.subplot(2, 3, 1)
        self.plot_knn_purity('seq')
        plt.subplot(2, 3, 2)
        self.plot_knn_purity('fish')
        plt.subplot(2, 3, 3)
        self.plot_entropy_batch_mixing()
        plt.subplot(2, 3, 4)
        self.plot_density('spearman_per_gene')
        plt.subplot(2, 3, 5)
        self.plot_time()
        plt.subplot(2, 3, 6)
        self.plot_summary()

    def plot_knn_purity(self, dataset, max_k=None, ratio=None):
        for i, b in enumerate(self.benchmarkables):
            if b.knn_purity is None:
                continue

            if max_k is None:
                max_k = self.MAX_KNN
                if dataset == "fish":
                    max_k = 500

            x = b.knn_purity[dataset]['k'][50:max_k]
            if ratio is not None:
                x = x.astype(float) * 100 / ratio
            y = b.knn_purity[dataset][0.5][50:max_k]

            y_smooth = PlotBenchmarkable.smooth(y)
            if 'kappa' in b.name:
                plt.plot(x, y_smooth, c=self.COLORS[i], label=b.name.replace('kappa', '$\\kappa=$'))
            else:
                plt.plot(x, y_smooth, c=self.COLORS[i], label=b.name)
            # plt.fill_between(x, b.knn_purity[dataset][0.2][50:max_k], b.knn_purity[dataset][0.8][50:max_k], color=self.COLORS[i], alpha=0.2)
            # plt.plot(x,, linestyle=':', c=self.COLORS[i])
            # plt.plot(x, b.knn_purity[dataset][0.2][50:], linestyle=':', c=self.COLORS[i])

        plt.title('k-nearest neighbors purity')  # - %s'%dataset)
        plt.xlabel('k' + (' (%% total cells= %d)' % ratio if ratio else ''))
        plt.ylabel('Jaccard index')

        # if dataset == "seq":
        plt.legend(prop={'size': 8})

    def plot_entropy_batch_mixing(self, max_k=None, ratio=None):
        for i, b in enumerate(self.benchmarkables):
            if b.entropy_batch_mixing is None:
                continue

            if max_k is None:
                max_k = self.MAX_KNN
            x = b.entropy_batch_mixing['k'][50:max_k]
            if ratio is not None:
                x = x.astype(float) * 100 / ratio
            y = 1 - b.entropy_batch_mixing[0.5][50:max_k]
            y_smooth = PlotBenchmarkable.smooth(y)

            if 'kappa = 0' in b.name:
                plt.plot(x, y_smooth, c=self.COLORS[i - 1], label=b.name, linestyle=':')
            else:
                plt.plot(x, y_smooth, c=self.COLORS[i], label=b.name)

            # plt.fill_between(x, 1-b.entropy_batch_mixing[0.8][50:self.MAX_KNN], 1-b.entropy_batch_mixing[0.2][50:self.MAX_KNN], color=self.COLORS[i], alpha=0.2)
            # plt.plot(x, , linestyle=':', c=self.COLORS[i])
            # plt.plot(x, ], linestyle=':', c=self.COLORS[i])

        plt.title('Entropy of mixing')
        plt.xlabel('k' + (' (%% total cells= %d)' % ratio if ratio else ''))
        plt.ylabel('Normalized negative KL')
        plt.legend()

    def plot_density(self, field):
        for b in self.benchmarkables:
            if b.imputation and field not in b.imputation is None:
                continue
            sns.distplot(b.imputation[field], label=b.name, hist=False)
        plt.title(field)
        plt.legend()

    def plot_time(self):
        df = pd.Series([b.train_time for b in self.benchmarkables], index=[b.name for b in self.benchmarkables])
        try:
            df.plot('bar', rot=0, alpha=0.8)
        except:
            print('Problem plotting training time, maybe data unavailable')
        plt.title('Training time')

    def compute_summary(self):
        cells = []
        col_titles = ['knn-purity', 'entropy bm', 'spearman']
        row_titles = []
        # should be added in metrics and benchmarkable.benchmark
        for b in self.benchmarkables:
            row = []
            # knn-purity
            try:
                x = b.knn_purity['fish']['k'][50:500]
                y = b.knn_purity['fish'][0.5][50:500]
                score = it.trapz(y=y, x=x) / (max(x) - min(x))

                # knn purity
                x = b.knn_purity['seq']['k'][50:self.MAX_KNN]
                y = b.knn_purity['seq'][0.5][50:self.MAX_KNN]
                score += it.trapz(y=y, x=x) / (max(x) - min(x))
                score /= 2
                row.append("%.4f" % score)
            except:
                row.append("NA")

            # entropy of bm
            try:
                x = b.entropy_batch_mixing['k'][50:self.MAX_KNN]
                y = 1 - b.entropy_batch_mixing[0.5][50:self.MAX_KNN]
                score = it.trapz(y=y, x=x) / (max(x) - min(x))
                row.append("%.4f" % score)
            except:
                row.append("NA")

            # spearman
            score = b.imputation['median_spearman_per_gene']
            row.append("%.4f" % score)
            row_titles.append(b.name)

            cells.append(row)

        return np.array(cells), row_titles, col_titles

    def evolution_summary(self):
        cells, row_titles, col_titles = tmp.compute_summary()
        cells = cells.astype(float)
        summary = pd.DataFrame(cells, columns=col_titles, index=row_titles)
        ax = summary.plot()
        ax.set_xticks(range(len(summary.index)))
        ax.set_xticklabels(summary.index, rotation=45)

    def plot_summary(self):
        cells, row_titles, col_titles = self.compute_summary()
        colors = np.zeros(cells.shape + (4,), )
        # colors.fill('w')

        cm = plt.cm.Reds(np.linspace(0, 0.5, cells.shape[0]), alpha=0.4)
        best = np.argsort(cells, axis=0)
        for j in range(cells.shape[1]):
            for i in range(cells.shape[0]):
                colors[best[i, j], j] = cm[i]
        # best = np.argmax(cells, axis=0)
        # for i in range(colors.shape[1]):
        #    colors[best[i],i]='r'
        plt.table(cellText=cells, cellColours=colors,
                  rowLabels=row_titles,
                  # rowColours=colors,
                  colLabels=col_titles,
                  loc='center')
        plt.axis('off')

    @staticmethod
    def single_plot_umap(benchmarkable):
        latent_seq, latent_fish = benchmarkable.latent_both_seq, benchmarkable.latent_both_fish
        latent2d = umap.UMAP().fit_transform(np.concatenate([latent_seq, latent_fish]))
        latent2d_seq = latent2d[: latent_seq.shape[0]]
        latent2d_fish = latent2d[latent_seq.shape[0]:]

        data_seq, data_fish = benchmarkable.data.data_seq, benchmarkable.data.data_fish

        colors = sns.color_palette(n_colors=30)
        plt.figure(figsize=(25, 10))
        ax = plt.subplot(1, 3, 1)
        ax.scatter(*latent2d_seq.T, color="r", label="seq", alpha=0.5, s=0.5)
        ax.scatter(*latent2d_fish.T, color="b", label="osm", alpha=0.5, s=0.5)
        ax.legend()

        ax = plt.subplot(1, 3, 2)
        labels = data_seq.labels.ravel()
        for i, label in enumerate(data_seq.cell_types):
            ax.scatter(
                *latent2d_seq[labels == i].T,
                color=colors[i],
                label=label[:12],
                alpha=0.5,
                s=5
            )
        ax.legend()
        ax.set_title("Seq cells")

        ax = plt.subplot(1, 3, 3)
        labels = data_fish.labels.ravel()
        for i, label in enumerate(data_fish.cell_types):
            ax.scatter(
                *latent2d_fish[labels == i].T, color=colors[i], label=label, alpha=0.5, s=5
            )
        ax.legend()
        ax.set_title("Spatial cells")
