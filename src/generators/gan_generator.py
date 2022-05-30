from normalizer import Normalizer
import numpy as np
import pandas as pd
import warnings
import datetime as dt

from sdv.tabular import CTGAN
from sklearn.manifold import TSNE
import hdbscan

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


class CTGANGenerator():

    def __init__(self, asset_returns, params=None, features=None):
        self.asset_returns = asset_returns
        self.features = features
        self.name = 'CTGAN'
        self.params = params if params else {'embedding_dim': 128,
                           'generator_dim': (256, 256),
                           'discriminator_dim': (256, 256),
                           'epochs': 1500,
                           'generator_lr': 1e-4,
                           'discriminator_lr': 1e-4}


    def generate_sample(self, sample_size, start_date, end_date, pca_construct=True):

        model = CTGAN(**self.params)
        returns_interval = self.asset_returns.loc[
            (self.asset_returns.index <= end_date) & (self.asset_returns.index >= start_date)]
        fit_cols = list(self.asset_returns.columns) + ['cluster']
        normalizer = None
        
        if self.features:
            returns_interval = returns_interval.join(self.features, how='left').fillna(method='ffill')
            normalizer = Normalizer()
            returns_interval = normalizer.normalize(returns_interval)
            fit_cols = list(self.asset_returns.columns) + list(self.features.columns) + ['cluster']


        # Applies PCA
        if pca_construct:
            pca, returns_interval = self._construct_pca(returns_interval)
            fit_cols = [f"C_{i}" for i in range(pca.n_components_)] + ['cluster']


        # Dimensionality reduction
        returns_interval, X_embedded = self._reduce_dim(returns_interval)

        # Clusters definition
        returns_interval = self._define_clusters(returns_interval, X_embedded)

        # Fits CTGAN using categorical variable of state       
        model.fit(returns_interval[fit_cols])

        sample = model.sample(sample_size)[fit_cols[:-1]]
        sample_val = sample.values

       # Reconstruct assets
        if pca_construct:
            sample_val = pca.inverse_transform(sample_val)
        
        # De-normalizes
        if self.features:
            sample_val = normalizer.denormalize(sample_val)


        return sample_val
    
    def _construct_pca(self, returns_interval):
        pca = PCA(n_components=returns_interval.shape[1])
        pca.fit(returns_interval)
        asset_returns_interval_trans = pca.transform(returns_interval)
        pca_cols = [f"C_{i}" for i in range(pca.n_components_)]
        return pca, pd.DataFrame(asset_returns_interval_trans,
                                        index=returns_interval.index,
                                        columns=pca_cols)

    def _reduce_dim(returns_interval, dims=2):
        X_embedded = TSNE(n_components=dims, learning_rate='auto', init='pca').fit_transform(returns_interval)
        returns_interval['x'] = X_embedded[:, 0]
        returns_interval['y'] = X_embedded[:, 1]
        return returns_interval, X_embedded

    def _define_clusters(returns_interval, X_embedded):
        cluster_dim = max(10, int(len(returns_interval) * 0.005))
        clusterer = hdbscan.HDBSCAN(min_samples=cluster_dim, min_cluster_size=cluster_dim)
        clusterer.fit(X_embedded)
        returns_interval['cluster'] = ['c_' + str(c) for c in clusterer.labels_]

        return returns_interval