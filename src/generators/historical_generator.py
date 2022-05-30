import numpy as np
from ..utils import save_file
from normalizer import Normalizer

class HistoricalGenerator():
    """
    Generates a random sample, based on a historical dataset.
    """
    def __init__(self, asset_returns, features=None):
        self.features = features
        self.asset_returns = asset_returns
        self.name = 'historical'
    
    def generate_sample(self, sample_size, start_date, end_date, normalize_features=False):
        asset_returns_interval = self.asset_returns.loc[(self.asset_returns.index <= end_date)&(self.asset_returns.index >= start_date)]
        if self.features is not None:
            asset_returns_interval = asset_returns_interval.join(self.features, how='left').fillna(method='ffill')

        if normalize_features:
            normalizer = Normalizer()
            asset_returns_interval = normalizer.normalize(asset_returns_interval)
        else:
            normalizer = None

        total_windows = len(asset_returns_interval)
        sample = np.random.choice(total_windows, sample_size, replace=False)
        sample = asset_returns_interval.iloc[sample, :].reset_index(drop=True)

        if normalize_features:
            sample = normalizer.denormalize(sample)

        return sample.values
