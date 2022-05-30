import pandas as pd
import numpy as np

def load_data(config):
    asset_prices = pd.read_csv(config['assets_path'], index_col=0)
    asset_prices.index = pd.to_datetime(asset_prices.index)
    asset_returns = asset_prices.pct_change(config['returns_timeframe'])

    if config['use_features']:
        asset_returns = asset_returns.shift(-1*config['returns_timeframe'])
        features = pd.read_csv(config['features_path'], index_col=0)
        features.index = pd.to_datetime(features.index)
    else:
        features = None
    asset_returns = asset_returns.dropna()

    # dates where we rebalance in backtest
    rebalance_dates = asset_prices.resample('Y').last().index[config['lookback_years']+1:-1]
    return asset_returns, features, rebalance_dates

def zscore_euclidean(spot_feature, sampled_features):
    mu = sampled_features.mean()
    sigma = sampled_features.std()
    spot_zscore = (spot_feature - mu)/sigma
    zscore = (sampled_features - mu)/sigma
    distance = np.sqrt(((zscore - spot_zscore)**2).sum(axis=1))
    return distance

def save_file(matrix, path):
    np.savetxt(path, matrix, delimiter=",")
