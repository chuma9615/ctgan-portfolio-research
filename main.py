# Standard library imports
import json

# Local application imports
from src.backtester import Backtester
from src.utils import load_data

config = json.load(open("./config.json"))
asset_prices, asset_returns, features, rebalance_dates = load_data(config)

backtester = Backtester(
                        asset_prices=asset_prices,
                        asset_returns=asset_returns,
                        config=config,
                        rebalance_dates=rebalance_dates,
                        features=features)

backtests = backtester.run_backtests()