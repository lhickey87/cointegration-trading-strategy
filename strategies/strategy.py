import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#strategy will be what is holding the tickers price data
#on top of that it should hold the performance of strategy currently
# capital, day_ret, capital_ret, sharpe
class Strategy:

    def __init__(self, name:str, price_data: pd.DataFrame):
        self.name = name
        self.price_data = price_data
        self.tickers = []
        self.fitted_params_ = {}
        self.portfolio = pd.DataFrame(None,columns = ["capital", "day_ret", "day_pnl", "sharpe"])
        self.ddivf = {}
        self.vols = {}
        self._current_data  = None
        self.capital = 0

    def fit(self, train_data, sector_map):
        """Prepare the strategy on training data — pair selection, parameter 
    estimation, threshold calibration. No ML required."""
        raise NotImplementedError
    
    def preprocess(self):
        raise NotImplementedError
    
    def evaluate(self, test_data):
        if not getattr(self, 'is_fitted', False):
            raise RuntimeError("Must call fit() before evaluate()")
        return self._evaluate(test_data)
    
    def initialize_positions(self, df: pd.DataFrame):
        """Class specific for position logic -> which tickers -> if pairs we can specify"""
        raise NotImplementedError

    def generate_signals(self, data):
        """Return signals given current bar. Called every bar during evaluate."""
        raise NotImplementedError

    def get_weights(self, signals, data):
        """Convert signals to position weights."""
        raise NotImplementedError

    def update(self, bar):
        """Online model updates every bar e.g. Kalman, DVIF."""
        #not sure how this would be a unified function
        raise NotImplementedError
    
    def update_capital(self, capital: np.float64):
        self.capital = capital
    
    def update_positions(self, date, signals, weights):
        """
        NEED TO ENSURE that weights and signals are lined up on tickers
        """
        if signals.columns != weights.columns:
            return
        self.positions.loc[date] = (signals*weights).values

    def evaluate(self, test_data):
        return self._evaluate(test_data)

    @property
    def get_capital(self):
        return self.capital