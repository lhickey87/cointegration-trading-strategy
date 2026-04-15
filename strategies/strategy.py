import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#strategy will be what is holding the tickers price data
#on top of that it should hold the performance of strategy currently
# capital, day_ret, capital_ret, sharpe
class Strategy:

    def __init__(self, name:str):
        self.name = None
        self.tickers = []
        self.fitted_params_ = {}
        self.portfolio = pd.DataFrame(None,columns = ["capital", "day_ret", "day_pnl", "sharpe"])
        self.ddivf = {}
        self.vols = {}
        self._current_data  = None
        self.capital = 0

    def fit(self, train_data):
        """Prepare the strategy on training data — pair selection, parameter 
    estimation, threshold calibration. No ML required."""
        raise NotImplementedError
    
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

    def _fit(self, train_data):
        self.fit(train_data)
        self.is_fitted_ = True

    def evaluate(self, test_data):
        return self._evaluate(test_data)
        return

    @property
    def get_capital(self):
        return self.capital