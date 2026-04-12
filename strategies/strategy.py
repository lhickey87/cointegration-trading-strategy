import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Strategy:

    def __init__(self):
        self.name           = None

        self.tickers         = []
        self.fitted_params_ = {}
        
        self.ddivf = {}
        self.vols = {}
        # injected by Backtest
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
        pass
    
    def update_capital(self, capital: np.float):
        self.capital = capital

    def on_trade_entry(self, pair, signal):
        """Called when a position is opened."""
        #on pair entry -> where do we actually execute the trade
        pass

    def on_trade_exit(self, pair, signal):
        """Called when a position is closed."""
        pass

    def _fit(self, train_data):
        self.fit(train_data)
        self.is_fitted_ = True

    def evaluate(self, test_data):
        return

    @property
    def get_capital(self):
        return self.capital