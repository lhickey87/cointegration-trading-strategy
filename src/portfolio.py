import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#need to define the universe and dates
# the actual portfolio might have two different dataframes
# one which shows the returns of our tickers
# second -> one that is actually updating our portfolio dataframe
# portfolio dataframe should hold:
    # capital
    # nominal
    # leverage
    # gain_total
    # gain_daily
    # sharpe_annualized
    # vol -> (some window)

#should be okay to assume
# if we pass in strategy object -> we will get all the methods needed
class Portfolio():

    def __init__(self, strategies, start, end, capital):
        self.strategies = strategies
        self.dates = pd.to_datetime(start,end)

        self.strategy_data   = {s.name: s.data for s in strategies}
        self.strategy_stats  = pd.DataFrame(None, columns=["rolling_sharpe", "ret_20", "ret_60", "vol_20", "vol_60"])
        self.cov_mat = np.zeros((len(strategies),len(strategies)))
        self.start_capital = capital
    
    def initialize_portfolio(self,dates: pd.Series):
        portfolio = pd.DataFrame(index = dates).reset_index().rename(columns={"index":"datetime"})
        portfolio.at[0,"capital"] = self.start_capital
        portfolio.at[0,"day_pnl"] = 0
        portfolio.at[0,"capital_ret"] = 0
        portfolio.at[0,"nominal_ret"] = 0
        return portfolio
    
    #portfolio -> each day we will be updating based on signals
    def get_weights(self):
        inv = np.linalg.inv(self.cov_mat)
        means = self.strategy_stats['ret_60']
        wgt = inv @ means
        wgt = wgt / np.abs(wgt).sum(1)
        return wgt
    
    def update(self):
        weights = self.get_weights()
        # self.cov_mat = 
        #now these weights dictate how much we actually put into each strategy
        #so now we allow each strategy to get the weight
    
        