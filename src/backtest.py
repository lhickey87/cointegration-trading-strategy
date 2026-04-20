
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from DDIVF import DDIVF
from HMM import fit_hmm, get_current_regime
from signals import get_signals, optimise_threshold

def get_years(index: pd.Series):
    return index.year.value_counts().keys()

def get_year(index: pd.Series):
    most_common_year = index.year.value_counts().idxmax()
    return most_common_year

class Backtest():

    def __init__(self, strategy):
        # self.portfolio = portfolio
        self.strategy = strategy
    
    def full_walk_forward(self, train_months: int = 24, test_months: int = 12, embargo: int = 10):
        folds = self.rolling_folds(train_months=train_months,
                                   test_months=test_months,
                                   embargo=embargo)
        return self._run_folds(folds)

    def walk_forward(self, train_months: int = 24, test_months: int = 12, embargo: int = 10):
        folds = self.make_folds(train_months, test_months, embargo_days=embargo)
        return self._run_folds(folds)
     
    def _run_folds(self, folds):
        pnl_by_fold = {}
        sharpes     = {}
        sharpes_by_pair = {}
        returns_by_pair = {}

        for fold in folds:
            train_data = self.strategy.preprocess(fold['train'])
            test_data  = self.strategy.preprocess(fold['test'])

            self.strategy.fit(train_data)

            year = get_year(index=test_data.index)
            print(f"Testing over: {year}")

            weights, signals = self.strategy.evaluate(test_data)

            pair_returns,daily_pnl, pair_sharpes, sharpe = self.strategy.backtest(test_data, 
                                                                                  weights=weights, 
                                                                                  signals=signals)

            print(f"sharpe for year: {year} is: {sharpe}")

            pnl_by_fold[year] = daily_pnl
            sharpes[year]     = sharpe
            sharpes_by_pair[year] = pair_sharpes
            returns_by_pair[year] = pair_returns

        sharpe_s = pd.Series(sharpes, name='sharpe')
        pnl_s    = pd.concat(pnl_by_fold)
        p_sharpes = pd.DataFrame(sharpes_by_pair)
        return returns_by_pair,sharpe_s, pnl_s,p_sharpes
    
    def rolling_folds(self, train_months: int = 24, test_months: int = 12, embargo: int = 10):
        """Rolling window — train window slides forward by test_months each fold.
        Unlike make_folds (disjoint), the train windows overlap so every fold
        sees a fixed-length training period."""
        folds      = []
        data       = self.strategy.price_data
        end        = data.index[-1]
        fold_start = data.index[0]

        while True:
            train_end  = fold_start + pd.DateOffset(months=train_months)
            test_start = train_end  + pd.DateOffset(days=embargo)
            test_end   = test_start + pd.DateOffset(months=test_months)

            if test_end > end:
                break

            folds.append({
                "train": data.loc[fold_start:train_end],
                "test":  data.loc[test_start:test_end]
            })

            fold_start = fold_start + pd.DateOffset(months=test_months)

        return folds

    def make_folds(self, train_months: int = 24, test_months: int = 12, embargo_days: int = 10):
        folds = []
        data  = self.strategy.price_data
        start = data.index[0]
        end   = data.index[-1]

        fold_start = start
        while True:
            train_end  = fold_start + pd.DateOffset(months=train_months)
            test_start = train_end  + pd.DateOffset(days=embargo_days)
            test_end   = test_start + pd.DateOffset(months=test_months)

            if test_end > end:
                break

            folds.append({
                "train": data.loc[fold_start:train_end],
                "test":  data.loc[test_start:test_end]
            })

            fold_start = test_end  # next fold starts where test ended

        return folds
   
    def get_groups(self, N, k):
        edges = np.linspace(0, N, k+1,dtype=int)
        return [np.arange(edges[i],edges[i+1]) for i in range(k-1)]
        
    def plot_paths(self, paths, ax=None, quantiles=[0.2, 0.8]):
        fig, ax = plt.subplots(figsize=(12, 4))

        all_cum = pd.concat([r['cum_ret'] for r in paths], axis=1).sort_index().interpolate()

        for col in all_cum.columns:
            all_cum[col].plot(ax=ax, color='steelblue', linewidth=0.6, alpha=0.2)

        ax.fill_between(all_cum.index,
                        all_cum.quantile(quantiles[0], axis=1),
                        all_cum.quantile(quantiles[1], axis=1),
                        alpha=0.25, color='steelblue', label='IQR band')

        all_cum.mean(axis=1).plot(ax=ax, color='navy', linewidth=2.2, label='Mean path')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title('CPCV Backtest Paths')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        return fig, ax
            
    def CPCV(self, N=5, k=2, embargo=5):
        groups = self.get_groups(N, k)
        for comb in combinations(range(N),k):
            test_ids = list(comb)
            #test_ids says -> 1,2,,3,4, -> need to map those to indices
            test_idx = [groups[g] for g in test_ids]
            train_idx = []
            for g in range(N):
                if g in test_ids:
                    continue
                grp = groups[g]
                #else we have training 
                purge_idx = np.ones(len(grp),dtype=bool)

                for test_group in test_ids:
                    if g == test_group + 1:
                        purge_idx[embargo:] = False
                    if g == test_group -1:
                        purge_idx[-embargo:] = False
                
                train_idx.append(grp[purge_idx])
            
            #by default what does this do? -> creates concatenated list 
            train_idx = np.concatenate(train_idx)

            meta = {
                'test_groups' : test_ids,
                'n_train'     : len(train_idx),
                'n_test'      : len(test_idx)
            }

            yield test_idx, train_idx, meta
    

#signals * returns
def sharpe_ratio(returns: pd.Series):
    return returns.mean() / returns.std()

def permutation_test(signals, returns, n_perms = 100):
    sharpe = sharpe_ratio(signals*returns)
    null_sharpes = []

    for _ in range(n_perms):
        shuffled_rets = np.random.permutation(returns)
        null_sharpes.append(sharpe_ratio(signals * shuffled_rets))
    
    p_val = np.mean(np.array(null_sharpes) >= sharpe)
    return sharpe, null_sharpes, p_val
