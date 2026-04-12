
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from kalman import kalman_filter, run_formation
from DDIVF import DDIVF
from HMM import fit_hmm, get_current_regime
from signals import get_signals, optimise_threshold



class Backtest():

    def __init__(self, portfolio):
        self.portfolio = portfolio
    
    def plot_paths(paths, ax = None, quantiles = [0.2,0.8]):
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
   
    def get_groups(N, k):
        edges = np.linspace(0, N, k+1,dtype=int)
        return [np.arange(edges[i],edges[i+1]) for i in range(k-1)]
            
    def CPCV(N = 5, k = 2, embargo = 5):
        #this could be many different stocks?
        groups = get_groups(N,k)
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
        null_sharpes.append(sharpe_ratio(signals*returns))
    
    p_val = np.mean(np.array(null_sharpes) >= sharpe)
    return sharpe, null_sharpes, p_val
