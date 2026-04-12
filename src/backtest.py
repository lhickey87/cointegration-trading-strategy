
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from kalman import kalman_filter, run_formation
from DDIVF import DDIVF
from HMM import fit_hmm, get_current_regime
from signals import get_signals, optimise_threshold



class Strategy():


    def __init__():
        pass

    
    def permutation_test(signals, returns, n_perms = 100):
        return
    
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
    
    def get_groups(self,N, k):
        edges = np.linspace(0, N, k+1,dtype=int)
        return [np.arange(edges[i],edges[i+1]) for i in range(k-1)]
            
    def CPCV(self, N = 5, k = 2, embargo = 5):
        #this could be many different stocks?
        groups = self.get_groups(N,k)
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

#
def CPCV(log_a,log_b, N = 5,k=2):
    #N = #of time periods
    #k = # of testing periods
    T = len(log_a)
    group_size = T // N

    #now we wanna define the groups
    #the way we need to do it is put groups 
    groups = []

    for i in range(N):
        start = i * group_size
        end   = (i+1) * group_size if i < N-1 else T
        groups.append((start, end))
    
    #this is the proper way to split them
    all_splits  = list(combinations(range(N), k))
    #all_splits is list of tuples length K
    results     = []
    for test_groups in all_splits:
        #we have test split
        # now we will want to obtain the training split
        train_groups = [elt for elt in range(N) if elt not in test_groups]

        train_idx = []
        for g in train_groups:
            s, e = groups[g]
            train_idx.extend(range(s, e))
        
        train_a = log_a.iloc[train_idx,:]
        train_b = log_b.iloc[train_idx,:]

        spreads, beta, delta, R, x_final,P_final = run_formation(train_a,train_b)
        opt_alpha, S, sigma_DD = DDIVF(spreads)

        model, low_vol_state = fit_hmm(spreads)
        hmm_states = get_current_regime(model,low_vol_state,spreads)
        p_val, best_profit_factor, results = optimise_threshold(innovations=spreads,
                                                                sigma_dd = sigma_DD,
                                                                hmm_states=hmm_states,
                                                                log_a = train_a,
                                                                log_b = train_b,
                                                                beta = beta)
        
        signals = get_signals(hmm_states=hmm_states,
                              pvals = p_val,
                              vols=S,
                              spreads=spreads)
        
        for time in test_groups:


        # -------- TRAINING OPS ----------
        #once we've obtained training data we do the following
        # call kalman_filter -> get hyperparams
        # after this we have our spreads -> call DDIVF to get updated vol for our spreads
        # now that we have DDIVF
        # fit hmm_model -> then obtain optimized p-val thresholds
        # now we can actually generate signals

        # ------- TESTING OPS ----------
        # now we will have k different splits
        for g in test_groups:
            continue





    #define groups