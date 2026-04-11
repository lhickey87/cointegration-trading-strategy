
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from kalman import kalman_filter, run_formation
from DDIVF import DDIVF
from HMM import fit_hmm, get_current_regime
from signals import get_signals, optimise_threshold

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