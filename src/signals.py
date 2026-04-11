
import pandas as pd
import numpy as np
from itertools import product
from plotting import plot_params

from statsmodels.tools.typing import NDArray

def get_zscores(vols: np.ndarray, innovations: np.ndarray):
    mean_in = np.mean(innovations)
    #what do we even do with vols?
    return (innovations-mean_in)/vols

def signal(spread, prev_spread, upper_t, prev_upper_t):
    if spread > upper_t and prev_spread < prev_upper_t:
            return -1
    elif spread < -upper_t and prev_spread > -prev_upper_t:
        return 1
    else:
        return 0

def get_signals(hmm_states: pd.Series,
                pvals: tuple,
                vols: pd.Series,
                spreads: pd.Series):

    threshold_t = hmm_states.map(pvals)
    threshold_prev_t = hmm_states.shift(1).map(pvals)

    upper_t = threshold_t * vols
    upper_prev_t = threshold_prev_t * vols.shift(1)

    spread_prev = spreads.shift(1)
    signals = pd.Series(0,index = spreads.index)

    signals[spreads > upper_t & (spread_prev < upper_prev_t)] = -1
    signals[spreads < upper_t & (spread_prev > upper_prev_t)] = 1

    return signals

#threshold is related to the hidden markov state of the model
# this is ONLY training_period
def optimise_threshold(innovations: pd.Series,
                       sigma_dd: pd.Series,
                       hmm_states: pd.Series,
                       log_a: pd.Series,
                       log_b: pd.Series,
                       beta: pd.Series,
                       k: int = 100,
                       n_states: int = 2):

    delta_a = log_a.diff().shift(-1)
    delta_b = log_b.diff().shift(-1)

    thresholds = np.linspace(0.5, 2.5, 21)
    combs      = list(product(thresholds, repeat=n_states))

    best_profit_factor = 0
    best_pvals  = None
    results = {}

    for state_thresholds in combs:

        signals = get_signals(
            hmm_states = hmm_states,
            pvals      = state_thresholds,
            vols       = sigma_dd,
            spreads    = innovations
        )

        signals = signals.iloc[k:]
        beta_t  = beta.iloc[k:]
        delta_a_t = delta_a.iloc[k:]
        delta_b_t = delta_b.iloc[k:]

        #may have to look into different position sizing 
        position_a = -1000 * beta_t * signals
        position_b =  1000 * signals

        pnl = (position_a * delta_a_t + position_b * delta_b_t).dropna()

        trade_returns = pnl[signals.iloc[k:].reindex(pnl.index) != 0]

        if len(trade_returns) < 20 or trade_returns.std() == 0:
            continue
            
        gross_wins = trade_returns[trade_returns > 0].sum()
        gross_losses = trade_returns[trade_returns < 0].sum()
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

        key = (round(state_thresholds[0],2),round(state_thresholds[1],2))
        results[key] = profit_factor

        if profit_factor > best_profit_factor:
            best_profit_factor = profit_factor
            best_pvals  = {s: state_thresholds[s] for s in range(n_states)}
    

    return best_pvals, best_profit_factor, results

