
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

# we should probably just start with equal weight across all

#this gives us the signals for the entire trading period
def get_signals(pair_hmm: pd.Series,
                pvals: tuple,
                vols: pd.Series,
                spreads: pd.Series,
                stop_z: int = 3.5,
                exit_z: int = 0.5):

    threshold_t      = pair_hmm.map(lambda r: pvals[int(r)] if pd.notna(r) else np.nan)
    threshold_prev_t = pair_hmm.shift(1).map(lambda r: pvals[int(r)] if pd.notna(r) else np.nan)

    upper_t      = (threshold_t      * vols).values
    upper_prev_t = (threshold_prev_t * vols.shift(1)).values
    exit_band    = (exit_z * vols).values
    stop_band    = (stop_z * vols).values
    s            = spreads.values

    position  = 0
    out       = np.zeros(len(s), dtype=int)

    for i in range(1, len(s)):
        if position != 0:
            if abs(s[i]) < exit_band[i] or abs(s[i]) > stop_band[i]:
                position = 0
        else:
            if s[i] > upper_t[i] and s[i - 1] < upper_prev_t[i]:
                position = -1
            elif s[i] < -upper_t[i] and s[i - 1] > -upper_prev_t[i]:
                position = 1
        out[i] = position

    return pd.Series(out, index=spreads.index)

#threshold is related to the hidden markov state of the model
#this is incredibly slow -> have to call this on every single 
def optimise_threshold(innovations: pd.Series,
                       sigma_dd: pd.Series,
                       hmm_states: pd.Series,
                       log_y: pd.Series,
                       log_x: pd.Series,
                       beta: pd.Series,
                       k: int = 100,
                       n_states: int = 2):

    # align everything to sigma_dd's index (which already starts at bar k
    # after the rolling burn-in — no need for explicit iloc[k:] slices)
    idx          = sigma_dd.index
    innovations  = innovations.reindex(idx)
    hmm_states   = hmm_states.reindex(idx)

    beta         = beta.reindex(idx)

    delta_y      = log_y.reindex(idx).diff().shift(-1)
    delta_x      = log_x.reindex(idx).diff().shift(-1)

    thresholds = np.linspace(0.5, 2.5, 21)
    combs      = list(product(thresholds, repeat=n_states))

    best_sharpe = 0
    best_pvals  = None
    results     = {}

    for state_thresholds in combs:

        signals = get_signals(
            pair_hmm = hmm_states,
            pvals    = state_thresholds,
            vols     = sigma_dd,
            spreads  = innovations
        )

        beta_t    = beta
        delta_y_t = delta_y
        delta_x_t = delta_x

        # signal at t → P&L from t to t+1 (delta already shifted one step ahead)
        position_y = signals
        position_x = -beta_t * signals

        pnl = (position_y * delta_y_t + position_x * delta_x_t).dropna()

        mask = pnl != 0

        trade_ids = (mask & ~mask.shift(fill_value=False)).cumsum()
        trade_ids = trade_ids.where(mask)
        trade_ids = trade_ids[~trade_ids.isna()]

        pnl_active = pnl.reindex(trade_ids.index)

        trade_returns = pnl_active.groupby(trade_ids).sum()

        if len(trade_returns) < 20:
            continue
            
        # gross_wins   = trade_returns[trade_returns > 0].sum()
        # gross_losses = trade_returns[trade_returns < 0].sum()
        #should probably optimize sharpe over profit_factor
        # profit_factor = gross_wins / abs(gross_losses) if gross_losses < 0 else np.inf
        sharpe = trade_returns.mean() / trade_returns.std()*np.sqrt(252)

        key = (round(state_thresholds[0],2),round(state_thresholds[1],2))
        results[key] = sharpe

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_pvals  = {s: state_thresholds[s] for s in range(n_states)}

    return best_pvals, best_sharpe, results

