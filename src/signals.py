import warnings
import pandas as pd
import numpy as np
from itertools import product
from numba import njit

@njit
def _shift(arr):
    out    = np.empty_like(arr)
    out[0] = arr[0]
    out[1:] = arr[:-1]
    return out

#the issue right now is that spreads is calculated via changing beta 
#instead we need beta to be fixed when we enter a position
@njit
def get_signals(hmm_vals, pvals, vols, spreads, log_y, log_x, betas_shifted,
                stop_z=3.0, exit_z=0.0):
    hmm_prev         = _shift(hmm_vals)
    vols_prev        = _shift(vols)

    threshold_t      = pvals[hmm_vals]
    threshold_prev_t = pvals[hmm_prev]

    upper_t          = threshold_t * vols
    upper_prev_t     = threshold_prev_t * vols_prev
    exit_band        = exit_z * vols
    stop_band        = stop_z * vols

    position    = 0
    beta_entered = 0.0
    out = np.zeros(len(spreads), dtype=np.int8)

    for i in range(1, len(spreads)):
        if position != 0:
            spread_held_i = log_y[i] - beta_entered * log_x[i]
            if abs(spread_held_i) < exit_band[i] or abs(spread_held_i) > stop_band[i]:
                position = 0
        
        if position == 0:
            if spreads[i] > upper_t[i] and spreads[i-1] < upper_prev_t[i-1]:
                position      = -1
                beta_entered  = betas_shifted[i]
            elif spreads[i] < -upper_t[i] and spreads[i-1] > -upper_prev_t[i-1]:
                position      = 1
                beta_entered  = betas_shifted[i]

        out[i] = position

    return out

def _align_index(innovations, sigma_dd, hmm_states, log_y, log_x, beta):
    idx = sigma_dd.index
    return (
        innovations.reindex(idx),
        hmm_states.reindex(idx),
        beta.reindex(idx),
        log_y.reindex(idx).diff().shift(-1),
        log_x.reindex(idx).diff().shift(-1),
    )

def optimise_threshold(innovations, sigma_dd, hmm_states, log_y, log_x, beta, k=100, n_states=2):
    innovations, hmm_states, beta, delta_y, delta_x = _align_index(
        innovations, sigma_dd,hmm_states, log_y, log_x, beta
        )
    
    valid       = delta_y.notna() & delta_x.notna()

    delta_y_arr = delta_y[valid].values
    delta_x_arr = delta_x[valid].values
    beta_arr    = beta[valid].values
    hmm_vals    = hmm_states[valid].values.astype(int)

    vols_arr    = sigma_dd.reindex(innovations[valid].index).values
    spreads_arr = innovations[valid].values

    combs = list(product(np.linspace(0.5, 2.5, 21), repeat=n_states))
    K, N  = len(combs), len(spreads_arr)

    signals_matrix = np.zeros((K, N), dtype=np.int8)
    for ki, state_thresholds in enumerate(combs):
        signals_matrix[ki] = get_signals(hmm_vals, np.array(state_thresholds),
                                         vols_arr, spreads_arr)

    pnl = (signals_matrix * delta_y_arr + (-beta_arr * signals_matrix) * delta_x_arr) / (1 + beta_arr)

    active = signals_matrix != 0
    pnl_active = np.where(active, pnl, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'), \
         warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sharpes = np.nanmean(pnl_active, axis=1) / np.nanstd(pnl_active, axis=1, ddof=1) * np.sqrt(252)

    sharpes[active.sum(axis=1) < 20] = -np.inf

    best_idx   = np.argmax(sharpes)
    best_pvals = np.array(combs[best_idx])
    results    = {tuple(round(t, 2) for t in c): s for c, s in zip(combs, sharpes)}

    return best_pvals, float(sharpes[best_idx]), results

_dummy_hmm    = np.zeros(10, dtype=np.int64)
_dummy_pvals  = np.array([1.0, 1.5])
_dummy_vols   = np.ones(10)
_dummy_spread = np.zeros(10)
get_signals(_dummy_hmm, _dummy_pvals, _dummy_vols, _dummy_spread)

