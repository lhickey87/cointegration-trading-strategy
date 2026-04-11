import pandas as pd
import numpy as np
from scipy.optimize import minimize
from cointegration import cointegration_test
from HMM import fit_hmm, get_current_regime
from DDIVF import DDIVF, ddivf_update, trading_ddivf
from signals import get_zscores, get_signal, optimise_threshold

def kalman_init(delta, R):
    x = np.array([0.0, 0.0])
    P = np.eye(2)
    Q = delta * np.eye(2)
    return x, P, Q

def kalman_step(x, P, H, y, Q, R):
    P = P + Q

    nu = y - (H @ x)[0]
    S  = (H @ P @ H.T)[0, 0] + R

    if S <= 0:
        return x, P, nu, S, None

    K = (P @ H.T) / S

    x = x + K.flatten() * nu

    I_KH = np.eye(2) - K @ H
    P    = I_KH @ P @ I_KH.T + K * R * K.T

    return x, P, nu, S, K

def tune_kalman_params(log_a, log_b):
    # Starting at δ=1e-4, R=1.0 — reasonable defaults
    x0 = [np.log(1e-4), np.log(1.0)]

    # Bounds in log space
    bounds = [
        (np.log(1e-6), np.log(1e-1)),
        (np.log(1e-3), np.log(100.0))
    ]

    result = minimize(
        kalman_LL,
        x0,
        args=(log_a, log_b),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )

    best_delta = np.exp(result.x[0])
    best_R     = np.exp(result.x[1])

    print(f"Optimisation converged: {result.success}")
    print(f"Best δ: {best_delta:.2e}")
    print(f"Best R: {best_R:.4f}")
    print(f"Log likelihood: {-result.fun:.2f}")

    return best_delta, best_R

def kalman_LL(params, log_a, log_b):
    delta = np.exp(params[0])
    R     = np.exp(params[1])

    x, P, Q = kalman_init(delta, R)
    log_like = 0.0

    for t in range(len(log_a)):
        H       = np.array([log_b.iloc[t], 1.0])
        y       = log_a.iloc[t]
        x, P, nu, S, K = kalman_step(x, P, H, y, Q, R)

        if S <= 0:
            return 1e10

        log_like += -0.5 * (np.log(S) + nu**2 / S)

    return -log_like

def run_formation(log_a, log_b):
    """
    Runs on formation window.
    Tunes params and returns calibrated state for trading window.
    """
    best_delta, best_R = tune_kalman_params(log_a, log_b)

    spread, beta_series, _, x_final, P_final = kalman_filter(
        log_a, log_b,
        delta=best_delta,
        R=best_R
    )

    return spread,beta_series, best_delta, best_R, x_final, P_final

#so on the actual trading period we will mainyl want spread and alpha seris
# spread -> what we use to calculate the
# pass in two price series log_a, log_b
def kalman_filter(log_a, log_b,
                  delta, R,
                  x_init = None, P_init = None):

    if delta is None or R is None:
        delta, R = tune_kalman_params(log_a, log_b)

    n            = len(log_a)
    beta_series  = np.zeros(n)
    alpha_series = np.zeros(n)

    x, P, Q = kalman_init(delta, R)
    if P_init:
        P = P_init.copy()
    if x_init:
        x = x_init.copy()

    for t in range(n):
        H          = np.array([log_b.iloc[t], 1.0])
        y          = log_a.iloc[t]
        x, P, nu, S, K = kalman_step(x, P, H, y, Q, R)

        beta_series[t]  = x[0]
        alpha_series[t] = x[1]

    beta_series  = pd.Series(beta_series,  index=log_a.index)
    alpha_series = pd.Series(alpha_series, index=log_a.index)

    beta_lagged  = beta_series.shift(1)
    alpha_lagged = alpha_series.shift(1)

    spread = log_a - beta_lagged * log_b - alpha_lagged

    return spread, beta_series, alpha_series, x, P

#so now we have defined the Hidden Markov Model fitted to innovations
# NOW we should look to find the optimal threshold for profitability
#right now this is doing lots of backtesting work which it shouldnt
def signals(t1, t2, price_data, form_start, form_end,
            rebalance, zscore_window=None):

    log_a = np.log(price_data.loc[form_start:form_end, t1])
    log_b = np.log(price_data.loc[form_start:form_end, t2])

    combined_form = pd.concat([log_a, log_b], axis=1).dropna()
    if len(combined_form) < 60:
        return None

    log_a = combined_form.iloc[:, 0]
    log_b = combined_form.iloc[:, 1]

    result = cointegration_test(log_a, log_b)
    if result is None:
        return None
    _, _, hl = result

    form_spread, beta_series, best_delta, best_R, x_final, P_final = run_formation(log_a, log_b)
    mean_spread = np.mean(form_spread)
    alpha_opt, vols, sigma_dd = DDIVF(form_spread)

    model,calm_state = fit_hmm(form_spread)
    hmm_states = get_current_regime(model,calm_state,full_spread)

    p_vals,_ = optimize_threshold(innovations = form_spread,
                                  sigma_dd = vols,
                                  hmm_states = hmm_states,
                                  log_a_ = log_a,
                                  log_b = log_b,
                                  beta = beta_series)

    trade_a = np.log(price_data.loc[form_end:rebalance, t1])
    trade_b = np.log(price_data.loc[form_end:rebalance, t2])

    combined_trade = pd.concat([trade_a, trade_b], axis=1).dropna()
    if len(combined_trade) < 5:
        return None
    trade_a = combined_trade.iloc[:, 0]
    trade_b = combined_trade.iloc[:, 1]

    trade_spread, beta_series, alpha_series, _, _ = kalman_filter(
        log_a=trade_a,
        log_b=trade_b,
        delta=best_delta,
        R=best_R,
        x_init=x_final,
        P_init=P_final
    )

    trade_window = len(trade_a)

    mean_spread = np.mean(form_spread)
    rho = np.corrcoef(trade_spread-mean_spread,np.sign(trade_spread-mean_spread))
    trade_vols = trading_ddivf(trade_window, trade_spread, sigma_dd, mean_spread,alpha_opt,rho)

    if zscore_window is None:
        zscore_window = int(np.clip(3 * hl, 20, 120))

    #new COLUMNS:
        # HMM_STATES
        # Z-scores
        # Threshold
    full_spread  = pd.concat([form_spread, trade_spread], axis = 0)


    #okay so now we've gotten the trade vols and the hmm_states
    train_zscore = get_zscores(vols,form_spread)
    trade_zscore = get_zscores(trade_vols,trade_spread)

    return {
        'spread'        : trade_spread,
        'zscore'        : trade_zscore,
        'beta'          : beta_series,
        'alpha'         : alpha_series,
        'half_life'     : hl,
        'delta'         : best_delta,
        'R'             : best_R,
        'zscore_window' : zscore_window
    }
