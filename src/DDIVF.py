import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#so eventually we will have to do testing to determine how often we should actually change alpha
#should I cretae a function/class for that?

def trading_ddivf(trading_window: int,
                spreads: np.ndarray,
                prev_vol: float,
                mean_spread: float,
                alpha: float,
                rho: float):
    #we should return the vols over trading period
    #why am I blanking on this
    S = np.zeros(trading_window)
    for i in range(trading_window):
        S[i] = ddivf_update(spreads[i],prev_vol,mean_spread,alpha,rho)
        prev_vol = S[i]
    return pd.Series(S)


def ddivf_update(new_innovation: float,
                 prev_vol: float,
                 alpha: float,
                 mean_innovation: float,
                 rho: float) -> float:
    """
    Lightweight daily update — no optimisation needed.
    Just applies the pre-optimised alpha_opt to update the EWMA.

    new_innovation: today's ν_t from Kalman filter
    prev_sigma_dd:  yesterday's σ̂^DD_{t-1}
    alpha_opt:      optimised on formation window, fixed
    nu_bar:         innovation mean from formation window, fixed
    rho:            sign correlation from formation window, fixed
    """
    V_t = abs(new_innovation-mean_innovation)/rho
    S = alpha*prev_vol+(1-alpha)*V_t
    return S

#we are only REALLY concerned with prev_vol or cur_vol as measure of z-score
def DDIVF(innovations: pd.Series, min_window: int = 10) -> tuple:
    """
    runs once each new rebalance date
    returns:
        optimal_alpha -> alpha that minimizes vol prediction error
        sigma_DD -> first rebalance day vol prediction
        S -> full vol data over reformation period
    """

    nu_bar     = np.mean(innovations)
    deviations = innovations - nu_bar
    signs      = np.sign(deviations)
    mask       = signs != 0
    rho        = np.corrcoef(deviations[mask], signs[mask])[0, 1]

    if abs(rho) < 1e-6:
        rho = 0.798

    V = np.abs(deviations) / rho
    S_init = V[0]

    alphas    = np.arange(0.01, 0.51, 0.01)
    best_fess = np.inf
    alpha_opt = 0.1

    for alpha in alphas:
        S    = S_init
        fess = 0.0

        for s in range(1, len(V)):
            one_step_error = (V[s] - S) ** 2
            if s >= min_window:                  # only count after burn-in
                fess += one_step_error
            S = alpha * V[s] + (1 - alpha) * S

        if fess < best_fess:
            best_fess = fess
            alpha_opt = alpha

#actually so we are technically
    S = [S_init]
    for s in range(1, len(V)):
        S.append(alpha_opt * V[s] + (1 - alpha_opt) * S[s-1])
    
    sigma_DD = S[-1]
    return alpha_opt, S, sigma_DD
