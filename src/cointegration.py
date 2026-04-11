import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from itertools import combinations
from scipy.optimize import minimize
from statsmodels.tools.tools import add_constant
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def is_I1(series, p_threshold=0.1):
    series = series.dropna()
    if len(series) < 50:
        return False

    p_level = adfuller(series, autolag='AIC')[1]
    p_diff = adfuller(series.diff().dropna(), autolag='AIC')[1]
    return (p_level > p_threshold) and (p_diff <= p_threshold)

def half_life(spread):
    spread   = spread.dropna()
    lag      = spread.shift(1).dropna()
    diff     = spread.diff().dropna()

    lag  = lag.iloc[1:]
    diff = diff.iloc[1:]

    model = OLS(diff, add_constant(lag)).fit()
    lam   = model.params.iloc[1]
    if lam >= 0:
        return np.inf

    return -np.log(2) / lam

def fdr_correction(combs, alpha=0.05):
    combs = combs.copy()
    combs['test_margin'] = combs['coint_t'] - combs['critical_value']
    
    combs = combs[combs['test_margin'] > 0]
    return combs.sort_values('test_margin', ascending=False)

def test_cointegration(prices, ticker_a, ticker_b):
    """
    Tests whether two assets are cointegrated using Johansen.
    
    Returns a dict with:
        - is_cointegrated: bool
        - dependent:       ticker of dependent asset
        - independent:     ticker of independent asset  
        - beta:            hedge ratio
        - half_life:       mean reversion speed in days
    
    Returns None if pair fails any check.
    """
    
    # --- clean data ---
    log_a = np.log(prices[ticker_a])
    log_b = np.log(prices[ticker_b])
    
    both = pd.concat([log_a, log_b], axis=1).dropna()
    if len(both) < 50:
        return None
    
    log_a = both.iloc[:, 0]
    log_b = both.iloc[:, 1]
    
    if not is_I1(log_a) or not is_I1(log_b):
        return None
    
    jres = coint_johansen(both, det_order=0, k_ar_diff=1)
    
    trace_stat     = jres.lr1[0]
    critical_value = jres.cvt[0, 1]   # 95% critical value
    
    if trace_stat <= critical_value:
        return None
    
    # --- extract hedge ratio and direction ---
    v1, v2 = jres.evec[:, 0]
    
    # asset with larger absolute loading is the dependent variable
    if abs(v1) >= abs(v2):
        dependent   = ticker_a
        independent = ticker_b
        beta        = -(v2 / v1)
        spread      = log_a - beta * log_b
    else:
        dependent   = ticker_b
        independent = ticker_a
        beta        = -(v1 / v2)
        spread      = log_b - beta * log_a
    
    if not (0.1 <= beta <= 10.0):
        return None
    
    hl = half_life(spread)
    if not (5 <= hl <= 60):
        return None
    
    return {
        'dependent':   dependent,
        'independent': independent,
        'beta':        beta,
        'half_life':   round(hl, 1),
        'trace_stat':  round(trace_stat, 3)
    }

def get_corr(log_prices,ticker1, ticker2):
    return log_prices[ticker1].corr(log_prices[ticker2])

def combination_filter(price_matrix, tickers, corr_threshold=0.95, fdr_alpha=0.05):
    
    log_prices = np.log(price_matrix[tickers].dropna(axis=1))
    
    # all pairs
    combs = pd.DataFrame(
        combinations(log_prices.columns, 2),
        columns=['stock1', 'stock2']
    )
    
    # correlation pre-filter
    combs['corr'] = combs.apply(
        lambda row: log_prices[row['stock1']].corr(log_prices[row['stock2']]),
        axis=1
    )
    combs = combs[abs(combs['corr']) >= corr_threshold].reset_index(drop=True)
    
    if combs.empty:
        return combs
    
    rows = []
    for _, row in combs.iterrows():
        result = test_cointegration(price_matrix, row['stock1'], row['stock2'])
        if result is not None:
            rows.append(result)
    
    if not rows:
        return pd.DataFrame()
    
    pairs = pd.DataFrame(rows)
    pairs = pairs.sort_values('trace_stat', ascending=False).reset_index(drop=True)
    
    pairs = fdr_correction(pairs, alpha=fdr_alpha)
    
    return pairs