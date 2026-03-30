import pandas as pd
import numpy as np
import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant, half_life

def is_I1(series, p_threshold=0.1):
    series = series.dropna()
    if len(series) < 50:
        return False
    
    p_level = adfuller(series, autolag='AIC')[1]
    p_diff = adfuller(series.diff().dropna(), autolag='AIC')[1]
    return (p_level > p_threshold) and (p_diff <= p_threshold)

#so we have cointegration test, what else do we need
#spread = log(y_t) - \beta*log(x_t)-\alpha
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

def cointegration_test(log_a,log_b):
    if not is_I1(log_a) or not is_I1(log_b):
        return None
    
    coint_t, pvalue, crit_value = coint(log_a,log_b)

    if pvalue < 0.05:
        X           = add_constant(log_b)
        model       = OLS(log_a, X).fit()
        beta        = model.params.iloc[1]
        spread      = log_a - beta * log_b
        hl          = half_life(spread)
        
        return {
            'eg_stat'    : coint_t,
            'p_value'    : pvalue,
            'beta'       : beta,
            'half_life'  : hl,
            'spread'     : spread
        }
    
    return None

def correlation_filter(price_matrix, threshold = 0.7):
    log_prices = np.log(price_matrix)
    corr = log_prices.corr()

    pairs = []
    tickers = corr.columns.tolist()
    #tickers will be cols and tickers will be rows
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if abs(corr.loc[t1,t2]) >= threshold:
                pairs.append((t1,t2))
    return pairs

    
