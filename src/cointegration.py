import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def is_I1(series: pd.Series, p_threshold: float = 0.1) -> bool:
    """True if series is I(1): non-stationary in levels, stationary in diffs."""
    series = series.dropna()
    if len(series) < 50:
        return False
    p_level = adfuller(series, autolag='AIC')[1]
    p_diff  = adfuller(series.diff().dropna(), autolag='AIC')[1]
    return (p_level > p_threshold) and (p_diff <= p_threshold)

def compute_beta(log_dep: pd.Series, log_indep: pd.Series) -> float:
    """OLS hedge ratio: slope from regressing log_dep on log_indep."""
    return np.cov(log_dep, log_indep)[0, 1] / np.var(log_indep)

def half_life(spread: pd.Series) -> float:
    """Mean reversion half-life in days via AR(1) regression on the spread."""
    spread = spread.dropna()
    lag    = spread.shift(1).dropna()
    diff   = spread.diff().dropna()
    lag, diff = lag.iloc[1:], diff.iloc[1:]
    lam = OLS(diff, add_constant(lag)).fit().params.iloc[1]
    return -np.log(2) / lam if lam < 0 else np.inf

def test_cointegration(prices: pd.DataFrame,
                       ticker_a: str,
                       ticker_b: str) -> dict | None:
    log_a = np.log(prices[ticker_a].dropna())
    log_b = np.log(prices[ticker_b].dropna())
    both  = pd.concat([log_a, log_b], axis=1).dropna()

    if len(both) < 60:
        return None

    log_a, log_b = both.iloc[:, 0], both.iloc[:, 1]

    if not is_I1(log_a) or not is_I1(log_b):
        return None

    beta_ab = compute_beta(log_a, log_b)
    beta_ba = compute_beta(log_b, log_a)
    hl_ab   = half_life(log_a - beta_ab * log_b)
    hl_ba   = half_life(log_b - beta_ba * log_a)

    if hl_ab <= hl_ba:
        dep, indep     = ticker_a, ticker_b
        log_dep, log_indep = log_a, log_b
        beta, hl       = beta_ab, hl_ab
    else:
        dep, indep     = ticker_b, ticker_a
        log_dep, log_indep = log_b, log_a
        beta, hl       = beta_ba, hl_ba

    _, pvalue, _ = coint(log_dep, log_indep)

    spread = log_dep - beta * log_indep

    if not (5 <= hl <= 60):
        return None

    if not (0.2 <= beta <= 5.0):
        return None

    return {
        'dependent':   dep,
        'independent': indep,
        'beta':        round(beta, 4),
        'half_life':   round(hl, 1),
        'pvalue':      round(pvalue, 4),
    }

def _filter_by_correlation(log_prices: pd.DataFrame,
                            sector_map: dict,
                            threshold: float) -> dict[str, pd.DataFrame]:
    """Returns sector -> DataFrame of (stock1, stock2) pairs passing correlation threshold."""
    combs_map = {}
    for sector, tickers in sector_map.items():
        combs = pd.DataFrame(combinations(tickers, 2), columns=['stock1', 'stock2'])
        combs['corr'] = combs.apply(
            lambda row: log_prices[row['stock1']].corr(log_prices[row['stock2']]),
            axis=1
        )
        combs = combs[abs(combs['corr']) >= threshold].reset_index(drop=True)
        if not combs.empty:
            combs_map[sector] = combs
    return combs_map


def filter_by_pvalue(pairs: pd.DataFrame, alpha: float = 0.025) -> pd.DataFrame:
    """Filter pairs by EG p-value and rank by strength of cointegration."""
    return pairs[pairs['pvalue'] < alpha].sort_values('pvalue').reset_index(drop=True)

#only thing is that alot of this does assume price_matrix is 

def combination_filter(price_matrix: pd.DataFrame,
                       sector_map: dict,
                       corr_threshold: float = 0.65,
                       alpha: float = 0.025) -> pd.DataFrame:
    """
    Full pair selection pipeline:
      1. Correlation pre-filter (same sector only)
      2. Cointegration test on each correlated pair
      3. FDR correction across all surviving pairs

    Returns a DataFrame of validated pairs with columns:
      dependent, independent, beta, half_life, pvalue, sector
    """
    log_prices = np.log(price_matrix)
    combs_map  = _filter_by_correlation(log_prices, sector_map, corr_threshold)

    if not combs_map:
        print("0 pairs after correlation filter — check sector_map tickers match price_matrix columns")
        return pd.DataFrame()

    n_corr = sum(len(df) for df in combs_map.values())
    print(f"After correlation filter:    {n_corr} pairs")

    rows = []
    for sector, df in combs_map.items():
        results = df.apply(
            lambda row: test_cointegration(price_matrix, row['stock1'], row['stock2']),
            axis=1
        ).dropna().tolist()
        rows.extend([{**r, 'sector': sector} for r in results])

    print(f"After cointegration tests:   {len(rows)} pairs")

    if not rows:
        return pd.DataFrame()

    final = filter_by_pvalue(pd.DataFrame(rows), alpha=alpha)
    print(f"After p-value filter:        {len(final)} pairs")
    return final
