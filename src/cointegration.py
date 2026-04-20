import pandas as pd
import numpy as np
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
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

def _test_pair(args):
    """Module-level wrapper so ProcessPoolExecutor can pickle it."""
    prices, x, y, sector = args
    r = test_cointegration(prices, x, y)
    return {**r, 'sector': sector} if r is not None else None


def test_cointegration(prices: pd.DataFrame,
                       ticker_a: str,
                       ticker_b: str) -> dict | None:
    log_y = np.log(prices[ticker_a].dropna())
    log_x = np.log(prices[ticker_b].dropna())
    both  = pd.concat([log_y, log_x], axis=1).dropna()

    if len(both) < 60:
        return None

    log_y, log_x = both.iloc[:, 0], both.iloc[:, 1]

    if not is_I1(log_y) or not is_I1(log_x):
        return None

    beta_ab = compute_beta(log_y, log_x)
    beta_ba = compute_beta(log_x, log_y)
    hl_ab   = half_life(log_y - beta_ab * log_x)
    hl_ba   = half_life(log_x - beta_ba * log_y)

    if hl_ab <= hl_ba:
        dep, indep = ticker_a, ticker_b
        log_dep, log_indep = log_y, log_x
        beta, hl  = beta_ab, hl_ab
    else:
        dep, indep = ticker_b, ticker_a
        log_dep, log_indep = log_x, log_y
        beta, hl  = beta_ba, hl_ba

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

from scipy.stats import chi2

def marchenko_pastur_lambda_max(T: int, N: int, sigma2: float = 1.0) -> float:
    Q = T / N
    return sigma2 * (1 + 1/Q + 2 * np.sqrt(1/Q))

def rmt_filter_correlation(corr_matrix: pd.DataFrame, T: int) -> pd.DataFrame:
    """
    Denoise a correlation matrix using RMT eigenvalue clipping.
    Eigenvectors with eigenvalues <= lambda_max are replaced with
    the identity (noise floor). Returns a cleaned correlation matrix.
    """
    N = corr_matrix.shape[0]
    if N < 2 or T < N:
        return corr_matrix

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
    lambda_max = marchenko_pastur_lambda_max(T, N)

    signal_mask = eigenvalues > lambda_max
    n_signal    = signal_mask.sum()

    if n_signal == 0:
        return pd.DataFrame(np.eye(N),
                            index=corr_matrix.index,
                            columns=corr_matrix.columns)

    # reconstruct from signal eigenvalues only
    signal_vals = eigenvalues.copy()

    # replace noise eigenvalues with their mean to preserve trace = N

    noise_mean  = eigenvalues[~signal_mask].mean()
    signal_vals[~signal_mask] = noise_mean

    cleaned = eigenvectors @ np.diag(signal_vals) @ eigenvectors.T

    # rescale diagonal back to 1 (correlation matrix requirement)
    d = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(d, d)

    return pd.DataFrame(cleaned,
                        index=corr_matrix.index,
                        columns=corr_matrix.columns)

def get_correlation(log_y, log_x):
    return np.corrcoef(log_y, log_x)[0,1]

def _filter_by_correlation(df: pd.DataFrame,
                            sector_map: dict,
                            threshold: float, 
                            use_rmt: bool = True) -> dict[str, pd.DataFrame]:
    T = len(df)
    combs_map = {}
    for sector, tickers in sector_map.items():

        raw_corr = df[tickers].corr()
        corr_matrix = (rmt_filter_correlation(raw_corr, T) if use_rmt else raw_corr)

        # upper triangle only → no duplicate pairs, no self-pairs
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        #.where(mask) -> 
        corr_matrix.index.name   = 'stock1'
        corr_matrix.columns.name = 'stock2'

        pairs = (corr_matrix.where(mask)
                             .stack()
                             .reset_index()
                             .set_axis(['stock1', 'stock2', 'corr'], axis=1))

        pairs = pairs[pairs['corr'].abs() >= threshold].reset_index(drop=True)

        #Only sectors
        if not pairs.empty:
            combs_map[sector] = pairs

    return combs_map

def filter_by_pvalue(pairs: pd.DataFrame, alpha: float = 0.025) -> pd.DataFrame:
    """Filter pairs by EG p-value and rank by strength of cointegration."""
    return pairs[pairs['pvalue'] < alpha].sort_values('pvalue').reset_index(drop=True)

def combination_filter(price_matrix: pd.DataFrame,
                       sector_map: dict,
                       use_rmt: bool = True,
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
    combs_map  = _filter_by_correlation(log_prices, sector_map, corr_threshold, use_rmt = use_rmt)

    if not combs_map:
        return pd.DataFrame()

    n_corr = sum(len(df) for df in combs_map.values())
    #combs_map then get our combinations that pass our correlation filter
    # so tasks is a series of tuples of arguments
    tasks = [
        (price_matrix, x, y, sector)
        for sector, df in combs_map.items()
        for x, y in zip(df['stock1'], df['stock2'])
    ]

    with ProcessPoolExecutor() as executor:
        rows = [r for r in executor.map(_test_pair, tasks) if r is not None]

    if not rows:
        return pd.DataFrame()

    final = filter_by_pvalue(pd.DataFrame(rows), alpha=alpha)
    print(f"pairs after p-val filter: {len(final)}")
    return final
