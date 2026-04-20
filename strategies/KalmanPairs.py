from dataclasses import dataclass
import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from strategy import Strategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from cointegration import combination_filter
from HMM import fit_hmm, get_current_regime
from DDIVF import DDIVF
from signals import optimise_threshold, get_signals
from universe import get_sector_map
from numba import njit

TRANSACTION_COST = 0.001

def entry_beta(signals: pd.Series, betas: pd.Series) -> pd.Series:
    """Beta locked in at trade entry, held constant for the duration."""
    result    = pd.Series(0.0, index=signals.index)
    in_pos    = signals != 0
    is_entry  = in_pos & ~in_pos.shift(1, fill_value=False)
    group_ids = is_entry.cumsum().where(in_pos)
    for _, group in group_ids.groupby(group_ids):
        result.loc[group.index] = betas.loc[group.index[0]]
    return result

def pair_col(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"

def col_to_pair(col: str) -> tuple[str, str]:
    y, x = col.split('/')
    return (y, x)

def match_index(log_y: pd.Series, log_x: pd.Series) -> tuple[pd.Series, pd.Series]:
    combined = pd.concat([log_y, log_x], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]

@dataclass
class DDIVFState:
    alpha:      float
    nu_bar:     float
    rho:        float
    vol:        float
    train_vols: pd.Series
    buffer:     np.ndarray

@dataclass
class KalmanConfig:
    ewm_halflife:        int             = 20
    optimise_thresholds: bool            = True
    thresholds:          tuple[float, float] = (0.7, 1.5)  # fallback / fixed thresholds
    use_hmm: bool             = True          # ← new

@dataclass
class KFState:
    x:     np.ndarray   # [beta, alpha]
    P:     np.ndarray   # 2×2 covariance
    Q:     np.ndarray   # process noise
    R:     float        # observation noise variance
    delta: float

@njit
def _kalman_ll_core(log_y: np.ndarray, log_x: np.ndarray, delta: float, R: float) -> float:
    x, P, Q, ll = np.zeros(2), np.eye(2), delta * np.eye(2), 0.0
    for t in range(len(log_y)):
        H  = np.array([log_x[t], 1.0])
        P  = P + Q
        nu = log_y[t] - H @ x
        S  = H @ P @ H + R
        if S <= 0:
            return 1e10
        K  = (P @ H) / S
        x  = x + K * nu
        P  = (np.eye(2) - np.outer(K, H)) @ P
        ll += -0.5 * (np.log(S) + nu ** 2 / S)
    return -ll

_kalman_ll_core(np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64), 1e-4, 1.0)  # warmup

class KalmanPairs(Strategy):

    def __init__(self, price_data: pd.DataFrame, company_df: pd.DataFrame,config: KalmanConfig = None):
        super().__init__(name="pairs", price_data=price_data)
        self.company_df    = company_df
        self.config        = config or KalmanConfig()
        self.pairs:        list[tuple]             = []
        self.kf_states:    dict[tuple, KFState]    = {}
        self.train_betas:  dict[tuple, pd.Series]  = {}
        self.train_alphas: dict[tuple, pd.Series]  = {}
        self.ddivf_states: dict[tuple, DDIVFState] = {}
        self.hmm_models:   dict                    = {}
        self.thresholds:   dict[tuple, np.ndarray] = {}
        self.profit_factor: dict                   = {}
        self.spreads:      pd.DataFrame = pd.DataFrame()
        self.test_spreads: pd.DataFrame = pd.DataFrame()
        self.test_betas:   pd.DataFrame = pd.DataFrame()
        self.test_alphas:  pd.DataFrame = pd.DataFrame()
        self.regimes:      pd.DataFrame = pd.DataFrame()
        self.vols:         pd.DataFrame = pd.DataFrame()
        self.is_fitted:    bool         = False

    def __str__(self):
        return f"KalmanPairs strategy with {len(self.pairs)} active pairs"

    @property
    def trading_days(self) -> int:
        return len(self.price_data.index)

    @property
    def betas(self) -> dict:
        return {pair: state.x[0] for pair, state in self.kf_states.items()}

    @property
    def alphas(self) -> dict:
        return {pair: state.x[1] for pair, state in self.kf_states.items()}

    def fit(self, train_data: pd.DataFrame):
        sector_map     = get_sector_map(train_data.columns, self.company_df)
        self.pairs     = self._get_pairs(train_data, sector_map)
        self.kf_states = {}
        self._train_kalman(train_data)
        self._fit_ddivf()
        if self.config.use_hmm:
            self._fit_pair_hmm()
        self._optimise_thresholds(train_data)
        self.is_fitted = True

    def evaluate(self, test_data: pd.DataFrame):
        if not self.is_fitted:
            return None, None

        spread_cols, beta_cols, alpha_cols, vol_cols, regime_cols = {}, {}, {}, {}, {}

        for pair in self.pairs:
            col = pair_col(pair)

            if not self._pair_available(pair, test_data):
                continue

            spread, betas, alphas = self._test_kalman(pair, test_data)
            if spread is None or spread.dropna().empty:
                continue

            spread_cols[col]  = spread
            beta_cols[col]    = betas.reindex(spread.index)
            alpha_cols[col]   = alphas.reindex(spread.index)
            vol_cols[col]     = self._rolling_test_vols(pair, spread)
            if self.config.use_hmm:
                regime_cols[col] = get_current_regime(*self.hmm_models[pair], spread)

        self.test_spreads = pd.DataFrame(spread_cols)
        self.test_betas   = pd.DataFrame(beta_cols)
        self.test_alphas  = pd.DataFrame(alpha_cols)
        self.vols         = pd.DataFrame(vol_cols)
        self.regimes      = pd.DataFrame(regime_cols)

        signals = self._generate_signals(self.regimes, test_data)
        weights = self._get_weights(signals)
        return weights, signals

    def backtest(self, test_data: pd.DataFrame, signals: pd.DataFrame, weights: pd.DataFrame):
        idx          = self.test_spreads.index
        pr_dict      = {}
        pair_sharpes = {}

        for pair in self.pairs:
            col = pair_col(pair)
            if col not in self.test_betas.columns or col not in signals.columns:
                continue
            sig             = signals[col].reindex(idx).fillna(0).astype(int)
            rets            = self._pair_pnl(pair, test_data, sig, idx)
            pr_dict[col]    = rets
            active_rets     = rets[sig!= 0]
            pair_sharpes[col] = _sharpe(active_rets)

        pair_returns = pd.DataFrame(pr_dict, index=idx).fillna(0)
        daily_pnl    = (weights.reindex(idx).fillna(0) * pair_returns).sum(axis=1)

        return pair_returns, daily_pnl, pd.Series(pair_sharpes), _sharpe(daily_pnl)

    def _pair_available(self, pair: tuple, test_data: pd.DataFrame) -> bool:
        y, x    = pair
        missing = [t for t in (y, x) if t not in test_data.columns]
        if missing:
            self.remove_missing_ticker(*missing)
            return False
        if pair not in self.ddivf_states:
            return False
        if self.config.use_hmm and pair not in self.hmm_models:
            return False
        return True

    def _test_kalman(self, pair: tuple, test_data: pd.DataFrame):
        """Kalman filter on test window. Returns (spread, betas, alphas) or (None, None, None)."""
        y, x           = pair
        log_y, log_x   = match_index(np.log(test_data[y]).dropna(), np.log(test_data[x]).dropna())
        spread, betas, alphas = self._kalman_filter(
            pair, log_y, log_x,
            delta=self.kf_states[pair].delta,
            R=self.kf_states[pair].R,
        )
        spread = spread.dropna()
        return (spread, betas, alphas) 

    def _rolling_test_vols(self, pair: tuple, spread: pd.Series) -> pd.Series:
        """Slide the DDIVF buffer across the test spread to produce vol estimates."""
        buf  = self.ddivf_states[pair].buffer.copy()
        vols = np.zeros(len(spread))
        for i, val in enumerate(spread.values):
            _, _, vols[i], _ = DDIVF(buf)
            buf = np.append(buf[1:], val)
        return pd.Series(vols, index=spread.index)

    def _pair_pnl(self, pair: tuple, test_data: pd.DataFrame,
                  signals: pd.Series, idx: pd.Index) -> pd.Series:
        y, x  = pair
        ret_y = np.log(test_data[y]).diff().shift(-1).reindex(idx)
        ret_x = np.log(test_data[x]).diff().shift(-1).reindex(idx)
        hedge = entry_beta(signals, self.test_betas[pair_col(pair)])
        TC = TRANSACTION_COST
        #now in this case we are using ret_y - hedge * ret_x -> should we also subtract alpha?
        signal_change = signals.diff().abs().fillna(signals.abs())

        rets  = (signals * (ret_y - hedge * ret_x) / (1 + hedge)).fillna(0)
        #TRANSACTION COSTS
        #this is wayyy too many transaction costs we wouldnt actually be tradingt this much
        rets = rets - TC * signal_change
        rets[signals == 0] = 0.0
        return rets

    def _generate_signals(self, regime_df: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        signals = {}
        for pair in self.pairs:
            col = pair_col(pair)
            if pair not in self.thresholds:
                continue
            if col not in self.test_spreads.columns:
                continue
            if self.config.use_hmm and col not in regime_df.columns:
                continue
            signals[col] = self._signals_for_pair(pair, regime_df, test_data)
        return pd.DataFrame(signals)
    
    #find price_series

    def _signals_for_pair(self, pair: tuple, regime_df: pd.DataFrame, test_data: pd.DataFrame) -> pd.Series:
        col = pair_col(pair)
        log_y = np.log(test_data[pair[0]]) #this should be an nparray
        log_x = np.log(test_data[pair[1]])
        # spread_vals = self.test_spreads[col].dropna()
        pair_idx    = spread_vals.index
        # spread_mean = spread_vals.ewm(halflife=self.config.ewm_halflife).mean().shift(1).fillna(0)
        # centered    = (spread_vals - spread_mean).values

        if self.config.use_hmm:
            hmm_vals = regime_df[col].reindex(pair_idx).fillna(0).values.astype(np.int64)
        else:
            hmm_vals = np.zeros(len(pair_idx), dtype=np.int64)  # single state

        #nvm we do actually pass in the demeaned spreads 
        raw_signals = get_signals(
            hmm_vals = hmm_vals,
            pvals    = self.thresholds[pair],
            vols     = self.vols[col].loc[pair_idx].values,
            spreads  = centered,
        )
        return pd.Series(raw_signals, index=pair_idx)

    def _get_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        num_active = (signals != 0).sum(axis=1).replace(0, np.nan)
        return (signals != 0).astype(float).div(num_active, axis=0).fillna(0)

    def _get_pairs(self, data: pd.DataFrame, sector_map: dict, use_rmt: bool = False) -> list[tuple]:
        pairs_df = combination_filter(data, sector_map, use_rmt=use_rmt)
        if pairs_df.empty:
            self.pairs_df = pd.DataFrame()
            return []
        self.pairs_df = pairs_df
        return list(zip(pairs_df["dependent"], pairs_df["independent"]))

    def _train_kalman(self, train_data: pd.DataFrame):
        self.train_betas = {}; self.train_alphas = {}; self.kf_states = {}
        spread_cols = {}
        for pair in self.pairs:
            y, x                   = pair
            spread, betas, alphas  = self._kalman_filter(pair, np.log(train_data[y]), np.log(train_data[x]))
            spread_cols[pair_col(pair)] = spread
            self.train_betas[pair]  = betas
            self.train_alphas[pair] = alphas
        self.spreads = pd.DataFrame(spread_cols)

    def _fit_ddivf(self, k: int = 100):
        self.ddivf_states = {}
        for pair in self.pairs:
            col    = pair_col(pair)
            spread = self.spreads[col].dropna()
            if len(spread) < k:
                continue

            train_vols          = spread.rolling(k).apply(lambda w: DDIVF(w)[2], raw=True)

            alpha, _, sigma, rho = DDIVF(spread.iloc[-k:].values)

            self.ddivf_states[pair] = DDIVFState(
                alpha=alpha, 
                nu_bar=spread.mean(), 
                rho=rho, vol=sigma,
                train_vols=train_vols.dropna(), 
                buffer=spread.iloc[-k:].values.copy(),
            )

    def _fit_pair_hmm(self):
        self.hmm_models = {}
        for pair in self.pairs:
            if pair in self.ddivf_states:
                self.hmm_models[pair] = fit_hmm(self.spreads[pair_col(pair)].dropna())

    def _optimise_thresholds(self, train_data: pd.DataFrame):
        self.profit_factor = {}
        self.thresholds    = {}

        default = np.array(self.config.thresholds)

        for pair in self.pairs:
            if pair not in self.ddivf_states:
                continue
            if self.config.use_hmm and pair not in self.hmm_models:
                continue
            if not self.config.optimise_thresholds:
                self.thresholds[pair]    = default
                self.profit_factor[pair] = np.nan
                continue
            self.thresholds[pair], self.profit_factor[pair] = \
                self._optimise_pair_thresholds(pair, train_data, default)

    def _optimise_pair_thresholds(self, pair: tuple, train_data: pd.DataFrame,
                                   default: np.ndarray) -> tuple[np.ndarray, float]:
        y, x     = pair
        col      = pair_col(pair)
        raw      = self.spreads[col]
        centered = raw - raw.ewm(halflife=self.config.ewm_halflife).mean().shift(1).fillna(0)

        if self.config.use_hmm:
            model, calm_state = self.hmm_models[pair]
            hmm_states = get_current_regime(model, calm_state, centered)
        else:
            hmm_states = pd.Series(0, index=centered.index)  # single state — no regime

        thresholds, profit_factor, _ = optimise_threshold(
            innovations = centered,
            sigma_dd    = self.ddivf_states[pair].train_vols,
            hmm_states  = hmm_states,
            log_y       = np.log(train_data[y]),
            log_x       = np.log(train_data[x]),
            beta        = self.train_betas[pair],
        )

        result = np.array([thresholds[s] for s in range(2)]) if thresholds is not None else default
        return result, profit_factor

    def _kalman_filter(self, pair: tuple, log_y: pd.Series, log_x: pd.Series,
                       delta: float = None, R: float = None):
        if delta is None or R is None:
            delta, R = self._tune_kalman_params(log_y, log_x)

        if pair in self.kf_states:
            state = self.kf_states[pair] 
            state.Q = delta * np.eye(2) 
            state.R = R
        else:
            state = KFState(x=np.zeros(2), P=np.eye(2), Q=delta * np.eye(2), R=R, delta=delta)
            self.kf_states[pair] = state

        betas, alphas = np.zeros(len(log_y)), np.zeros(len(log_y))
        for t in range(len(log_y)):
            self._kalman_step(state, np.array([log_x.iloc[t], 1.0]), log_y.iloc[t])
            betas[t], alphas[t] = state.x
        return self._build_spread(log_y, log_x, betas, alphas)

    def _kalman_step(self, state: KFState, H: np.ndarray, y: float):
        state.P = state.P + state.Q
        nu      = y - H @ state.x
        S       = float(H @ state.P @ H.T + state.R)
        if S <= 0:
            return nu, S
        K       = (state.P @ H) / S
        state.x = state.x + K * nu
        I_KH    = np.eye(2) - np.outer(K, H)
        state.P = I_KH @ state.P @ I_KH.T + state.R * np.outer(K, K)
        return nu, S

    def _tune_kalman_params(self, log_y: pd.Series, log_x: pd.Series) -> tuple[float, float]:
        result = minimize(
            lambda p: _kalman_ll_core(log_y.values, log_x.values, np.exp(p[0]), np.exp(p[1])),
            x0=[np.log(1e-4), np.log(1.0)],
            method='L-BFGS-B',
            bounds=[(np.log(1e-6), np.log(1e-1)), (np.log(1e-3), np.log(100.0))],
            options={'maxiter': 1000},
        )
        return np.exp(result.x[0]), np.exp(result.x[1])

    #the spread is actually calculated via the alpha_s
    #so if we deamean we are effectively taking 
    @staticmethod
    def _build_spread(log_y: pd.Series, log_x: pd.Series,
                      betas: np.ndarray, alphas: np.ndarray):
        """Lagged beta/alpha avoids lookahead — first bar is always NaN."""
        beta_s  = pd.Series(betas,  index=log_y.index)
        alpha_s = pd.Series(alphas, index=log_y.index)
        spread  = (log_y - beta_s.shift(1) * log_x - alpha_s.shift(1)).dropna()
        return spread, beta_s, alpha_s

    def remove_missing_ticker(self, tick_a: str, tick_b: str = None):
        to_remove = {tick_a, tick_b} if tick_b else {tick_a}
        self.pairs = [p for p in self.pairs if not (p[0] in to_remove or p[1] in to_remove)]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, how='all')


def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


#where would the issue come in is my question. calc spread via log_y - beta.shift(1)*log_x - alpha
# then in signals we actually use a demeaned version
# so we aren't actually getting a good idea of the spread without subtracting alpha away? 
