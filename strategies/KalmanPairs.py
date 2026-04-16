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
from DDIVF import DDIVF, trading_ddivf
from signals import optimise_threshold, get_signals
from universe import sector_map


def entry_beta(signals: pd.Series, betas: pd.Series) -> pd.Series:
    """Returns the beta locked in at the start of each trade, held constant 
    for the duration of the position. Zero when flat."""
    result     = pd.Series(0.0, index=signals.index)
    in_pos     = (signals != 0)
    is_entry   = in_pos & ~in_pos.shift(1, fill_value=False)
    group_ids  = is_entry.cumsum().where(in_pos)

    for gid, group in group_ids.groupby(group_ids):
        result.loc[group.index] = betas.loc[group.index[0]]

    return result

def pair_col(pair: tuple[str, str]) -> str:
    """Tuple pair → DataFrame column string ('AAPL', 'MSFT') → 'AAPL/MSFT'."""
    return f"{pair[0]}/{pair[1]}"

def col_to_pair(col: str) -> tuple[str, str]:
    """DataFrame column string → tuple pair 'AAPL/MSFT' → ('AAPL', 'MSFT')."""
    y, x = col.split('/')
    return (y, x)

def match_index(log_y, log_x):
        comb = pd.concat([log_y,log_x],axis=1).dropna()
        log_y = comb.iloc[:,0]
        log_x = comb.iloc[:,1]
        return log_y, log_x

@dataclass
class DDIVFState:
    alpha:      float
    nu_bar:     float
    rho:        float
    vol:        float
    train_vols: pd.Series
    buffer:     np.ndarray    # last k innovations from training

@dataclass
class KFState:
    x:     np.ndarray  # [beta, alpha]
    P:     np.ndarray  # 2×2 covariance
    Q:     np.ndarray  # process noise (delta * I)
    R:     float       # observation noise variance
    delta: float       # stored so filter can resume after formation window

#so now what we need to do is get sectors_map, then also figure out our data
class KalmanPairs(Strategy):

    def __init__(self, price_data: pd.DataFrame, company_df: pd.DataFrame):
        super().__init__(name="pairs", price_data=price_data)
        self.trading_days = len(price_data.index)
        self.company_df = company_df
        #this should probably be throughouit time
        self.pairs:       list[tuple]          = []
        self.kf_states:   dict[tuple, KFState] = {}
        self.spreads:     pd.DataFrame         = pd.DataFrame()  # training spreads
        self.is_fitted: bool = False
        
        # TESTING
        self.test_spreads: pd.DataFrame        = pd.DataFrame()  # evaluation spreads
        self.test_betas:   pd.DataFrame        = pd.DataFrame()  # evaluation betas
        self.test_alphas:  pd.DataFrame        = pd.DataFrame()  # evaluation alphas
        self.regimes: pd.DataFrame = pd.DataFrame()
        self.vols: pd.DataFrame = pd.DataFrame()

        # TRAINING
        self.train_betas: dict = {}
        self.train_alphas: dict = {}
        self.ddivf_states: dict = {}
        self.hmm_models: dict  = {}
        self.thresholds: dict  = {}
        self.profit_factor: dict = {}
    
    def __repr__(self):
        pass 
    
    def __str__(self):
        return f"KalmanPairs strategy with {len(self.pairs)} active pairs"
    
    def __len__(self):
        return len(self.pairs)
    
    @property
    def trading_days(self):
        return len(self.price_data.index)
    
    @property
    def betas(self) -> dict:
        return {pair: state.x[0] for pair, state in self.kf_states.items()}

    @property
    def alphas(self) -> dict:
        return {pair: state.x[1] for pair, state in self.kf_states.items()}
    

    def preprocess(self, df: pd.DataFrame):
        df = df.dropna(axis=1, how='all')
        #now once we've preprocessed what else?
        return df

    #call fit on each fold on train_data
    #after preprocess we get the tickers from the data
    # then we get the sector_map
    # 3. we reinitialize self.pairs from the new sector_map and train_data
    # 4. we then call _train_kalman on train data

    def fit(self, train_data: pd.DataFrame):
        tickers = train_data.columns
        sector_map = sector_map(tickers,self.company_df)

        self.pairs = self._get_pairs(train_data, sector_map)
        self.kf_states = {}

        #fitting kalman filter over training data
        # onyl reason we would reinitialize
        # kalman is that in test period we wanna be using our previously used info
        self._train_kalman(train_data)

        #fitting the variance factors which deal with fat tailedness of spreads
        self._fit_ddivf()

        #regime fitting over training period
        self._fit_pair_hmm()

        #obtaining the optimal thresholds for each regime based on maximized sharpe
        self._optimise_thresholds(train_data)
        self.is_fitted = True

    def evaluate(self, test_data: pd.DataFrame):
        #so actually not much needs to be done here
        if not self.is_fitted:
            return None, None

        vols         = {}
        regimes      = {}
        test_spread_cols = {}
        test_beta_cols   = {}
        test_alpha_cols  = {}

        for pair in self.pairs:
            y_tick, x_tick = pair
            if y_tick not in test_data.columns or x_tick not in test_data.columns:
                continue
            col = pair_col(pair)

            log_y = np.log(test_data[y_tick]).dropna()
            log_x = np.log(test_data[x_tick]).dropna()

            log_y, log_x = match_index(log_y=log_y,
                                       log_x=log_x)

            spread, betas, alphas = self._kalman_filter(
                pair,
                log_y,
                log_x,
                delta=self.kf_states[pair].delta,
                R=self.kf_states[pair].R
            )

            spread = spread.dropna()

            #ensuring 
            test_spread_cols[col] = spread
            test_beta_cols[col]   = betas.reindex(spread.index)
            test_alpha_cols[col]  = alphas.reindex(spread.index)

            #get volatilities from pair
            ddivf = self.ddivf_states[pair]

            #this returns the last k volatilities -> to be used to project forward and minimized
            # FESS error (foward Error Sum Squared)
            buf   = ddivf.buffer.copy()
            test_vols = np.zeros(len(spread))

            #we've already calculated spread over the full test period
            #what we did above was to take the buffer (last 100 vols)
            # we will be using that as our buffer for the first entry in this
            #then continually uopdatre it
            for i in range(len(spread)):
                _, _, sigma_dd, _ = DDIVF(pd.Series(buf))  # vol from past k obs only
                test_vols[i]      = sigma_dd
                buf = np.append(buf[1:], spread.values[i]) # slide window after

            vols[col] = pd.Series(test_vols, index=spread.index)

            model, calm_state = self.hmm_models[pair]
            regimes[col] = get_current_regime(model, calm_state, spread)

        self.test_spreads = pd.DataFrame(test_spread_cols)
        self.test_betas   = pd.DataFrame(test_beta_cols)
        self.test_alphas  = pd.DataFrame(test_alpha_cols)
        self.vols         = pd.DataFrame(vols)
        regime_df         = pd.DataFrame(regimes)
        self.regimes = regime_df

        signals = self._generate_signals(regime_df)
        weights = self._get_weights(signals)
        return weights, signals
    
    #we have weights throughout these periods but we need to monitor positions
    
    #signals vs weihgts
    def backtest(self, test_data: pd.DataFrame, signals: pd.DataFrame, weights: pd.DataFrame):
        pr_dict = {}
        for pair in self.pairs:
            col = pair_col(pair)
            if col not in self.test_betas.columns:
                continue
            y, x = pair
            ret_y  = np.log(test_data[y]).diff().reindex(self.test_betas.index)
            ret_x  = np.log(test_data[x]).diff().reindex(self.test_betas.index)
            hedge  = entry_beta(signals[col], self.test_betas[col])
            pr_dict[col] = signals[col] * (ret_y - hedge * ret_x) / (1 + hedge)
        pair_returns = pd.DataFrame(pr_dict, index=self.test_betas.index)

        daily_pnl  = (weights * pair_returns).sum(axis=1)
        cumulative = daily_pnl.cumsum()
        sharpe     = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        return daily_pnl, cumulative, sharpe

    def _generate_signals(self, regime_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized signal generation over the full test window."""
        signals = {}
        for pair in self.pairs:
            col = pair_col(pair)
            signals[col] = get_signals(
                pair_hmm=regime_df[col],
                pvals=self.thresholds[pair],
                vols=self.vols[col],
                spreads=self.test_spreads[col]
            )
        return pd.DataFrame(signals)

    def _get_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Equal weight across all active positions. Flat when no signals."""
        num_positions = (signals != 0).sum(axis=1).replace(0, np.nan)
        return signals.div(num_positions, axis=0).fillna(0)

    def _update(self, pair: tuple, log_y_t: float, log_x_t: float):
        """Online Kalman update for a single pair — used in live trading."""
        state = self.kf_states[pair]
        H = np.array([log_x_t, 1.0])
        return self._kalman_step(state, H, log_y_t)

    

    def get_pairs(self,data, sector_map):
        self.pairs = self._get_pairs(data,sector_map)

    def _get_pairs(self, data: pd.DataFrame, sector_map: dict) -> list[tuple]:
        """Cointegration + FDR filtering to select tradeable pairs."""
        pairs_df = combination_filter(data, sector_map)
        if pairs_df.empty:
            return []
        return list(zip(pairs_df["dependent"], pairs_df["independent"]))
    
    def train_kalman(self, train_data: pd.DataFrame):
        self._train_kalman(train_data=train_data)

    def _train_kalman(self, train_data: pd.DataFrame):
        """Run Kalman filter over training data. Populates spreads and kf_states."""
        #this goes through and gives us -> spreads over the train period and betas
        self.train_betas = {}
        self.train_alphas = {}
        self.kf_states = {}
        spread_cols = {}
        for pair in self.pairs:
            y_tick, x_tick = pair
            log_y = np.log(train_data[y_tick])
            log_x = np.log(train_data[x_tick])
            spread, betas, alphas = self._kalman_filter(pair, log_y, log_x)
            spread_cols[pair_col(pair)] = spread
            self.train_betas[pair] = betas
            self.train_alphas[pair] = alphas
        self.spreads = pd.DataFrame(spread_cols)
    
    def _fit_ddivf(self, k: int = 100):
        self.ddivf_states = {}
        for pair in self.pairs:
            col    = pair_col(pair)
            spread = self.spreads[col].dropna()

            # one-step-ahead vol for each bar t using only [t-k : t]
            # first k-1 bars are NaN naturally — no explicit loop needed
            train_vols = spread.rolling(k).apply(
                lambda w: DDIVF(pd.Series(w))[2], raw=True
            )

            # final window params carried into test period
            alpha_final, _, sigma_final, rho_final = DDIVF(spread.iloc[-k:])

            self.ddivf_states[pair] = DDIVFState(
                alpha      = alpha_final,
                nu_bar     = spread.mean(),
                rho        = rho_final,
                vol        = sigma_final,
                train_vols = train_vols.dropna(),
                buffer     = spread.iloc[-k:].values.copy()
            )

    def _fit_pair_hmm(self):
        """Fit Gaussian HMM on each pair's training spread."""
        self.hmm_models = {}
        for pair in self.pairs:
            self.hmm_models[pair] = fit_hmm(self.spreads[pair_col(pair)])

    def _optimise_thresholds(self, train_data: pd.DataFrame):
        """Grid search for optimal HMM-state-dependent signal thresholds."""
        self.profit_factor = {}
        self.thresholds = {}
        for pair in self.pairs:
            y_tick, x_tick = pair
            col = pair_col(pair)
            model, calm_state = self.hmm_models[pair]
            thresholds, profit_factor, _ = optimise_threshold(
                innovations=self.spreads[col],
                sigma_dd=self.ddivf_states[pair].train_vols,
                hmm_states=get_current_regime(model, calm_state, self.spreads[col]),
                log_y=np.log(train_data[y_tick]),
                log_x=np.log(train_data[x_tick]),
                beta=self.train_betas[pair],
            )
            self.profit_factor[pair] = profit_factor
            # fall back to a neutral threshold if optimisation found no valid combos
            if thresholds is None:
                thresholds = {0: 1.0, 1: 1.5}
            self.thresholds[pair] = thresholds

    def _kalman_step(self, state: KFState, H: np.ndarray, y: float):
        """Single predict-update step. Mutates state in place. Returns (nu, S)."""
        state.P = state.P + state.Q
        nu = y - H @ state.x
        #S can be thought of as the uncertainty for this time around
        # so generally if we have a very high S -> trust our 
        # trust prediction or trust measurement?
        S  = float(H @ state.P @ H.T + state.R)
        if S <= 0:
            return nu, S
        K       = (state.P @ H) / S
        state.x = state.x + K * nu
        I_KH    = np.eye(2) - np.outer(K, H)
        state.P = I_KH @ state.P @ I_KH.T + state.R * np.outer(K, K)
        return nu, S

    #kf_states only reinitialized in train_kalman
    def _kalman_filter(self, pair: tuple, log_y: pd.Series, log_x: pd.Series,
                       delta: float = None, R: float = None):
        """Run filter over a full price series. Stores final state in kf_states[pair].

        If kf_states[pair] already exists (i.e. called during evaluation after training),
        warm-starts from the trained state rather than reinitialising from zero.
        """
        if delta is None or R is None:
            delta, R = self._tune_kalman_params(log_y, log_x)

        #kf_states should be reinitialized in train period NOT TEST
        if pair in self.kf_states:
            # warm-start: keep trained x and P, just update Q/R if needed
            state = self.kf_states[pair]
            state.Q = delta * np.eye(2)
            state.R = R
        else:
            state = KFState(x=np.zeros(2), P=np.eye(2), Q=delta * np.eye(2), R=R, delta=delta)
            self.kf_states[pair] = state

        betas  = np.zeros(len(log_y))
        alphas = np.zeros(len(log_y))
        for t in range(len(log_y)):
            H = np.array([log_x.iloc[t], 1.0])
            self._kalman_step(state, H, log_y.iloc[t])
            betas[t], alphas[t] = state.x

        return self._build_spread(log_y, log_x, betas, alphas)

    def _kalman_LL(self, params, log_y: pd.Series, log_x: pd.Series) -> float:
        """Negative log-likelihood for Kalman parameter tuning."""
        delta, R = np.exp(params[0]), np.exp(params[1])
        state = KFState(x=np.zeros(2), P=np.eye(2), Q=delta * np.eye(2), R=R, delta=delta)
        ll = 0.0
        for t in range(len(log_y)):
            H = np.array([log_x.iloc[t], 1.0])
            nu, S = self._kalman_step(state, H, log_y.iloc[t])
            if S <= 0:
                return 1e10
            ll += -0.5 * (np.log(S) + nu ** 2 / S)
        return -ll

    def _tune_kalman_params(self, log_y: pd.Series, log_x: pd.Series):
        """MLE optimisation of delta and R via L-BFGS-B."""
        result = minimize(
            self._kalman_LL,
            x0=[np.log(1e-4), np.log(1.0)],
            args=(log_y, log_x),
            method='L-BFGS-B',
            bounds=[(np.log(1e-6), np.log(1e-1)),
                    (np.log(1e-3), np.log(100.0))],
            options={'maxiter': 1000}
        )
        return np.exp(result.x[0]), np.exp(result.x[1])

    @staticmethod
    def _build_spread(log_y: pd.Series, log_x: pd.Series,
                      betas: np.ndarray, alphas: np.ndarray):
        """Construct spread using lagged beta/alpha to avoid lookahead."""
        beta_s  = pd.Series(betas,  index=log_y.index)
        alpha_s = pd.Series(alphas, index=log_y.index)
        spread  = log_y - beta_s.shift(1) * log_x - alpha_s.shift(1)
        return spread, beta_s, alpha_s
