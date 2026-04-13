from strategy import Strategy
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class KalmanPairs(Strategy):

    def __init__(self, data):
        super().__init__()
        self.name = "pairs"
        self.price_data = None
        #we need to have spreads
        # but also we need to be continually keeping track of beta and al pha for this pair
        # we can do self.spreads -> 'tick1/tick2' -> col.split('/') -> 0 ind is first
        self.spreads = {}
        self.betas = {}
        self.alphas = {}
        self.kf_x = {}
        self.kf_P = {}

    def fit(self,train_data):
        self.pairs = self._get_pairs(train_data)
        spread, beta, alpha = self._train_kalman(train_data)
        self.regimes = self._fit_hmm()
        return
    
    def initialize_positions(self, df: pd.DataFrame):
        """Class specific for position logic -> which tickers -> if pairs we can specify"""
        return

    def generate_signals(self, data):
        """Return signals given current bar. Called every bar during evaluate."""
        return

    def get_weights(self, signals, data):
        """Convert signals to position weights."""
        return

    def update(self, bar):
        """Online model updates every bar e.g. Kalman, DVIF."""
        #not sure how this would be a unified function
        return
    
    def _get_pairs(self,data: pd.DataFrame):
        return

    def _fit_hmm(self,data: pd.DataFrame):
        return
    
    def _kalman_step(self, pair, H, y):
        self.kf_P[pair]  = self.kf_P[pair] + self.kf_Q_[pair]
        nu = y - H @ self.kf_x[pair]
        S  = H @ self.kf_P[pair] @ H.T + self.kf_R_[pair]
        if S <= 0:
            return nu, S
        K              = (self.kf_P[pair] @ H) / S
        self.kf_x[pair] = self.kf_x[pair] + K * nu
        I_KH           = np.eye(2) - np.outer(K, H)
        self.kf_P[pair] = I_KH @ self.kf_P[pair] @ I_KH.T + self.kf_R_[pair] * np.outer(K, K)
        return nu, S

    def _kalman_step(self, H, y):
        self.kf_P  = self.kf_P + self.kf_Q
        nu = y - H @ self.kf_x
        S  = H @ self.kf_P @ H.T + self.kf_R
        if S <= 0:
            return nu, S
        K          = (self.kf_P @ H) / S
        self.kf_x  = self.kf_x + K * nu
        I_KH       = np.eye(2) - np.outer(K, H)
        self.kf_P  = I_KH @ self.kf_P @ I_KH.T + self.kf_R * np.outer(K, K)
        return nu, S
    
    def _build_spread(self, log_a, log_b, betas, alphas):
        beta_s  = pd.Series(betas,  index=log_a.index)
        alpha_s = pd.Series(alphas, index=log_a.index)
        spread  = log_a - beta_s.shift(1) * log_b - alpha_s.shift(1)
        return spread, beta_s, alpha_s

    def _kalman_LL(self, params, pair, log_a, log_b):
        delta, R = np.exp(params[0]), np.exp(params[1])
        # reset state for this pair before running
        self.kf_x[pair] = np.array([0.0, 0.0])
        self.kf_P[pair] = np.eye(2)
        self.kf_Q_[pair] = delta * np.eye(2)
        self.kf_R_[pair] = R
        ll = 0.0
        for t in range(len(log_a)):
            H = np.array([log_b.iloc[t], 1.0])
            nu, S = self._kalman_step(pair, H, log_a.iloc[t])
            if S <= 0:
                return 1e10
            ll += -0.5 * (np.log(S) + nu**2 / S)
        return -ll

    def _tune_kalman_params(self, pair,log_a, log_b):
        result = minimize(
            self._kalman_LL,
            x0=[np.log(1e-4), np.log(1.0)],
            args=(pair,log_a, log_b),
            method='L-BFGS-B',
            bounds=[(np.log(1e-6), np.log(1e-1)),   # delta
                    (np.log(1e-3), np.log(100.0))],  # R
            options={'maxiter': 1000}
        )
        return np.exp(result.x[0]), np.exp(result.x[1])  

    def _kalman_filter(self, pair,log_a, log_b, delta=None, R=None):
        if delta is None or R is None:
            self.delta, self.R = self._tune_kalman_params(pair,log_a, log_b)
        
        self.kf_x[pair]  = np.array([0.0, 0.0])
        self.kf_P[pair]  = np.eye(2)
        self.kf_Q_[pair] = delta * np.eye(2)
        self.kf_R_[pair] = R


        betas, alphas = np.zeros(len(log_a)), np.zeros(len(log_a))
        for t in range(len(log_a)):
            H = np.array([log_b.iloc[t], 1.0])
            self._kalman_step(pair,H, log_a.iloc[t])
            betas[t], alphas[t] = self.kf_x

        return self._build_spread(log_a, log_b, betas, alphas)
    
    def _train_kalman(self, train_data):  # needs train_data as argument, not self._current_data
        for pair in self.pairs_:          # pairs_ not pairs (fitted attribute convention)
            y_tick, x_tick = pair[0],pair[1]
            log_a = train_data[y_tick]
            log_b = train_data[x_tick]

            spread, beta_s, alpha_s = self._kalman_filter(pair, log_a, log_b)

            #this gives us what we need for tommorow
            self.spreads[pair] = spread
            self.betas[pair]   = self.kf_x[pair][0]  # current beta after filter
            self.alphas[pair]  = self.kf_x[pair][1]  # current alpha after filter
   