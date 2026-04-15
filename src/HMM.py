from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd

def fit_hmm(spreads: pd.Series, n_states: int = 2):
    """
    Fits a Gaussian HMM to SPY log returns.
    Returns the fitted model and which state index = calm regime.
    Run this on the formation window only.
    """
    data = spreads.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=1000,
        random_state=42      # reproducibility
    )
    model.fit(data)

    variances  = [model.covars_[i][0][0] for i in range(n_states)]
    calm_state = int(np.argmin(variances))

    return model, calm_state

    #get_current_regime is the call to make


def get_current_regime(model, calm_state: int, spreads: pd.Series) -> pd.Series:
    """
    Returns an integer Series aligned to spreads.index with HMM state per date (0 or 1).
    Reindexes back to the full spreads index so it aligns cleanly with other Series.
    """
    clean  = spreads.dropna()
    states = model.predict(clean.values.reshape(-1, 1))
    return (pd.Series(states, index=clean.index)
              .reindex(spreads.index)
              .ffill()
              .astype(int))
