from hmmlearn import GaussianHMM
import numpy as np
import pandas as pd

def fit_hmm(spreads: pd.Series, n_states: int = 2):
    """
    Fits a Gaussian HMM to SPY log returns.
    Returns the fitted model and which state index = calm regime.
    Run this on the formation window only.
    """
    data = spreads.dropna().values.reshape(-1, 1)

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


def get_current_regime(model, calm_state: int, spreads: pd.Series | pd.DataFrame):
    """
    Predicts current regime based on previous spreads
    if spreads high -> predicts high vol regime -> uses higher threshold
    """
    data = spreads.dropna().values.reshape(-1, 1)

    if len(data) < 10:
        return True   # not enough data — default to trading

    states        = model.predict(data)
    current_state = states[-1]

    return current_state == calm_state
