from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import os, sys
from contextlib import redirect_stdout, redirect_stderr

def fit_hmm(spreads: pd.Series, n_states: int = 2):
    data = spreads.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=1000,
        random_state=42
    )

    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            model.fit(data)

    variances  = [model.covars_[i][0][0] for i in range(n_states)]
    calm_state = int(np.argmin(variances))

    return model, calm_state

#spreads should already be droppedna
def get_current_regime(model, calm_state: int, spreads: pd.Series) -> pd.Series:
    clean  = spreads.dropna()
    states = model.predict(clean.values.reshape(-1, 1))
    return (pd.Series(states, index=clean.index)
              .reindex(spreads.index)
              .ffill()
              .astype(int))
