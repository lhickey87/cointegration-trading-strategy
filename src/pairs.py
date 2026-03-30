import pandas as pd
import numpy as np


#so this only works for a particular window as well as for a particular sector
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

