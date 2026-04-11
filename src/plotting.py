import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#results should be an array or Series?
# thing is we might not always be using a heat map

def plot_params(results: dict, n_states = 2):

    thresholds = np.linspace(0.5,2.5,21)
    n = len(thresholds)
    Z = np.full([n,n],np.nan)

    for i,p0 in enumerate(thresholds):
        for j,p1 in enumerate(thresholds):
            key = (round(p0,2),round(p1,2))
            if key in results:
                Z[i,j] = results[key]
    
    marginal_s0 = np.nanmax(Z, axis=1)  # collapse columns (State 1)
    marginal_s1 = np.nanmax(Z, axis=0)  # collapse rows    (State 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    im = ax1.imshow(Z, origin="lower", aspect="auto",
                    extent=[0.5, 2.5, 0.5, 2.5], cmap="RdYlGn")

    plt.colorbar(im, ax=ax1)

    ax1.set_xlabel("State 1 threshold)")
    ax1.set_ylabel("State 0 threshold")
    ax1.set_title("Profit Factor — all combinations")

    ax2 = axes[1]
    ax2.plot(thresholds, marginal_s0, color="steelblue",
             linewidth=2, marker="o", label="State 0")
    ax2.plot(thresholds, marginal_s1, color="darkorange",
             linewidth=2, marker="s", label="State 1")
    ax2.axhline(y=1.0, color="black", linestyle="--",
                alpha=0.5, label="Break even")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Profit Factor")
    ax2.set_title("Marginal response per state")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("threshold_analysis.png", dpi=150)
    plt.show() 

