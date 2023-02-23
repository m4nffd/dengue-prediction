import pandas as pd
import matplotlib.pyplot as plt


def plot_preds(y_true: pd.Series, y_preds: pd.Series, lc=None, uc=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(y_true, label="true")
    ax.plot(y_preds, label="pred")

    if lc is not None:
        ax.fill_between(range(len(lc)), lc, uc, alpha=1, color="orange")



    ax.set_ylabel("Cases")
    ax.set_xlabel("Week number")
    ax.legend()

    return fig

