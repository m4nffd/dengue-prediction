import pandas as pd
import matplotlib.pyplot as plt


def plot_preds(y_true: pd.Series, y_preds: pd.Series, lc=None, uc=None, test=True):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    l = int(len(y_true) * 0.7)
    if test:
        ax.plot(y_true[:l], label="true")
        ax.plot(y_preds[:l], label="pred")
        ax.plot(y_true[l:], label="true_test")
        ax.plot(y_preds[l:], label="pred_test")
    else:
        ax.plot(y_true, label="true")
        ax.plot(y_preds, label="pred")

    if lc is not None:
        ax.fill_between(range(l), lc[:l], uc[:l], alpha=1, color="orange")
        ax.fill_between(range(l, len(lc)), lc[l:], uc[l:], alpha=1, color="red")

    ax.set_ylabel("Cases")
    ax.set_xlabel("Week number")
    ax.legend()

    return fig
