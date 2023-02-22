import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_error

def _lag_training_scores(model:object, X:pd.DataFrame, y:pd.Series, n_lag:int):

    _X = X.copy()
    _y = y.copy()

    _y = _y.shift(-n_lag).dropna()
    _X = _X.iloc[:-n_lag]

    model.fit(_X, _y)
    preds = model.predict(_X)

    mae = -mean_absolute_error(_y, preds)

    return mae

def lag_training_scores(base_model:object, X:pd.DataFrame, y:pd.Series, N:int = 200):

    model = deepcopy(base_model)

    results = {}
    for lag in range(1, N):
        results[lag] = _lag_training_scores(model, X, y, n_lag=lag)

    return results



