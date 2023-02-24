import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from dengue_prediction.preprocessing.sklearn_wrapper_statsmodel import SMFormulaWrapper
import statsmodels.formula.api as smf
import statsmodels.api as sm


def create_lagged_features(df: pd.DataFrame, col: str, lag: int):
    df = df.copy()

    df[col + f"_lag_{lag}"] = df.groupby("city")[col].shift(lag)

    return df


def _get_lagged_features(df: pd.DataFrame, lag: int, lag_label: bool = True):

    df = df.copy()
    # Create lagged features
    for col in df.drop(["city", "weekofyear"], axis=1).columns:
        for l in range(1, lag + 1):
            if (not lag_label) & (col == "total_cases"):
                continue

            df = create_lagged_features(df, col, l)

    return df


def create_lagged_training_sets(
    df: pd.DataFrame,
    lag: int,
    pred_lag: int = 1,
    lag_label: bool = True,
    use_predictions: bool = True,
    only_train: bool = False,
):

    df = df.copy()

    # Create lagged features
    df = _get_lagged_features(df, lag=lag, lag_label=lag_label)

    # Drop the nulls due to lag featurs
    X = df.drop("total_cases", axis=1).copy()
    X.dropna(inplace=True)
    df = df.loc[X.index].copy()
    y = df["total_cases"].copy()

    if not only_train:
        l = int(len(X) * 0.7)

        X_train = X.iloc[:l].copy()
        y_train = df["total_cases"].iloc[:l].copy()

        X_test = X.iloc[l:].copy()
        y_test = df["total_cases"].iloc[l:].copy()
    else:
        X_train = X_test = X.copy()
        y_train = y_test = y.copy()

    # Create lagged label

    # Use prediction: add new columns to the dataset with prediction
    # for the n lags
    if (use_predictions) & (pred_lag > 1):
        for n in range(1, pred_lag):
            y_train = y_train.shift(-1).dropna()
            X_train = X_train.iloc[:-1].copy()

            model = _train(X_train, y_train)
            X_train[f"preds_{n}"] = model.predict(X_train)
            X_test[f"preds_{n}"] = model.predict(X_test)
    else:

        X_train = X_train.iloc[: len(X_train) - pred_lag].copy()
        y_train = y_train.shift(-pred_lag).dropna()

        X_test = X_test.iloc[: len(X_test) - pred_lag].copy()
        y_test = y_test.shift(-pred_lag).dropna().copy()

        # X = X.iloc[:-pred_lag].copy().reset_index(drop=True)
        # y = df["total_cases"].shift(-pred_lag).dropna().reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def select_features(X: pd.DataFrame, y: pd.Series):

    X = X.copy()
    y = y.copy()

    lasso = Lasso(alpha=0.5)
    lasso.fit(X, y)

    coef = pd.DataFrame({"features": X.columns, "coef": lasso.coef_})
    coef = coef[coef["coef"] != 0].copy()

    return coef["features"].tolist()


def _train(X, y, conf: bool = False):
    features = select_features(X, y)

    model_formula = "total_cases ~ 1 + city + " + " + ".join(features)

    model = SMFormulaWrapper(
        model_class=smf.glm,
        family=sm.families.NegativeBinomial(alpha=0.001),
        formula=model_formula,
    )

    model.fit(X, y)

    return model


def get_preds_lagged_model(
    df: pd.DataFrame,
    lag: int,
    pred_lag: int = 1,
    lag_label: bool = True,
    use_predictions: bool = False,
    only_train: bool = False,
    return_model: bool = False,
):

    df = df.copy()

    X_train, y_train, X_test, y_test = create_lagged_training_sets(
        df,
        lag=lag,
        pred_lag=pred_lag,
        lag_label=lag_label,
        use_predictions=use_predictions,
    )

    model = _train(X_train, y_train, conf=True)
    if return_model:
        model.columns = list(X_train.columns)
        return model
    preds, low_conf, upp_conf = model.predict(X_train, conf=True)

    preds_test, low_conf_test, upp_conf_test = model.predict(X_test, conf=True)
    preds = pd.concat((preds, preds_test)).reset_index(drop=True)
    y = pd.concat((y_train, y_test)).reset_index(drop=True)
    low_conf = pd.concat((low_conf, low_conf_test)).reset_index(drop=True)
    upp_conf = pd.concat((upp_conf, upp_conf_test)).reset_index(drop=True)

    mae = model._get_metrics(X_test, y_test)

    return preds, y, low_conf, upp_conf, mae


def evaluate_test(df, city: int, lag: int = 10):

    df = df.copy()

    df = df[df["city"] == city].copy()

    A = df[df["train"] == 1].copy()
    B = df[df["train"] == 0].copy()

    C = _get_lagged_features(pd.concat((A, B)), lag=lag, lag_label=True)
    C = C[C["train"] == 0].copy().reset_index(drop=True)

    model = get_preds_lagged_model(
        A, lag=lag, pred_lag=0, only_train=True, return_model=True
    )

    COLS = list(C.filter(regex="cases").columns)[1:]

    p = [np.nan] * lag
    preds = []
    for i in range(len(C)):
        # print(i, end=" ")
        x = model.predict(C.iloc[i]).values[0]
        preds.append(x)
        if i == len(C) - 1:
            break
        _ = p.pop()
        p = [x] + p
        assert len(p) == lag  # print(p)
        # print(dict(zip(COLS[1:], p)))
        C.loc[i + 1, COLS] = C.loc[i + 1, COLS].fillna(value=dict(zip(COLS, p)))
        # print( C.loc[i+1, COLS])

    return preds
