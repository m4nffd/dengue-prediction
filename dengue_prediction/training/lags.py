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


def create_lagged_training_sets(
    df: pd.DataFrame,
    lag: int,
    pred_lag: int = 1,
    lag_label: bool = True,
    use_predictions: bool = True,
):

    df = df.copy()

    for col in df.drop(["city", "weekofyear"], axis=1).columns:
        for l in range(1, lag + 1):
            if (not lag_label) & (col == "total_cases"):
                continue

            df = create_lagged_features(df, col, l)

    # We need to drop the nulls
    X = df.drop("total_cases", axis=1).copy()
    X.dropna(inplace=True)
    df = df.loc[X.index].copy()
    y = df["total_cases"].copy()

    if (use_predictions) & (pred_lag > 1):
        for n in range(1, pred_lag):
            y = y.shift(-1).dropna().reset_index(drop=True)
            X = X.iloc[:-1].copy().reset_index(drop=True)
            X[f"preds_{n}"] = _train(X, y).values
    else:
        X = X.iloc[:-pred_lag].copy().reset_index(drop=True)
        y = df["total_cases"].shift(-pred_lag).dropna().reset_index(drop=True)

    return X, y


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
        family=sm.families.NegativeBinomial(alpha=0.00001),
        formula=model_formula,
    )

    model.fit(X, y)

    return model.predict(X, conf=conf)


def get_preds_lagged_model(
    df: pd.DataFrame,
    lag: int,
    pred_lag: int = 1,
    lag_label: bool = True,
    use_predictions: bool = False,
):

    df = df.copy()

    X, y = create_lagged_training_sets(
        df,
        lag=lag,
        pred_lag=pred_lag,
        lag_label=lag_label,
        use_predictions=use_predictions,
    )

    preds, low_conf, upp_conf = _train(X, y, conf=True)

    return preds, y, low_conf, upp_conf
