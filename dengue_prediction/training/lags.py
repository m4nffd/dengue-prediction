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

def create_lagged_training_sets(df: pd.DataFrame, lag: int, pred_lag: int = 1):

    df = df.copy()

    for col in df.drop(["city", "weekofyear"], axis=1).columns:
        for l in range(1, lag+1):
            df = create_lagged_features(df, col, l)


    y = df["total_cases"].shift(-pred_lag)

    df.drop("total_cases", axis=1, inplace=True)
    df["label"] = y.values
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    return  df.drop("label", axis=1), df[["label"]]

def select_features(X: pd.DataFrame, y:pd.Series):

    X = X.copy()
    y = y.copy()

    lasso = Lasso(alpha=0.5)
    lasso.fit(X, y)

    coef = pd.DataFrame({"features": X.columns, "coef": lasso.coef_})
    coef = coef[coef["coef"] != 0].copy()

    return coef["features"].tolist()

def get_preds_lagged_model(df: pd.DataFrame, lag: int, pred_lag: int = 1):

    df = df.copy()

    X, y = create_lagged_training_sets(df, lag=lag, pred_lag=pred_lag)

    features = select_features(X, y)

    model_formula = "label ~ 1 + city + " + " + ".join(features)

    model = SMFormulaWrapper(model_class=smf.glm, family=sm.families.NegativeBinomial(alpha=0.00001),
                             formula=model_formula)

    model.fit(X, y)

    preds, low_conf, upp_conf = model.predict(X, conf=True)

    return preds, y, low_conf, upp_conf








