import pandas as pd
import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression

def get_regression_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    coefficients = {}

    for c in df.columns:
        _cols = [_c for _c in df.columns if _c != c]
        lr = LinearRegression()
        lr.fit(df[_cols], df[[c]])
        coefficients[c] = {"intercept": lr.intercept_[0], **{_cols[i]: lr.coef_[0][i] for i in range(len(_cols))}}

    return coefficients


def _fill_nulls(df: pd.DataFrame, coeff: dict, col: dict) -> pd.DataFrame:

    d = coeff[col]

    df = df.copy()

    _cols = [c for c in df.columns if c != col]

    f = lambda x, d: d["intercept"] + np.sum([d[c] * x[c] for c in _cols])
    df.loc[df[col].isna(), col] = df[df[col].isna()].apply(lambda x: f(x, d), axis=1)

    return df

def fill_nulls(df:pd.DataFrame, cols = List) -> pd.DataFrame:

    df = df.copy()

    coeffs = get_regression_coefficients(df[cols].dropna())

    for c in cols:
        df[cols] = _fill_nulls(df[cols], coeffs, c)

    return df




def find_null_window_size(df: pd.DataFrame, col: str):
    df = df.copy()
    df["window"] = 0

    start = (df[col].isna()) & (df[col].shift(1).notna())
    df.loc[start, "window"] = 1

    df["n_window"] = df["window"].cumsum()

    end = (df[col].notna()) & (df[col].shift(1).isna())
    df.loc[end, "window"] = -1

    df.loc[df.index.min(), "window"] = 0

    df["window"] = df["window"].cumsum()

    df.loc[df["window"] == 0, "n_window"] = None

    df["window_size"] = df.groupby("n_window")["window"].transform("count")

    return df["window_size"]


def interpolate(df: pd.DataFrame, col: str, thresh=3):
    df = df.copy()

    df["window_size"] = find_null_window_size(df, col)

    df[col] = df[col].interpolate()
    df.loc[df["window_size"] > thresh, col] = None

    df.drop("window_size", axis=1, inplace=True)

    return df





