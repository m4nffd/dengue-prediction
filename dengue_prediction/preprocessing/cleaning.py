import pandas as pd
import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LinearRegression


def get_time_diff_col(df: pd.DataFrame):

    df = df.copy()

    df["time_difference"] = pd.to_datetime(df["week_start_date"]) - pd.to_datetime(
        df.groupby("city")["week_start_date"].shift(1)
    )

    return df


def get_regression_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    coefficients = {}

    for c in df.columns:
        coefficients[c] = {}
        _cols = [_c for _c in df.columns if _c != c]
        lr = LinearRegression()
        lr.fit(df[_cols], df[[c]])
        coefficients[c]["all"] = {
            "intercept": lr.intercept_[0],
            **{_cols[i]: lr.coef_[0][i] for i in range(len(_cols))},
        }
        for cc in _cols:
            __cols = [_c for _c in _cols if _c != cc]
            lr = LinearRegression()
            lr.fit(df[__cols], df[[c]])
            coefficients[c][f"no_{cc}"] = {
                "intercept": lr.intercept_[0],
                **{__cols[i]: lr.coef_[0][i] for i in range(len(__cols))},
            }

    return coefficients


def _fill_nulls(df: pd.DataFrame, coeff: dict, col: dict) -> pd.DataFrame:

    df = df.copy()

    kk = [k for k in coeff.keys() if k != "intercept"]

    f = lambda x, d: d["intercept"] + np.sum([d[c] * x[c] for c in kk])
    df.loc[df[col].isna(), col] = df[df[col].isna()].apply(
        lambda x: f(x, coeff), axis=1
    )

    return df


def fill_nulls(df: pd.DataFrame, cols=List) -> pd.DataFrame:

    df = df.copy()

    coeffs = get_regression_coefficients(df[cols].dropna())

    for c in cols:
        for k in coeffs[c].keys():
            df[cols] = _fill_nulls(df[cols], coeffs[c][k], c)

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
