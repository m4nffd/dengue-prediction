import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd


def get_clusters_correlated_variables(df: pd.DataFrame, correlation_threshold=0.9):

    df = df.copy()

    corr = df.corr()
    # If corr = 1 -> distance = 0, if corr = -1 -> distance = 2
    distances = np.round(
        1 - corr.values, 8
    )  # np.round needed ot avoid numerical fictitious negative

    distArray = ssd.squareform(distances)  # scipy converts matridf to 1d array
    hier = hierarchy.linkage(distArray, method="single")

    clusters = hierarchy.fcluster(
        hier, t=1 - correlation_threshold, criterion="distance"
    )  # t=0 is clusters of perfect correlation

    return pd.DataFrame({"feature": corr.columns, "cluster": clusters})


def define_correlated_variable_groups(
    df: pd.DataFrame, correlation_threshold: float = 0.9
):
    df = df.copy()

    clusters = get_clusters_correlated_variables(
        df, correlation_threshold=correlation_threshold
    )

    variables_dictionary = {}
    correlated = clusters.groupby("cluster")["feature"].apply(list)
    n = 1
    for group in correlated:
        if len(group) == 1:
            continue
        else:
            variables_dictionary[str(n)] = group
        n += 1

    return variables_dictionary
