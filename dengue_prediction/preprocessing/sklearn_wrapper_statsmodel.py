from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np


class SMFormulaWrapper(BaseEstimator, RegressorMixin):


    def __init__(self, model_class, formula, family):
        self.model_class = model_class
        self.formula = formula
        self.family = family
    def fit(self, X, y):
        _X = pd.concat((X, y), axis=1)
        self.model_ = self.model_class(self.formula, data=_X, family=self.family)
        self.results_ = self.model_.fit()

    def predict(self, X, conf=False):
        if conf:
            pred = self.results_.get_prediction(X)
            pred_conf_int = pred.summary_frame(alpha=0.05)
            return pred_conf_int['mean'], pred_conf_int['mean_ci_lower'], pred_conf_int['mean_ci_upper']
        else:
            return self.results_.predict(X)
