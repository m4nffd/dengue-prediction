from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


class SMFormulaWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, formula, family):
        self.model_class = model_class
        self.formula = formula
        self.family = family

    def fit(self, X, y, test=False):
        if test:
            l = int(len(X) * 0.7)
            self.X_train = X.iloc[: l].copy()
            self.y_train = y.iloc[: l].copy()
            self.X_test = X.iloc[l :].copy()
            self.y_test = y.iloc[l :].copy()
        else:
            self.X_train = X.copy()
            self.y_train = y.copy()

        _X = pd.concat((self.X_train, self.y_train), axis=1)
        self.model_ = self.model_class(self.formula, data=_X, family=self.family)
        self.results_ = self.model_.fit()

    def predict(self, X, conf=False):
        if conf:
            pred = self.results_.get_prediction(X)
            pred_conf_int = pred.summary_frame(alpha=0.05)
            return (
                pred_conf_int["mean"],
                pred_conf_int["mean_ci_lower"],
                pred_conf_int["mean_ci_upper"],
            )
        else:
            return self.results_.predict(X)

    def get_metrics(self):
        preds = self.results_.predict(self.X_test)
        return mean_absolute_error(self.y_test, preds)

    def _get_metrics(self, X_test, y_test):

        preds = self.results_.predict(X_test)
        return mean_absolute_error(y_test, preds)




        
