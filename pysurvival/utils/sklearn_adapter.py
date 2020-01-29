"""
Inspired by module `lifelines.utils.sklearn_adapter` in lifelines (https://github.com/CamDavidsonPilon/lifelines)
"""
import inspect
import pandas as pd
import warnings

try:
    from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin
except ImportError:
    raise ImportError("scikit-learn must be installed on the local system to use this utility class.")
from .metrics import concordance_index

__all__ = ["sklearn_adapter"]


def filter_kwargs(f, kwargs):
    s = inspect.signature(f)
    res = {k: kwargs[k] for k in s.parameters if k in kwargs}
    return res


class _SklearnModel(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    def __init__(self, **kwargs):
        self._params = kwargs
        self.pysurvival_model = self.pysurvival_model(**filter_kwargs(self.pysurvival_model.__init__, self._params))

    @property
    def _yColumn(self):
        return self._time_col

    @property
    def _eventColumn(self):
        return self._event_col

    def fit(self, X, Y, **kwargs):
        """

        Parameters
        -----------
        X: Predictors
            DataFrame or numpy array
        Y: Time and event data
            DataFrame or numpy array
        """
        X = X.copy()
        Y = Y.copy()

        if isinstance(Y, pd.DataFrame):
            if self._eventColumn not in Y.columns.values:
                raise ValueError("Event data (E) missing in Y DataFrame")
        else:
            warnings.warn("Y is not a DataFrame. Will assume event data is in its first column.")
            Y = pd.DataFrame(Y, columns=[self._eventColumn, self._yColumn])

        E = Y[self._eventColumn].copy()
        Y.drop(self._eventColumn, axis=1, inplace=True)

        fit = getattr(self.pysurvival_model, self._fit_method)

        for k in self._params.keys():
            if k not in kwargs.keys():
                kwargs[k] = self._params[k]

        self.pysurvival_model = fit(X=X, T=Y.values.flatten(), E=E.values.flatten(), **filter_kwargs(fit, kwargs))
        return self

    def set_params(self, **params):
        for key, value in params.items():
            self._params[key] = value
        return self

    def get_params(self, deep=True):
        out = {}
        for name, p in inspect.signature(self.pysurvival_model.__init__).parameters.items():
            if p.kind < 4:  # ignore kwargs
                out[name] = getattr(self.pysurvival_model, name)
        return out

    def predict(self, X, **kwargs):
        """
        Parameters
        ------------
        X: Predictors
            DataFrame or numpy array
        """
        X = X.copy()

        predictions = getattr(self.pysurvival_model, self._predict_method)(X, **kwargs).squeeze()
        return predictions

    def score(self, X, Y, **kwargs):
        """

        Parameters
        -----------
        X: Predictors
            DataFrame or numpy array
        Y: Time and event data
            DataFrame or numpy array
        """
        X = X.copy()
        Y = Y.copy()

        if isinstance(Y, pd.DataFrame):
            if self._eventColumn not in Y.columns.values:
                raise ValueError("Event data (E) missing in Y DataFrame")
        else:
            warnings.warn("Y is not a DataFrame. Will assume event data is in its first column.")
            Y = pd.DataFrame(Y, columns=[self._eventColumn, self._yColumn])

        X = X.copy()
        E = Y.loc[:, self._eventColumn].copy()
        Y.drop(self._eventColumn, axis=1, inplace=True)

        if callable(self._scoring_method):
            res = self._scoring_method(self.pysurvival_model, X=X, T=Y.values.flatten(), E=E.values.flatten(), **kwargs)
        else:
            raise ValueError("Error calculating score")
        return res


def sklearn_adapter(fitter, time_col, event_col, predict_method="predict_survival",
                    scoring_method=concordance_index):
    """
    This function wraps pysurvival models into a scikit-learn compatible API. The function returns a
    class that can be instantiated with parameters (similar to a scikit-learn class).

    Parameters
    ----------

    fitter: class
        The class (not an instance) to be wrapper. Example: ``CoxPHModel``
    time_col: string
        The column in your DataFrame that represents the time column
    event_col: string
        The column in your DataFrame that represents the event column
    predict_method: string
        Can be the string ``"predict_survival", "predict_hazard"``
    scoring_method: function
        Provide a way to produce a ``score`` on the scikit-learn model. Signature should look like (predictors, durations, event_observed)

    """
    name = "SkLearn" + fitter.__name__
    klass = type(
        name,
        (_SklearnModel,),
        {
            "pysurvival_model": fitter,
            "_time_col": time_col,
            "_event_col": event_col,
            "_predict_method": predict_method,
            "_fit_method": "fit",
            "_scoring_method": staticmethod(scoring_method),
        },
    )
    globals()[klass.__name__] = klass
    return klass
