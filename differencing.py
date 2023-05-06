import numpy as np
import statsmodels.api as sm
from functools import partial


def _detrend(x, lo_frac=0.6, lo_delta=0.01, return_trend=False):
    # use some existing pieces of statsmodels
    lowess = sm.nonparametric.lowess
    # get plain np array
    observed = np.asanyarray(x).squeeze()
    # calc trend, remove from observation
    trend = lowess(
        observed,
        [x for x in range(len(observed))],
        frac=lo_frac,
        delta=lo_delta * len(observed),
        return_sorted=False,
    )
    detrended = observed - trend
    return detrended, trend if return_trend else detrended

def make_stationary(x: np.ndarray, method: str="detrend", detrend_kwargs:dict={}):
    """Utility to make time series stationary
    Args:
        x (np.ndarray): The time series array to be made stationary
        method (str, optional): {"detrend","logdiff"}. Defaults to "detrend".
        detrend_kwargs (dict, optional): These kwargs will be passed on to the detrend method
    """
    if method=="detrend":
        detrend_kwargs["return_trend"] = True
        stationary, trend = _detrend(x, **detrend_kwargs)
        def inverse_transform(st, trend):
            return st+trend
        return stationary, partial(inverse_transform, trend=trend)
    elif method == "logdiff":
        stationary = np.log(x[:-1]/x[1:])
        def inverse_transform(st, x):
            _x = np.exp(st)
            return _x*x[1:]
        return stationary, partial(inverse_transform, x=x)