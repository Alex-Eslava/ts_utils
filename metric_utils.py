import numpy as np

def mae(actuals, predictions):
    return np.nanmean(np.abs(actuals-predictions))

def mse(actuals, predictions):
    return np.nanmean(np.power(actuals-predictions, 2))

def mape(actuals, predictions):
    return np.nanmean(np.abs((actuals - predictions) / actuals)) * 100

def smape(actuals, predictions):
    return np.nanmean(2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions))) * 100

def mase(actuals, predictions, seasonal_period=None):
    diff = np.abs(actuals - predictions)
    if seasonal_period is None:
        scale = np.nanmean(np.abs(np.diff(actuals, n=1)))
    else:
        scale = np.nanmean(np.abs(np.diff(actuals, n=seasonal_period)))
    return np.nanmean(diff / scale)

