import numpy as np


def calculate_RMSLE(y_hat, y):
    n = len(y_hat)-1
    df = y_hat[1:].values - y[:-1].values
    result = (1/n) * (df**2).sum()
    return np.sqrt(result)