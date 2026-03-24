import numpy as np


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Internal EMA helper."""
    n = len(data)
    alpha = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average.

    Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    Further reduces lag beyond DEMA by incorporating a third level of
    exponential smoothing.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    ema1 = _ema(data, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    return 3.0 * ema1 - 3.0 * ema2 + ema3
