import numpy as np


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.

    Uses the standard alpha = 2 / (period + 1) smoothing factor.
    Initialises with the first value; no NaN warmup since every element
    has a defined recursive value.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    alpha = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema_cross(
    data: np.ndarray,
    fast_period: int,
    slow_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EMA crossover detector.

    Returns (fast_ema, slow_ema, signal) where signal is:
      +1 = golden cross (fast crosses above slow)
      -1 = death cross  (fast crosses below slow)
       0 = no crossover
    """
    n = len(data)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)

    signal = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        prev_diff = fast_ema[i - 1] - slow_ema[i - 1]
        curr_diff = fast_ema[i] - slow_ema[i]
        if prev_diff <= 0.0 and curr_diff > 0.0:
            signal[i] = 1.0   # golden cross
        elif prev_diff >= 0.0 and curr_diff < 0.0:
            signal[i] = -1.0  # death cross

    return fast_ema, slow_ema, signal
