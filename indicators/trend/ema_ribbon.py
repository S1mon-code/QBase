import numpy as np


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Internal EMA helper."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(data), dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema_ribbon(
    data: np.ndarray,
    periods: tuple[int, ...] = (8, 13, 21, 34, 55),
) -> list[np.ndarray]:
    """EMA Ribbon — multiple EMAs computed in parallel.

    Returns a list of np.ndarray, one per period (same order as `periods`).
    """
    if len(data) == 0:
        return [np.array([], dtype=np.float64) for _ in periods]

    return [_ema(data, p) for p in periods]


def ema_ribbon_signal(
    data: np.ndarray,
    periods: tuple[int, ...] = (8, 13, 21, 34, 55),
) -> np.ndarray:
    """EMA Ribbon alignment signal.

    Returns an array where each element is:
      +1 = bullish (all EMAs ordered fastest > slowest)
      -1 = bearish (all EMAs ordered fastest < slowest)
       0 = mixed / no clear alignment
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    ribbons = ema_ribbon(data, periods)
    signal = np.zeros(n, dtype=np.float64)

    num_pairs = len(periods) - 1

    for i in range(n):
        bull_count = 0
        bear_count = 0
        for j in range(num_pairs):
            if ribbons[j][i] > ribbons[j + 1][i]:
                bull_count += 1
            elif ribbons[j][i] < ribbons[j + 1][i]:
                bear_count += 1

        if bull_count == num_pairs:
            signal[i] = 1.0
        elif bear_count == num_pairs:
            signal[i] = -1.0

    return signal
