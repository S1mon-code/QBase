import numpy as np


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. First `period-1` values are np.nan."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result
    result[period - 1] = arr[:period].mean()
    k = 2.0 / (period + 1.0)
    for i in range(period, arr.size):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def elder_force_index(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 13,
) -> np.ndarray:
    """Elder Force Index.

    Raw Force = (Close - Prior Close) * Volume
    EFI = EMA(Raw Force, period)

    Combines price change and volume to measure the force behind moves.
    No fixed range.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= period:
        return np.full(n, np.nan)

    # 1-period force index
    raw_force = np.diff(closes) * volumes[1:]

    # EMA smoothing
    ema_force = _ema(raw_force, period)

    # Map back to full-size array (index 0 is NaN)
    result = np.full(n, np.nan)
    result[1:] = ema_force

    return result
