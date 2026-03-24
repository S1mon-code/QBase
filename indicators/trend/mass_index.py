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


def mass_index(
    highs: np.ndarray,
    lows: np.ndarray,
    ema_period: int = 9,
    sum_period: int = 25,
) -> np.ndarray:
    """Mass Index (Donald Dorsey).

    Detects trend reversals by identifying "reversal bulges" in the
    high-low range.

    Calculation:
      1. Single EMA = EMA(High - Low, ema_period)
      2. Double EMA = EMA(Single EMA, ema_period)
      3. EMA Ratio  = Single EMA / Double EMA
      4. Mass Index = rolling sum of EMA Ratio over sum_period bars

    A "reversal bulge" occurs when the Mass Index rises above 27 and
    subsequently drops below 26.5, signalling a potential trend reversal.

    First sum_period-1 values are np.nan.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < sum_period:
        return out

    hl_range = highs - lows
    single_ema = _ema(hl_range, ema_period)
    double_ema = _ema(single_ema, ema_period)

    # Avoid division by zero
    ratio = np.where(double_ema != 0.0, single_ema / double_ema, 1.0)

    # Rolling sum of ratio
    cum = np.cumsum(ratio)
    out[sum_period - 1] = cum[sum_period - 1]
    out[sum_period:] = cum[sum_period:] - cum[:-sum_period]

    return out
