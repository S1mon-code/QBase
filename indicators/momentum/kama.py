import numpy as np


def kama(closes: np.ndarray, period: int = 10,
         fast_sc: int = 2, slow_sc: int = 30) -> np.ndarray:
    """Kaufman Adaptive Moving Average.

    Adapts smoothing constant via the Efficiency Ratio (ER):
        ER  = |direction| / volatility
        SC  = (ER * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        KAMA[i] = KAMA[i-1] + SC * (price[i] - KAMA[i-1])
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < period + 1:
        return np.full(n, np.nan)

    fast_alpha = 2.0 / (fast_sc + 1.0)
    slow_alpha = 2.0 / (slow_sc + 1.0)

    result = np.full(n, np.nan)
    result[period] = closes[period]

    for i in range(period + 1, n):
        direction = abs(closes[i] - closes[i - period])
        volatility = np.sum(np.abs(np.diff(closes[i - period:i + 1])))
        er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        result[i] = result[i - 1] + sc * (closes[i] - result[i - 1])

    return result
