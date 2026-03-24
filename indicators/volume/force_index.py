import numpy as np


def force_index(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 13,
) -> np.ndarray:
    """Force Index (Alexander Elder).

    Combines price change, direction, and volume:
      Force(1) = (Close - prev_Close) * Volume
      Force(period) = EMA(Force(1), period)

    First ``period`` values are np.nan.

    Source: StockCharts ChartSchool / Elder, *Trading for a Living*.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return result

    # 1-period force index (length n-1, starting at index 1)
    fi1 = np.diff(closes) * volumes[1:]

    if len(fi1) < period:
        return result

    # EMA of fi1 with SMA seed
    alpha = 2.0 / (period + 1)
    ema = np.mean(fi1[:period])
    result[period] = ema

    for i in range(period, len(fi1)):
        ema = alpha * fi1[i] + (1.0 - alpha) * ema
        result[i + 1] = ema

    return result
