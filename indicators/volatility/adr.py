import numpy as np


def average_day_range(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Day Range (ADR).

    Simple rolling average of the daily (High - Low) range.

    Formula:
        ADR = SMA(High - Low, period)

    Reference: Common technical analysis metric for gauging typical
    daily price movement.
    """
    n = len(highs)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    day_range = highs - lows
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.mean(day_range[i - period + 1 : i + 1])

    return out


def adr_percent(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Day Range as a percentage of closing price.

    Formula:
        ADR% = ADR / Close * 100

    Useful for comparing volatility across instruments with different prices.
    """
    adr = average_day_range(highs, lows, period)
    n = len(closes)
    out = np.full(n, np.nan)
    valid = ~np.isnan(adr) & (closes != 0)
    out[valid] = adr[valid] / closes[valid] * 100.0
    return out
