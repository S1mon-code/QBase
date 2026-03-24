import numpy as np


def oi_divergence(
    closes: np.ndarray, oi: np.ndarray, period: int = 20
) -> np.ndarray:
    """Open Interest Divergence score.

    Compares the direction of price change vs OI change over a rolling window.
    Positive score = bullish divergence (price down + OI down = long liquidation).
    Negative score = bearish divergence (price up + OI down = short covering).
    Score is normalised to [-1, 1] by dividing by the period.
    """
    if closes.size == 0 or oi.size == 0:
        return np.array([], dtype=np.float64)

    n = len(closes)
    closes = closes.astype(np.float64)
    oi = oi.astype(np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(period + 1, n):
        bullish = 0
        bearish = 0
        for j in range(i - period, i):
            price_chg = closes[j] - closes[j - 1]
            oi_chg = oi[j] - oi[j - 1]
            if price_chg > 0 and oi_chg < 0:
                bearish += 1  # price up + OI down = short covering
            elif price_chg < 0 and oi_chg < 0:
                bullish += 1  # price down + OI down = long liquidation
        # Positive = bullish divergence, negative = bearish divergence
        out[i] = (bullish - bearish) / period

    return out
