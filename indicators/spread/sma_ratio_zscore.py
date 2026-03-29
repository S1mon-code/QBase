import numpy as np


def sma_ratio_zscore(
    closes: np.ndarray,
    lookback: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score of price-to-SMA ratio for single-instrument mean reversion.

    Constructs a synthetic ratio: price / SMA(price, lookback), then computes
    rolling z-score of this ratio. High z-score = price far above its long-term
    average (overextended), low z-score = undervalued relative to trend.

    Also returns a structural breakout filter: the raw price z-score over the
    same window. When |price_z| is very large, mean-reversion is unreliable
    (structural regime change).

    Returns (ratio_z, price_z, ratio):
      ratio_z — z-score of price/SMA ratio
      price_z — raw price z-score (for breakout filtering)
      ratio   — raw price/SMA ratio

    Source: BlackEdge S17, inspired by cross-commodity ratio trading.
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()

    closes = closes.astype(np.float64)
    ratio_z = np.full(n, np.nan, dtype=np.float64)
    price_z = np.full(n, np.nan, dtype=np.float64)
    ratio = np.full(n, np.nan, dtype=np.float64)

    if n < lookback:
        return ratio_z, price_z, ratio

    for i in range(lookback - 1, n):
        window = closes[i - lookback + 1 : i + 1]
        sma = np.mean(window)

        if sma < 1e-9:
            continue

        # Ratio series: each price in window / sma
        ratio_series = window / sma
        ratio[i] = ratio_series[-1]

        r_mean = np.mean(ratio_series)
        r_std = np.std(ratio_series)
        if r_std > 1e-12:
            ratio_z[i] = (ratio_series[-1] - r_mean) / r_std

        # Price z-score (structural breakout filter)
        p_mean = np.mean(window)
        p_std = np.std(window)
        if p_std > 1e-12:
            price_z[i] = (closes[i] - p_mean) / p_std

    return ratio_z, price_z, ratio
