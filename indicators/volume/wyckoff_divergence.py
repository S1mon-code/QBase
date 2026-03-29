import numpy as np


def wyckoff_divergence(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wyckoff volume-price divergence detector.

    Identifies when price makes a new extreme but the Accumulation/Distribution
    line does not, signaling institutional money flow against the price trend.

    Bullish divergence: price at lookback-period low, but A/D line NOT at low
      → institutions accumulating despite falling price.
    Bearish divergence: price at lookback-period high, but A/D line NOT at high
      → institutions distributing despite rising price.

    Returns (bullish_div, bearish_div, ad_line):
      bullish_div — 1.0 on bullish divergence bars, else 0.0
      bearish_div — 1.0 on bearish divergence bars, else 0.0
      ad_line     — cumulative Accumulation/Distribution line
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    bullish_div = np.zeros(n, dtype=np.float64)
    bearish_div = np.zeros(n, dtype=np.float64)

    # Compute A/D line
    hl_range = highs - lows
    clv = np.where(hl_range > 1e-12,
                   ((closes - lows) - (highs - closes)) / hl_range,
                   0.0)
    ad_values = clv * volumes
    ad_line = np.cumsum(ad_values)

    # Detect divergences
    for i in range(lookback, n):
        window_closes = closes[i - lookback + 1 : i + 1]
        window_ad = ad_line[i - lookback + 1 : i + 1]

        price_at_low = closes[i] <= np.min(window_closes)
        price_at_high = closes[i] >= np.max(window_closes)
        ad_at_low = ad_line[i] <= np.min(window_ad)
        ad_at_high = ad_line[i] >= np.max(window_ad)

        # Bullish: price at low but A/D not at low
        if price_at_low and not ad_at_low:
            bullish_div[i] = 1.0

        # Bearish: price at high but A/D not at high
        if price_at_high and not ad_at_high:
            bearish_div[i] = 1.0

    return bullish_div, bearish_div, ad_line
