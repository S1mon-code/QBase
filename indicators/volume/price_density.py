import numpy as np


def price_density(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Price Density.

    Measures the fraction of the last ``period`` bars whose close is within
    one ATR of the current price. High density (near 1.0) indicates
    consolidation / congestion; low density (near 0.0) indicates a breakout
    or thin price action where price has moved away from recent levels.

    ATR is computed as the simple mean of true ranges over the same window.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    n = len(closes)
    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    # True range array (element 0 uses high-low only)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    for i in range(period, n):
        atr = np.mean(tr[i - period:i])
        if atr < 1e-12:
            out[i] = 1.0  # zero volatility = everything at same price
            continue
        window = closes[i - period:i]
        out[i] = float(np.sum(np.abs(window - closes[i]) <= atr)) / period

    return out
