import numpy as np


def ulcer_index(
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Ulcer Index (Peter Martin, 1987).

    Measures downside risk by computing the depth of percentage drawdowns
    from the highest close over a rolling period.

    Steps:
        1. pct_drawdown = (Close - Max(Close, period)) / Max(Close, period) * 100
        2. UI = sqrt(mean(pct_drawdown^2, period))

    Higher values indicate deeper/longer drawdowns.

    Reference: Peter Martin & Byron McCann, "The Investor's Guide to
    Fidelity Funds" (1989).
    """
    n = len(closes)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        rolling_max = np.maximum.accumulate(window)
        pct_dd = (window - rolling_max) / rolling_max * 100.0
        out[i] = np.sqrt(np.mean(pct_dd ** 2))

    return out
