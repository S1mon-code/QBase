import numpy as np


def momentum_acceleration(closes: np.ndarray, fast_period: int = 10,
                          slow_period: int = 20) -> np.ndarray:
    """Momentum Acceleration — rate of change of momentum (2nd derivative).

    Computes ROC over fast_period, then takes the difference of that ROC
    over slow_period to measure whether momentum is accelerating or
    decelerating.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    warmup = fast_period + slow_period
    if n <= warmup:
        return np.full(n, np.nan)

    # First compute full ROC series
    roc = np.full(n, np.nan)
    for i in range(fast_period, n):
        ref = closes[i - fast_period]
        if abs(ref) > 0:
            roc[i] = (closes[i] - ref) / abs(ref)

    # Acceleration = ROC[i] - ROC[i - slow_period]
    result = np.full(n, np.nan)
    for i in range(warmup, n):
        if not np.isnan(roc[i]) and not np.isnan(roc[i - slow_period]):
            result[i] = roc[i] - roc[i - slow_period]

    return result
