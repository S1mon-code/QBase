import numpy as np


def parkinson(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Parkinson Volatility Estimator (1980).

    Uses High-Low range; more efficient than close-to-close (5.2x).

    Formula:
        sigma = sqrt( (1 / (4 * n * ln(2))) * sum( ln(H_i / L_i)^2 ) )

    Reference: Parkinson, M. (1980), "The Extreme Value Method for
    Estimating the Variance of the Rate of Return," Journal of Business.
    """
    n = len(highs)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    log_hl_sq = np.log(highs / lows) ** 2

    out = np.full(n, np.nan)
    factor = 1.0 / (4.0 * np.log(2.0))

    for i in range(period - 1, n):
        window = log_hl_sq[i - period + 1 : i + 1]
        variance = factor * np.mean(window)
        out[i] = np.sqrt(max(variance, 0.0))

    return out
