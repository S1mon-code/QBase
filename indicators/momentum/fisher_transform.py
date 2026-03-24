import numpy as np


def fisher_transform(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fisher Transform (John Ehlers).

    Normalizes price into a Gaussian distribution using inverse hyperbolic tangent.
    1. Midpoint = (High + Low) / 2
    2. Normalize to [-1, 1] over lookback period
    3. Apply smoothing: value = 0.33*normalized + 0.67*prev_value
    4. Clamp to (-0.999, 0.999)
    5. Fisher = 0.5 * ln((1 + value) / (1 - value))
    Trigger (signal) = previous bar's Fisher value.

    Returns (fisher_line, trigger_line).
    """
    n = highs.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    if n < period:
        nans = np.full(n, np.nan)
        return nans.copy(), nans.copy()

    midpoint = (highs + lows) / 2.0

    fisher_line = np.full(n, np.nan)
    trigger_line = np.full(n, np.nan)

    value = 0.0
    fish = 0.0

    for i in range(period - 1, n):
        window = midpoint[i - period + 1:i + 1]
        hi = np.max(window)
        lo = np.min(window)

        if hi == lo:
            normalized = 0.0
        else:
            normalized = 2.0 * (midpoint[i] - lo) / (hi - lo) - 1.0

        # Smooth the normalized value
        value = 0.33 * normalized + 0.67 * value

        # Clamp to avoid log singularity
        value = max(-0.999, min(0.999, value))

        # Fisher transform = arctanh(value)
        prev_fish = fish
        fish = 0.5 * np.log((1.0 + value) / (1.0 - value))

        fisher_line[i] = fish
        trigger_line[i] = prev_fish

    return fisher_line, trigger_line
