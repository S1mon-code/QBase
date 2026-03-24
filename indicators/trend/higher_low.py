import numpy as np


def _find_swing_lows(lows: np.ndarray, lookback: int) -> list[tuple[int, float]]:
    """Detect swing lows — a bar whose low is strictly less than the
    `lookback` bars on each side."""
    n = len(lows)
    swings: list[tuple[int, float]] = []
    for i in range(lookback, n - lookback):
        is_swing = True
        for offset in range(1, lookback + 1):
            if lows[i - offset] <= lows[i] or lows[i + offset] <= lows[i]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, float(lows[i])))
    return swings


def _find_swing_highs(highs: np.ndarray, lookback: int) -> list[tuple[int, float]]:
    """Detect swing highs — a bar whose high is strictly greater than the
    `lookback` bars on each side."""
    n = len(highs)
    swings: list[tuple[int, float]] = []
    for i in range(lookback, n - lookback):
        is_swing = True
        for offset in range(1, lookback + 1):
            if highs[i - offset] >= highs[i] or highs[i + offset] >= highs[i]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, float(highs[i])))
    return swings


def higher_lows(
    lows: np.ndarray,
    lookback: int = 4,
) -> np.ndarray:
    """Count of consecutive higher swing lows at each bar.

    Uses `lookback` bars on each side to confirm a swing low.
    The value at each bar is the length of the current higher-low chain
    that has been confirmed up to (and including) that bar.
    Bars before any swing low are 0.
    """
    n = len(lows)
    if n == 0:
        return np.array([], dtype=np.float64)

    swings = _find_swing_lows(lows, lookback)
    out = np.zeros(n, dtype=np.float64)

    if not swings:
        return out

    # Build chain lengths for each swing point
    chain_lengths: list[tuple[int, int]] = []  # (bar_index, chain_count)
    chain = 1
    chain_lengths.append((swings[0][0], chain))
    for k in range(1, len(swings)):
        if swings[k][1] > swings[k - 1][1]:
            chain += 1
        else:
            chain = 1
        chain_lengths.append((swings[k][0], chain))

    # Forward-fill chain count across bars
    cl_idx = 0
    current_count = 0.0
    for i in range(n):
        if cl_idx < len(chain_lengths) and i >= chain_lengths[cl_idx][0]:
            current_count = float(chain_lengths[cl_idx][1])
            cl_idx += 1
        out[i] = current_count

    return out


def lower_highs(
    highs: np.ndarray,
    lookback: int = 4,
) -> np.ndarray:
    """Count of consecutive lower swing highs at each bar.

    Mirror of `higher_lows` — counts how many successive swing highs
    are each lower than the previous.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=np.float64)

    swings = _find_swing_highs(highs, lookback)
    out = np.zeros(n, dtype=np.float64)

    if not swings:
        return out

    chain_lengths: list[tuple[int, int]] = []
    chain = 1
    chain_lengths.append((swings[0][0], chain))
    for k in range(1, len(swings)):
        if swings[k][1] < swings[k - 1][1]:
            chain += 1
        else:
            chain = 1
        chain_lengths.append((swings[k][0], chain))

    cl_idx = 0
    current_count = 0.0
    for i in range(n):
        if cl_idx < len(chain_lengths) and i >= chain_lengths[cl_idx][0]:
            current_count = float(chain_lengths[cl_idx][1])
            cl_idx += 1
        out[i] = current_count

    return out
