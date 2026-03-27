"""
Shared utility functions for all_time AG strategies.

Provides optimized common computations that all strategies use:
- Fast average volume (numpy, not Python loop)
- Pre-computed tradeable mask (rollover + low volume + NaN filter)
- Standard position sizing and state management
"""

import numpy as np


def fast_avg_volume(volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling average volume using numpy convolution.

    ~100x faster than Python for-loop version.

    Args:
        volumes: Full volume array.
        window: Lookback window.

    Returns:
        Rolling mean volume array (first `window` values are NaN).
    """
    n = len(volumes)
    cumsum = np.cumsum(np.insert(volumes, 0, 0.0))
    avg = np.full(n, np.nan)
    avg[window:] = (cumsum[window:n] - cumsum[:n - window]) / window
    return avg


def compute_tradeable_mask(
    volumes: np.ndarray,
    avg_volume: np.ndarray,
    indicator_arrays: list[np.ndarray],
    vol_threshold: float = 0.1,
) -> np.ndarray:
    """Pre-compute boolean mask: True = bar is tradeable.

    Combines:
    1. Volume filter: volume >= avg_volume * vol_threshold
    2. NaN filter: all indicator arrays must be non-NaN

    Note: rollover check still done in on_bar (needs context.current_bar).

    Args:
        volumes: Full volume array.
        avg_volume: Rolling average volume array.
        indicator_arrays: List of pre-computed indicator arrays to check for NaN.
        vol_threshold: Minimum volume as fraction of average (default 0.1 = 10%).

    Returns:
        Boolean array, True = tradeable.
    """
    n = len(volumes)
    mask = np.ones(n, dtype=bool)

    # Volume filter
    valid_avg = ~np.isnan(avg_volume)
    low_vol = valid_avg & (volumes < avg_volume * vol_threshold)
    mask[low_vol] = False

    # NaN filter on all indicators
    for arr in indicator_arrays:
        if arr is not None and len(arr) == n:
            mask[np.isnan(arr)] = False

    return mask
