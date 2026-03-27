"""
Shared utility functions for all_time Iron Ore (I) strategies.

Identical to AG utils — provides optimized common computations:
- Fast average volume (numpy, not Python loop)
- Pre-computed tradeable mask (rollover + low volume + NaN filter)
"""

import numpy as np


def fast_avg_volume(volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling average volume using numpy convolution. ~100x faster."""
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
    """Pre-compute boolean mask: True = bar is tradeable."""
    n = len(volumes)
    mask = np.ones(n, dtype=bool)
    valid_avg = ~np.isnan(avg_volume)
    low_vol = valid_avg & (volumes < avg_volume * vol_threshold)
    mask[low_vol] = False
    for arr in indicator_arrays:
        if arr is not None and len(arr) == n:
            mask[np.isnan(arr)] = False
    return mask
