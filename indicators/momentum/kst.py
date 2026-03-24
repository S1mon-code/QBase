import numpy as np


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average with NaN warmup."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result
    cumsum = np.cumsum(arr)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))) / period
    return result


def kst(
    closes: np.ndarray,
    roc_periods: tuple[int, ...] = (10, 15, 20, 30),
    sma_periods: tuple[int, ...] = (10, 10, 10, 15),
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Know Sure Thing (Martin Pring).

    KST = SMA(ROC1)*1 + SMA(ROC2)*2 + SMA(ROC3)*3 + SMA(ROC4)*4
    Signal = SMA(KST, signal_period)

    Weighted sum of four smoothed rate-of-change values. No fixed range.
    Returns (kst_line, signal_line).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty

    # Minimum data needed
    max_warmup = max(r + s for r, s in zip(roc_periods, sma_periods))
    if n <= max_warmup:
        nans = np.full(n, np.nan)
        return nans.copy(), nans.copy()

    weights = [1, 2, 3, 4]
    kst_line = np.zeros(n)
    all_valid = np.ones(n, dtype=bool)

    for roc_p, sma_p, w in zip(roc_periods, sma_periods, weights):
        roc = np.full(n, np.nan)
        roc[roc_p:] = (closes[roc_p:] / closes[:-roc_p] - 1.0) * 100.0

        # SMA of ROC — need to handle NaN prefix
        smoothed = np.full(n, np.nan)
        valid_start = roc_p
        roc_valid = roc[valid_start:]
        sma_valid = _sma(roc_valid, sma_p)
        smoothed[valid_start:] = sma_valid

        kst_line += np.where(np.isnan(smoothed), 0.0, smoothed * w)
        all_valid &= ~np.isnan(smoothed)

    kst_line = np.where(all_valid, kst_line, np.nan)

    # Signal line: SMA of valid KST
    first_valid = np.argmax(all_valid) if all_valid.any() else n
    sig = np.full(n, np.nan)
    if first_valid < n:
        kst_valid = kst_line[first_valid:]
        sma_sig = _sma(kst_valid, signal_period)
        sig[first_valid:] = sma_sig

    return kst_line, sig
