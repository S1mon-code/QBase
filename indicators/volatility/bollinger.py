import numpy as np


def bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (upper, middle, lower).

    Middle = SMA, bands = middle +/- num_std * rolling std (ddof=1).
    First `period - 1` values are np.nan.
    """
    n = len(closes)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    if n < period:
        return upper, middle, lower

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=1)
        middle[i] = sma
        upper[i] = sma + num_std * std
        lower[i] = sma - num_std * std

    return upper, middle, lower


def bollinger_width(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> np.ndarray:
    """Bollinger Band Width: (upper - lower) / middle."""
    upper, middle, lower = bollinger_bands(closes, period, num_std)
    with np.errstate(divide="ignore", invalid="ignore"):
        width = (upper - lower) / middle
    return width
