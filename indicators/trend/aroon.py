import numpy as np


def aroon(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aroon Up, Aroon Down, and Aroon Oscillator.

    Aroon Up  = 100 * (period - bars_since_highest) / period
    Aroon Down = 100 * (period - bars_since_lowest) / period
    Oscillator = Up - Down

    First `period` values are np.nan.
    """
    n = len(highs)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    aroon_osc = np.full(n, np.nan)

    if n == 0 or n <= period:
        return aroon_up, aroon_down, aroon_osc

    for i in range(period, n):
        window_high = highs[i - period : i + 1]
        window_low = lows[i - period : i + 1]

        # Bars since highest/lowest within the lookback window.
        # argmax/argmin return the first occurrence; we want the most recent,
        # so flip the window before searching.
        bars_since_high = np.argmax(window_high[::-1])
        bars_since_low = np.argmin(window_low[::-1])

        aroon_up[i] = 100.0 * (period - bars_since_high) / period
        aroon_down[i] = 100.0 * (period - bars_since_low) / period
        aroon_osc[i] = aroon_up[i] - aroon_down[i]

    return aroon_up, aroon_down, aroon_osc
