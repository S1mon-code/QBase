import numpy as np


def cmo(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Chande Momentum Oscillator.

    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    over the lookback period. Range [-100, 100].
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    if closes.size <= period:
        return np.full(closes.size, np.nan)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = np.full(closes.size, np.nan)

    for i in range(period, len(deltas) + 1):
        sg = gains[i - period:i].sum()
        sl = losses[i - period:i].sum()
        denom = sg + sl
        if denom == 0:
            result[i] = 0.0
        else:
            result[i] = 100.0 * (sg - sl) / denom

    return result
