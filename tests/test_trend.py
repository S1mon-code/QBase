"""Unit tests for trend indicators: ADX, Supertrend, Aroon."""

import numpy as np
import pytest

from indicators.trend.adx import adx, adx_with_di
from indicators.trend.supertrend import supertrend
from indicators.trend.aroon import aroon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_ohlcv(n: int, seed: int = 42) -> tuple:
    """Generate plausible OHLCV data for *n* bars."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    volume = rng.uniform(100, 1000, n)
    return high, low, close, volume


# ===== ADX ==================================================================

class TestADX:
    def test_adx_basic(self):
        """adx() returns a valid array; non-nan values lie in [0, 100]."""
        h, l, c, _ = _random_ohlcv(100)
        result = adx(h, l, c, period=14)

        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_adx_with_di(self):
        """adx_with_di() returns a 3-tuple; DI values in [0, 100]."""
        h, l, c, _ = _random_ohlcv(100)
        adx_vals, plus_di, minus_di = adx_with_di(h, l, c, period=14)

        assert len(adx_vals) == 100
        assert len(plus_di) == 100
        assert len(minus_di) == 100

        for arr in (plus_di, minus_di):
            valid = arr[~np.isnan(arr)]
            assert len(valid) > 0
            assert np.all(valid >= 0)
            assert np.all(valid <= 100)

    def test_adx_strong_trend(self):
        """Monotonically increasing prices should produce a high ADX."""
        n = 80
        close = np.linspace(100, 200, n)
        high = close + 0.5
        low = close - 0.5

        result = adx(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # ADX should be quite high for a perfectly linear trend
        assert np.mean(valid) > 40

    def test_adx_warmup(self):
        """First ~2*period values should be np.nan."""
        period = 14
        h, l, c, _ = _random_ohlcv(100)
        result = adx(h, l, c, period=period)

        warmup = 2 * period - 1  # index where first valid ADX appears
        assert np.all(np.isnan(result[:warmup]))
        # The value at the warmup index should be valid
        assert not np.isnan(result[warmup])


# ===== Supertrend ===========================================================

class TestSupertrend:
    def test_supertrend_basic(self):
        """supertrend() returns a 2-tuple (line, direction)."""
        h, l, c, _ = _random_ohlcv(80)
        st_line, direction = supertrend(h, l, c, period=10, multiplier=3.0)

        assert isinstance(st_line, np.ndarray)
        assert isinstance(direction, np.ndarray)
        assert len(st_line) == 80
        assert len(direction) == 80

    def test_supertrend_direction(self):
        """Direction values (after warmup) are only 1 or -1."""
        h, l, c, _ = _random_ohlcv(80)
        _, direction = supertrend(h, l, c, period=10)

        valid = direction[~np.isnan(direction)]
        assert len(valid) > 0
        assert set(valid.tolist()).issubset({1.0, -1.0})

    def test_supertrend_uptrend(self):
        """Strong uptrend should produce direction mostly 1 (bullish)."""
        n = 80
        close = np.linspace(100, 200, n)
        high = close + 0.3
        low = close - 0.3

        _, direction = supertrend(high, low, close, period=10, multiplier=3.0)
        valid = direction[~np.isnan(direction)]
        bullish_pct = np.sum(valid == 1) / len(valid)
        assert bullish_pct > 0.7


# ===== Aroon ================================================================

class TestAroon:
    def test_aroon_basic(self):
        """aroon() returns a 3-tuple; up/down values in [0, 100]."""
        h, l, c, _ = _random_ohlcv(80)
        up, down, osc = aroon(h, l, period=25)

        assert len(up) == 80
        assert len(down) == 80
        assert len(osc) == 80

        for arr in (up, down):
            valid = arr[~np.isnan(arr)]
            assert len(valid) > 0
            assert np.all(valid >= 0)
            assert np.all(valid <= 100)

    def test_aroon_new_high(self):
        """If price just made the highest high, aroon_up should be 100."""
        period = 5
        # Build data where the last bar is the highest high in the window
        highs = np.array([10, 11, 12, 11, 10, 9, 8, 15.0])
        lows = highs - 1.0

        up, _, _ = aroon(highs, lows, period=period)
        # Last bar (index 7) has highest high=15 in window [3..7]
        np.testing.assert_allclose(up[-1], 100.0)

    def test_aroon_warmup(self):
        """First `period` values should be np.nan."""
        period = 25
        h, l, c, _ = _random_ohlcv(80)
        up, down, osc = aroon(h, l, period=period)

        assert np.all(np.isnan(up[:period]))
        assert np.all(np.isnan(down[:period]))
        assert np.all(np.isnan(osc[:period]))
        # Value at index `period` should be valid
        assert not np.isnan(up[period])
