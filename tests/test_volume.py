"""Unit tests for volume indicators: OBV, MFI, VWAP."""

import numpy as np
import pytest

from indicators.volume.obv import obv
from indicators.volume.mfi import mfi
from indicators.volume.vwap import vwap


# ===== OBV ==================================================================

class TestOBV:
    def test_obv_basic(self):
        """Known input: verify cumulative logic step by step."""
        closes = np.array([10.0, 11.0, 10.5, 12.0, 11.0])
        volumes = np.array([100.0, 200.0, 150.0, 300.0, 250.0])

        result = obv(closes, volumes)

        # Bar 0: direction=0 but code sets directed_volume[0]=volumes[0]=100
        # Bar 1: close up   → +200  → cumsum = 300
        # Bar 2: close down → -150  → cumsum = 150
        # Bar 3: close up   → +300  → cumsum = 450
        # Bar 4: close down → -250  → cumsum = 200
        expected = np.array([100.0, 300.0, 150.0, 450.0, 200.0])
        np.testing.assert_allclose(result, expected)

    def test_obv_up(self):
        """Rising prices should make OBV increase overall."""
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        result = obv(closes, volumes)
        # Each bar after bar 0 adds +100
        assert result[-1] > result[0]
        # OBV should be monotonically non-decreasing
        assert np.all(np.diff(result) >= 0)

    def test_obv_empty(self):
        """Empty arrays should return an empty result."""
        result = obv(np.array([]), np.array([]))
        assert len(result) == 0


# ===== MFI ==================================================================

class TestMFI:
    def test_mfi_basic(self):
        """MFI returns values in [0, 100] after warmup."""
        rng = np.random.default_rng(42)
        n = 60
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        vol = rng.uniform(100, 1000, n)

        result = mfi(high, low, close, vol, period=14)
        assert len(result) == n

        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_mfi_warmup(self):
        """First `period` values should be np.nan."""
        period = 14
        rng = np.random.default_rng(7)
        n = 60
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        vol = rng.uniform(100, 1000, n)

        result = mfi(high, low, close, vol, period=period)
        assert np.all(np.isnan(result[:period]))
        assert not np.isnan(result[period])

    def test_mfi_empty(self):
        """Empty arrays should return an empty result."""
        empty = np.array([])
        result = mfi(empty, empty, empty, empty, period=14)
        assert len(result) == 0


# ===== VWAP =================================================================

class TestVWAP:
    def test_vwap_basic(self):
        """Known input: verify TP * Vol / cumVol calculation."""
        highs = np.array([12.0, 13.0, 14.0])
        lows = np.array([10.0, 11.0, 12.0])
        closes = np.array([11.0, 12.0, 13.0])
        volumes = np.array([100.0, 200.0, 300.0])

        result = vwap(highs, lows, closes, volumes)

        # TP = (H+L+C)/3 = [11.0, 12.0, 13.0]
        tp = np.array([11.0, 12.0, 13.0])
        cum_tp_vol = np.cumsum(tp * volumes)   # [1100, 3500, 7400]
        cum_vol = np.cumsum(volumes)            # [100, 300, 600]
        expected = cum_tp_vol / cum_vol         # [11.0, 11.6667, 12.3333]

        np.testing.assert_allclose(result, expected)

    def test_vwap_equal_volumes(self):
        """If all volumes are equal, VWAP = cumulative mean of typical price."""
        highs = np.array([12.0, 15.0, 18.0, 21.0])
        lows = np.array([10.0, 13.0, 16.0, 19.0])
        closes = np.array([11.0, 14.0, 17.0, 20.0])
        volumes = np.array([50.0, 50.0, 50.0, 50.0])

        result = vwap(highs, lows, closes, volumes)

        tp = (highs + lows + closes) / 3.0
        expected = np.cumsum(tp) / np.arange(1, len(tp) + 1)

        np.testing.assert_allclose(result, expected)

    def test_vwap_empty(self):
        """Empty arrays should return an empty result."""
        empty = np.array([])
        result = vwap(empty, empty, empty, empty)
        assert len(result) == 0
