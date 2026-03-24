"""Tests for momentum indicators: ROC, RSI, MACD."""

import numpy as np
import pytest
from indicators.momentum.roc import rate_of_change
from indicators.momentum.rsi import rsi
from indicators.momentum.macd import macd


# ── ROC ──────────────────────────────────────────────────────────────────────


def test_roc_basic():
    """Hand-calculated ROC with period=3: (close / close_3_ago - 1) * 100."""
    closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = rate_of_change(closes, period=3)
    # index 3: (13/10 - 1)*100 = 30.0
    # index 4: (14/11 - 1)*100 ≈ 27.2727...
    # index 5: (15/12 - 1)*100 = 25.0
    np.testing.assert_allclose(result[3], 30.0)
    np.testing.assert_allclose(result[4], (14.0 / 11.0 - 1.0) * 100.0)
    np.testing.assert_allclose(result[5], 25.0)


def test_roc_warmup():
    """First `period` values should be np.nan."""
    closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    period = 5
    result = rate_of_change(closes, period=period)
    assert all(np.isnan(result[:period]))
    assert not np.isnan(result[period])


def test_roc_empty():
    """Empty input → empty output."""
    result = rate_of_change(np.array([]))
    assert result.size == 0


# ── RSI ──────────────────────────────────────────────────────────────────────


def test_rsi_basic():
    """RSI values should always be in [0, 100]."""
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(200)) + 100.0
    result = rsi(closes, period=14)
    valid = result[~np.isnan(result)]
    assert valid.size > 0
    assert np.all(valid >= 0.0)
    assert np.all(valid <= 100.0)


def test_rsi_warmup():
    """First `period` values are np.nan."""
    closes = np.arange(1.0, 51.0)
    period = 14
    result = rsi(closes, period=period)
    assert all(np.isnan(result[:period]))
    assert not np.isnan(result[period])


def test_rsi_all_up():
    """Monotonically increasing prices → RSI near 100."""
    closes = np.arange(1.0, 102.0)  # 1, 2, ..., 101
    result = rsi(closes, period=14)
    # All gains, no losses → RSI should be 100
    valid = result[~np.isnan(result)]
    np.testing.assert_allclose(valid, 100.0)


def test_rsi_all_down():
    """Monotonically decreasing prices → RSI near 0."""
    closes = np.arange(200.0, 99.0, -1.0)  # 200, 199, ..., 100
    result = rsi(closes, period=14)
    valid = result[~np.isnan(result)]
    np.testing.assert_allclose(valid, 0.0, atol=1e-10)


# ── MACD ─────────────────────────────────────────────────────────────────────


def test_macd_basic():
    """MACD returns a 3-tuple of arrays with correct length."""
    closes = np.cumsum(np.random.randn(100)) + 100.0
    macd_line, signal_line, histogram = macd(closes)
    assert isinstance(macd_line, np.ndarray)
    assert isinstance(signal_line, np.ndarray)
    assert isinstance(histogram, np.ndarray)
    assert len(macd_line) == len(closes)
    assert len(signal_line) == len(closes)
    assert len(histogram) == len(closes)


def test_macd_signal():
    """Signal line (EMA of MACD) should be smoother (lower std) than MACD line."""
    np.random.seed(0)
    closes = np.cumsum(np.random.randn(200)) + 100.0
    macd_line, signal_line, _ = macd(closes)

    # Compare only where both are valid
    valid = ~np.isnan(macd_line) & ~np.isnan(signal_line)
    macd_std = np.std(np.diff(macd_line[valid]))
    signal_std = np.std(np.diff(signal_line[valid]))
    assert signal_std < macd_std, "Signal line should be smoother than MACD line"
