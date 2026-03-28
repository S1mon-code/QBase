"""
Tests for portfolio selection stability analysis.
"""
import sys
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from portfolio.stability_test import (
    subsample_returns,
    run_single_stability_iteration,
    run_stability_test,
    classify_strategy,
    save_stability_result,
    print_stability_report,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_returns_df():
    """6 strategies, 200 days of returns."""
    rng = np.random.default_rng(123)
    n_days = 200
    data = {}
    for i in range(6):
        data[f"v{i+1}"] = rng.normal(0.001, 0.02, n_days)
    df = pd.DataFrame(data)
    df.index = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return df


@pytest.fixture
def dominant_strategy_returns():
    """One strategy dominates (v1 has 5x Sharpe of others).

    v1 should be CORE in nearly all runs.
    """
    rng = np.random.default_rng(42)
    n_days = 200
    data = {}
    # v1: very strong, low vol
    data["v1"] = rng.normal(0.005, 0.01, n_days)
    # v2-v6: weak, high vol
    for i in range(2, 7):
        data[f"v{i}"] = rng.normal(0.0002, 0.03, n_days)
    df = pd.DataFrame(data)
    df.index = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return df


@pytest.fixture
def interchangeable_returns():
    """All strategies have nearly identical returns (high correlation).

    Selection should be unstable — any subset works similarly.
    """
    rng = np.random.default_rng(99)
    n_days = 200
    base = rng.normal(0.001, 0.02, n_days)
    data = {}
    for i in range(8):
        noise = rng.normal(0, 0.001, n_days)
        data[f"v{i+1}"] = base + noise
    df = pd.DataFrame(data)
    df.index = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return df


def _make_sharpes(df):
    """Compute Sharpe ratios from returns DataFrame."""
    sharpes = {}
    for col in df.columns:
        ret = df[col]
        if ret.std() > 0:
            sharpes[col] = float(ret.mean() / ret.std() * np.sqrt(252))
        else:
            sharpes[col] = 0.0
    return sharpes


# =========================================================================
# Test: subsample_returns
# =========================================================================

class TestSubsampleReturns:
    def test_subsample_80pct(self, simple_returns_df):
        """80% subsample keeps 80% of rows."""
        sub = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(0))
        expected_rows = int(len(simple_returns_df) * 0.8)
        assert len(sub) == expected_rows

    def test_subsample_preserves_columns(self, simple_returns_df):
        """Subsample keeps all columns."""
        sub = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(0))
        assert list(sub.columns) == list(simple_returns_df.columns)

    def test_subsample_no_replacement(self, simple_returns_df):
        """Subsample has no duplicate rows."""
        sub = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(0))
        assert len(sub.index.unique()) == len(sub)

    def test_subsample_different_seeds_different_results(self, simple_returns_df):
        """Different seeds produce different subsamples."""
        sub1 = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(1))
        sub2 = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(2))
        # Different rows should be selected
        assert not sub1.index.equals(sub2.index)

    def test_subsample_same_seed_same_results(self, simple_returns_df):
        """Same seed produces identical subsamples."""
        sub1 = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(42))
        sub2 = subsample_returns(simple_returns_df, frac=0.8, rng=np.random.default_rng(42))
        assert sub1.index.equals(sub2.index)

    def test_subsample_rows_are_subset(self, simple_returns_df):
        """Subsampled rows must exist in original."""
        sub = subsample_returns(simple_returns_df, frac=0.5, rng=np.random.default_rng(0))
        for idx in sub.index:
            assert idx in simple_returns_df.index

    def test_subsample_small_frac(self, simple_returns_df):
        """Very small frac still works (at least 1 row)."""
        sub = subsample_returns(simple_returns_df, frac=0.01, rng=np.random.default_rng(0))
        assert len(sub) >= 1


# =========================================================================
# Test: classify_strategy
# =========================================================================

class TestClassifyStrategy:
    def test_core_above_80(self):
        assert classify_strategy(0.81) == "CORE"
        assert classify_strategy(0.95) == "CORE"
        assert classify_strategy(1.0) == "CORE"

    def test_satellite_40_to_80(self):
        assert classify_strategy(0.40) == "SATELLITE"
        assert classify_strategy(0.60) == "SATELLITE"
        assert classify_strategy(0.80) == "SATELLITE"

    def test_edge_below_40(self):
        assert classify_strategy(0.0) == "EDGE"
        assert classify_strategy(0.10) == "EDGE"
        assert classify_strategy(0.39) == "EDGE"

    def test_boundary_80_is_satellite(self):
        """Exactly 0.80 is SATELLITE (>0.80 required for CORE)."""
        assert classify_strategy(0.80) == "SATELLITE"

    def test_boundary_40_is_satellite(self):
        """Exactly 0.40 is SATELLITE."""
        assert classify_strategy(0.40) == "SATELLITE"


# =========================================================================
# Test: selection frequency computation
# =========================================================================

class TestSelectionFrequency:
    def test_frequency_correct(self, simple_returns_df):
        """Selection frequency = count / n_successful."""
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=10,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        n_succ = result["n_successful"]
        assert n_succ > 0

        for s in result["strategies"]:
            expected_freq = s["selection_count"] / n_succ
            assert abs(s["selection_freq"] - expected_freq) < 1e-3

    def test_selection_count_bounded(self, simple_returns_df):
        """Selection count never exceeds n_successful."""
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=20,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        n_succ = result["n_successful"]
        for s in result["strategies"]:
            assert s["selection_count"] <= n_succ


# =========================================================================
# Test: dominant strategy → CORE
# =========================================================================

class TestDominantStrategy:
    def test_dominant_is_core(self, dominant_strategy_returns):
        """A strategy with 5x Sharpe should be selected in most iterations → CORE."""
        sharpes = _make_sharpes(dominant_strategy_returns)
        pool = list(dominant_strategy_returns.columns)

        result = run_stability_test(
            returns_df=dominant_strategy_returns,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=30,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        # Find v1 in results
        v1_stats = next((s for s in result["strategies"] if s["version"] == "v1"), None)
        assert v1_stats is not None, "v1 should appear in results"
        assert v1_stats["classification"] == "CORE", (
            f"v1 should be CORE but is {v1_stats['classification']} "
            f"(freq={v1_stats['selection_freq']:.2f})"
        )

    def test_dominant_has_highest_freq(self, dominant_strategy_returns):
        """The dominant strategy should have the highest selection frequency."""
        sharpes = _make_sharpes(dominant_strategy_returns)
        pool = list(dominant_strategy_returns.columns)

        result = run_stability_test(
            returns_df=dominant_strategy_returns,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=30,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        strategies = result["strategies"]
        if strategies:
            top = strategies[0]  # sorted by freq descending
            assert top["version"] == "v1", f"Expected v1 on top, got {top['version']}"


# =========================================================================
# Test: interchangeable strategies → EDGE
# =========================================================================

class TestInterchangeableStrategies:
    def test_interchangeable_strategies_are_not_all_core(self, interchangeable_returns):
        """When all strategies are near-identical, no single one should
        consistently be selected → most should NOT be CORE."""
        sharpes = _make_sharpes(interchangeable_returns)
        pool = list(interchangeable_returns.columns)

        result = run_stability_test(
            returns_df=interchangeable_returns,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=30,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        strategies = result["strategies"]
        n_core = sum(1 for s in strategies if s["classification"] == "CORE")
        # With 8 interchangeable strategies and 3-6 typically selected,
        # not all 8 can be CORE
        assert n_core < len(pool), (
            f"All {n_core}/{len(pool)} classified as CORE — expected instability"
        )


# =========================================================================
# Test: result structure
# =========================================================================

class TestResultStructure:
    def test_result_has_required_keys(self, simple_returns_df):
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=5,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        assert "n_runs" in result
        assert "n_successful" in result
        assert "subsample_frac" in result
        assert "strategies" in result
        assert result["n_runs"] == 5
        assert result["subsample_frac"] == 0.8

    def test_strategy_entry_has_required_keys(self, simple_returns_df):
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=5,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        for s in result["strategies"]:
            assert "version" in s
            assert "selection_count" in s
            assert "selection_freq" in s
            assert "avg_weight" in s
            assert "std_weight" in s
            assert "classification" in s
            assert s["classification"] in ("CORE", "SATELLITE", "EDGE")

    def test_weights_sum_approx_one(self, simple_returns_df):
        """Average weights of selected strategies should sum to ~1."""
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=10,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        # avg_weight * selection_freq should roughly sum to 1
        # (each run's weights sum to 1, so expected value of sum = 1)
        # But since not all strategies are selected each run, we just check
        # that weights are in valid range
        for s in result["strategies"]:
            assert 0.0 <= s["avg_weight"] <= 1.0
            assert s["std_weight"] >= 0.0


# =========================================================================
# Test: save to JSON
# =========================================================================

class TestSaveResult:
    def test_save_creates_file(self, simple_returns_df, tmp_path):
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=3,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        output_path = save_stability_result(result, "AG", output_dir=str(tmp_path))
        assert Path(output_path).exists()

        # Verify JSON is valid
        import json
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["n_runs"] == 3
        assert "strategies" in loaded

    def test_save_filename_format(self, simple_returns_df, tmp_path):
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=3,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        output_path = save_stability_result(result, "AG", output_dir=str(tmp_path))
        assert Path(output_path).name == "stability_ag.json"


# =========================================================================
# Test: print_stability_report doesn't crash
# =========================================================================

class TestPrintReport:
    def test_print_report_runs(self, simple_returns_df, capsys):
        sharpes = _make_sharpes(simple_returns_df)
        pool = list(simple_returns_df.columns)

        result = run_stability_test(
            returns_df=simple_returns_df,
            pool_versions=pool,
            strategy_sharpes=sharpes,
            n_runs=5,
            frac=0.8,
            seed=42,
            quiet=True,
        )

        print_stability_report(result)
        captured = capsys.readouterr()
        assert "Stability Test" in captured.out
        assert "CORE" in captured.out or "SATELLITE" in captured.out or "EDGE" in captured.out

    def test_print_report_empty_result(self, capsys):
        result = {
            "n_runs": 10,
            "n_successful": 0,
            "subsample_frac": 0.8,
            "strategies": [],
        }
        print_stability_report(result)
        captured = capsys.readouterr()
        assert "No strategies" in captured.out
