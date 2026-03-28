"""Tests for the all-time failure analysis script."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

QBASE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(QBASE_ROOT))

from strategies.all_time.ag.analyze_failures import (
    build_strategy_records,
    get_categories,
    get_category,
    group_by_category,
    analyze_failure_patterns,
    generate_recommendations,
    load_results,
    run_analysis,
    version_to_int,
    CATEGORIES_100,
    CATEGORIES_200,
)


# =====================================================================
# 1. Category Grouping Tests
# =====================================================================

class TestCategoryGrouping:

    def test_version_to_int(self):
        assert version_to_int("v1") == 1
        assert version_to_int("v100") == 100
        assert version_to_int("v200") == 200

    def test_get_categories_100(self):
        cats = get_categories(100)
        assert cats == CATEGORIES_100

    def test_get_categories_200(self):
        cats = get_categories(200)
        assert cats == CATEGORIES_200
        cats = get_categories(150)
        assert cats == CATEGORIES_200

    def test_get_category_100(self):
        cats = CATEGORIES_100
        assert get_category(1, cats) == "Trend"
        assert get_category(20, cats) == "Trend"
        assert get_category(21, cats) == "Mean Reversion"
        assert get_category(40, cats) == "Mean Reversion"
        assert get_category(41, cats) == "Breakout"
        assert get_category(61, cats) == "Multi-TF"
        assert get_category(81, cats) == "Adaptive"
        assert get_category(100, cats) == "Adaptive"

    def test_get_category_200(self):
        cats = CATEGORIES_200
        assert get_category(1, cats) == "Trend"
        assert get_category(40, cats) == "Trend"
        assert get_category(41, cats) == "Mean Reversion"
        assert get_category(80, cats) == "Mean Reversion"
        assert get_category(81, cats) == "Breakout"
        assert get_category(121, cats) == "Multi-TF"
        assert get_category(161, cats) == "Adaptive"
        assert get_category(200, cats) == "Adaptive"

    def test_get_category_unknown(self):
        assert get_category(999, CATEGORIES_100) == "Unknown"

    def test_group_by_category(self):
        records = [
            {"version": 1, "sharpe": 0.5},
            {"version": 5, "sharpe": -0.1},
            {"version": 25, "sharpe": 0.3},
        ]
        groups = group_by_category(records, CATEGORIES_100)
        assert len(groups["Trend"]) == 2
        assert len(groups["Mean Reversion"]) == 1


# =====================================================================
# 2. Record Building Tests
# =====================================================================

class TestBuildRecords:

    def test_opt_only(self):
        opt_data = [
            {"version": "v1", "best_sharpe": 0.5, "best_params": {"atr_stop_mult": 3.0}, "freq": "daily"},
            {"version": "v2", "best_sharpe": -0.1, "best_params": {"atr_stop_mult": 2.0}, "freq": "4h"},
        ]
        records = build_strategy_records(opt_data, None)
        assert len(records) == 2
        # Without test data, sharpe should fall back to opt_sharpe
        r1 = [r for r in records if r["version"] == 1][0]
        assert r1["sharpe"] == 0.5

    def test_test_only(self):
        test_data = [
            {"version": "v1", "sharpe": 0.8, "total_return": 0.5, "max_drawdown": 0.1,
             "n_trades": 50, "calmar": 0.9, "freq": "daily"},
        ]
        records = build_strategy_records(None, test_data)
        assert len(records) == 1
        assert records[0]["sharpe"] == 0.8
        assert records[0]["n_trades"] == 50

    def test_merged(self):
        opt_data = [
            {"version": "v1", "best_sharpe": 0.5, "best_params": {"atr_stop_mult": 3.0}, "freq": "daily"},
        ]
        test_data = [
            {"version": "v1", "sharpe": 0.3, "total_return": 0.2, "max_drawdown": 0.15,
             "n_trades": 40, "calmar": 0.5, "freq": "daily"},
        ]
        records = build_strategy_records(opt_data, test_data)
        assert len(records) == 1
        # Test sharpe should override opt sharpe
        assert records[0]["sharpe"] == 0.3
        assert records[0]["opt_sharpe"] == 0.5
        assert records[0]["best_params"]["atr_stop_mult"] == 3.0

    def test_empty_data(self):
        records = build_strategy_records(None, None)
        assert records == []


# =====================================================================
# 3. Failure Pattern Analysis Tests
# =====================================================================

class TestFailurePatterns:

    def _make_records(self):
        """Create a mix of passing and failing strategies."""
        return [
            # Narrow stop, failing
            {"version": 1, "sharpe": -0.5, "best_params": {"atr_stop_mult": 2.0}, "freq": "1h", "n_trades": 600},
            {"version": 2, "sharpe": -0.2, "best_params": {"atr_stop_mult": 2.5}, "freq": "1h", "n_trades": 100},
            # Wide stop, passing
            {"version": 3, "sharpe": 0.8, "best_params": {"atr_stop_mult": 4.0}, "freq": "daily", "n_trades": 50},
            {"version": 4, "sharpe": 1.2, "best_params": {"atr_stop_mult": 5.0}, "freq": "daily", "n_trades": 30},
            # Zero trades
            {"version": 5, "sharpe": 0, "best_params": {}, "freq": "daily", "n_trades": 0},
            # Many params, failing
            {"version": 6, "sharpe": -0.3, "best_params": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, "freq": "4h", "n_trades": 20},
        ]

    def test_patterns_not_empty(self):
        records = self._make_records()
        patterns = analyze_failure_patterns(records)
        assert len(patterns) > 0

    def test_freq_pattern_detected(self):
        records = self._make_records()
        patterns = analyze_failure_patterns(records)
        freq_patterns = [p for p in patterns if "freq=" in p[0]]
        assert len(freq_patterns) > 0

    def test_zero_trades_detected(self):
        records = self._make_records()
        patterns = analyze_failure_patterns(records)
        zero_patterns = [p for p in patterns if "n_trades == 0" in p[0]]
        assert len(zero_patterns) == 1
        assert zero_patterns[0][1] == 1  # 1 dead strategy

    def test_overtrading_detected(self):
        records = self._make_records()
        patterns = analyze_failure_patterns(records)
        ot_patterns = [p for p in patterns if "overtrading" in p[0]]
        assert len(ot_patterns) == 1

    def test_many_params_detected(self):
        records = self._make_records()
        patterns = analyze_failure_patterns(records)
        param_patterns = [p for p in patterns if "params > 5" in p[0]]
        assert len(param_patterns) == 1


# =====================================================================
# 4. File Loading Tests
# =====================================================================

class TestFileLoading:

    def test_missing_directory(self):
        opt, test = load_results(Path("/nonexistent/path"))
        assert opt is None
        assert test is None

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt, test = load_results(Path(tmpdir))
            assert opt is None
            assert test is None

    def test_load_optimization_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"version": "v1", "best_sharpe": 0.5, "best_params": {}}]
            (Path(tmpdir) / "optimization_results.json").write_text(json.dumps(data))
            opt, test = load_results(Path(tmpdir))
            assert opt is not None
            assert len(opt) == 1
            assert test is None

    def test_load_coarse_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"version": "v1", "best_sharpe": 0.3, "best_params": {}, "phase": "coarse"}]
            (Path(tmpdir) / "optimization_coarse.json").write_text(json.dumps(data))
            opt, test = load_results(Path(tmpdir))
            assert opt is not None
            assert opt[0]["phase"] == "coarse"

    def test_load_test_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = [{"version": "v1", "sharpe": 0.8, "n_trades": 50}]
            (Path(tmpdir) / "test_results.json").write_text(json.dumps(data))
            opt, test = load_results(Path(tmpdir))
            assert opt is None
            assert test is not None
            assert len(test) == 1


# =====================================================================
# 5. Recommendations Tests
# =====================================================================

class TestRecommendations:

    def test_high_failure_rate(self):
        # All failing
        records = [
            {"version": i, "sharpe": -0.5, "best_params": {}, "freq": "daily"}
            for i in range(1, 21)
        ]
        cats = get_categories(20)
        groups = group_by_category(records, cats)
        recs = generate_recommendations(records, groups, cats)
        assert any("failure rate" in r.lower() for r in recs)

    def test_no_issues(self):
        # All passing
        records = [
            {"version": i, "sharpe": 1.0, "best_params": {"atr_stop_mult": 4.0},
             "freq": "daily", "n_trades": 50}
            for i in range(1, 11)
        ]
        cats = get_categories(10)
        groups = group_by_category(records, cats)
        recs = generate_recommendations(records, groups, cats)
        assert len(recs) >= 1  # At least the "no major patterns" message


# =====================================================================
# 6. Integration — run_analysis with mock data
# =====================================================================

class TestRunAnalysis:

    def test_run_with_mock_data(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            opt = [
                {"version": f"v{i}", "best_sharpe": 0.5 if i % 2 == 0 else -0.1,
                 "best_params": {"atr_stop_mult": 3.0 + i * 0.1}, "freq": "daily"}
                for i in range(1, 11)
            ]
            (Path(tmpdir) / "optimization_results.json").write_text(json.dumps(opt))
            run_analysis(Path(tmpdir))
            captured = capsys.readouterr()
            assert "Failure Analysis" in captured.out
            assert "Recommendations" in captured.out

    def test_run_with_no_data(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_analysis(Path(tmpdir))
            captured = capsys.readouterr()
            assert "ERROR" in captured.out
