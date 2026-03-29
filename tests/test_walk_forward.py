"""Unit tests for walk-forward validator."""

import sys
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.walk_forward import (
    generate_windows,
    load_strategy_class_from_path,
    run_walk_forward,
    save_results,
    _compute_summary,
    _serialize_params,
    parse_args,
)


# ---------------------------------------------------------------------------
# Window Generation
# ---------------------------------------------------------------------------

class TestGenerateWindows:
    """Tests for generate_windows — the core scheduling logic."""

    def test_basic_windows(self):
        """5-year train, 1-year test, 2015-2026 → 6 windows."""
        windows = generate_windows(2015, 2026, train_years=5, test_years=1)

        assert len(windows) == 6

        # Window 1: train 2015-2019, test 2020
        w1 = windows[0]
        assert w1["window_id"] == 1
        assert w1["train_start"] == "2015-01-01"
        assert w1["train_end"] == "2019-12-31"
        assert w1["test_start"] == "2020-01-01"
        assert w1["test_end"] == "2020-12-31"

        # Window 6: train 2020-2024, test 2025
        w6 = windows[5]
        assert w6["window_id"] == 6
        assert w6["train_start"] == "2020-01-01"
        assert w6["train_end"] == "2024-12-31"
        assert w6["test_start"] == "2025-01-01"
        assert w6["test_end"] == "2025-12-31"

    def test_two_year_test_windows(self):
        """3-year train, 2-year test, 2015-2025 → 3 windows."""
        windows = generate_windows(2015, 2025, train_years=3, test_years=2)

        assert len(windows) == 3

        # Window 1: train 2015-2017, test 2018-2019
        assert windows[0]["train_start"] == "2015-01-01"
        assert windows[0]["train_end"] == "2017-12-31"
        assert windows[0]["test_start"] == "2018-01-01"
        assert windows[0]["test_end"] == "2019-12-31"

        # Window 3: train 2019-2021, test 2022-2023
        assert windows[2]["train_start"] == "2019-01-01"
        assert windows[2]["train_end"] == "2021-12-31"
        assert windows[2]["test_start"] == "2022-01-01"
        assert windows[2]["test_end"] == "2023-12-31"

    def test_no_windows_if_range_too_short(self):
        """If range is too short for even one window, return empty."""
        windows = generate_windows(2020, 2023, train_years=5, test_years=1)
        assert windows == []

    def test_single_window(self):
        """Exactly enough for one window."""
        windows = generate_windows(2015, 2021, train_years=5, test_years=1)
        assert len(windows) == 1
        assert windows[0]["train_start"] == "2015-01-01"
        assert windows[0]["test_end"] == "2020-12-31"

    def test_window_ids_sequential(self):
        """Window IDs should be 1, 2, 3, ..."""
        windows = generate_windows(2015, 2026, train_years=5, test_years=1)
        ids = [w["window_id"] for w in windows]
        assert ids == list(range(1, len(windows) + 1))

    def test_no_overlap_between_train_and_test(self):
        """Training end should be strictly before test start."""
        windows = generate_windows(2015, 2026, train_years=5, test_years=1)
        for w in windows:
            assert w["train_end"] < w["test_start"]

    def test_windows_roll_by_test_years(self):
        """Consecutive windows should shift by test_years."""
        windows = generate_windows(2010, 2026, train_years=5, test_years=2)
        for i in range(len(windows) - 1):
            # Train start should advance by test_years
            y1 = int(windows[i]["train_start"][:4])
            y2 = int(windows[i + 1]["train_start"][:4])
            assert y2 - y1 == 2


# ---------------------------------------------------------------------------
# Summary Computation
# ---------------------------------------------------------------------------

class TestComputeSummary:
    """Tests for _compute_summary."""

    def test_all_ok_windows(self):
        """Summary with all valid windows."""
        results = [
            {"window_id": 1, "status": "ok", "test_sharpe": 0.82},
            {"window_id": 2, "status": "ok", "test_sharpe": 1.45},
            {"window_id": 3, "status": "ok", "test_sharpe": -0.31},
            {"window_id": 4, "status": "ok", "test_sharpe": 0.67},
        ]
        summary = _compute_summary(results)

        assert summary["n_windows"] == 4
        assert summary["n_valid"] == 4
        assert summary["win_count"] == 3
        assert summary["win_rate"] == pytest.approx(0.75)
        assert summary["worst_sharpe"] == pytest.approx(-0.31)
        assert summary["best_sharpe"] == pytest.approx(1.45)
        assert summary["mean_sharpe"] == pytest.approx(np.mean([0.82, 1.45, -0.31, 0.67]))

    def test_with_failed_windows(self):
        """Failed windows should not count in summary stats."""
        results = [
            {"window_id": 1, "status": "ok", "test_sharpe": 1.0},
            {"window_id": 2, "status": "optimization_failed", "test_sharpe": None},
            {"window_id": 3, "status": "ok", "test_sharpe": -0.5},
        ]
        summary = _compute_summary(results)

        assert summary["n_windows"] == 3
        assert summary["n_valid"] == 2
        assert summary["win_count"] == 1
        assert summary["mean_sharpe"] == pytest.approx(0.25)

    def test_no_valid_windows(self):
        """All windows failed."""
        results = [
            {"window_id": 1, "status": "optimization_failed", "test_sharpe": None},
            {"window_id": 2, "status": "test_failed", "test_sharpe": None},
        ]
        summary = _compute_summary(results)

        assert summary["n_valid"] == 0
        assert summary["mean_sharpe"] is None
        assert summary["win_rate"] == 0.0

    def test_empty_results(self):
        """Empty results list."""
        summary = _compute_summary([])
        assert summary["n_windows"] == 0
        assert summary["n_valid"] == 0

    def test_all_negative_sharpes(self):
        """All windows have negative Sharpe — win rate should be 0."""
        results = [
            {"window_id": 1, "status": "ok", "test_sharpe": -0.5},
            {"window_id": 2, "status": "ok", "test_sharpe": -1.2},
        ]
        summary = _compute_summary(results)

        assert summary["win_count"] == 0
        assert summary["win_rate"] == 0.0
        assert summary["negative_windows"] == [1, 2]


# ---------------------------------------------------------------------------
# Param Serialization
# ---------------------------------------------------------------------------

class TestSerializeParams:
    """Tests for _serialize_params."""

    def test_basic(self):
        params = {"period": 14, "mult": 2.5}
        result = _serialize_params(params)
        assert result == {"period": 14, "mult": 2.5}

    def test_numpy_types(self):
        params = {"period": np.int64(14), "mult": np.float64(2.5)}
        result = _serialize_params(params)
        assert isinstance(result["period"], int)
        assert isinstance(result["mult"], float)

    def test_none(self):
        assert _serialize_params(None) is None


# ---------------------------------------------------------------------------
# JSON Output
# ---------------------------------------------------------------------------

class TestSaveResults:
    """Tests for save_results — JSON output format."""

    def test_save_and_load(self):
        """Saved JSON should be loadable and contain all required fields."""
        output = {
            "strategy": "v12",
            "strategy_class": "StrongTrendV12",
            "symbol": "AG",
            "freq": "daily",
            "start_year": 2015,
            "end_year": 2026,
            "train_years": 5,
            "test_years": 1,
            "n_trials": 30,
            "timestamp": "2026-03-28T12:00:00",
            "windows": [
                {
                    "window_id": 1,
                    "train_start": "2015-01-01",
                    "train_end": "2019-12-31",
                    "test_start": "2020-01-01",
                    "test_end": "2020-12-31",
                    "status": "ok",
                    "train_score": 3.5,
                    "test_sharpe": 0.82,
                    "test_trades": 12,
                    "test_return": 0.15,
                    "test_max_dd": -0.08,
                    "best_params": {"period": 14, "mult": 2.5},
                },
            ],
            "summary": {
                "n_windows": 1,
                "n_valid": 1,
                "mean_sharpe": 0.82,
                "std_sharpe": 0.0,
                "worst_sharpe": 0.82,
                "best_sharpe": 0.82,
                "win_rate": 1.0,
                "win_count": 1,
                "positive_windows": [1],
                "negative_windows": [],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_results(output, output_dir=tmpdir)

            assert os.path.exists(filepath)
            assert filepath.endswith("v12_AG.json")

            with open(filepath) as f:
                loaded = json.load(f)

            assert loaded["strategy"] == "v12"
            assert loaded["symbol"] == "AG"
            assert len(loaded["windows"]) == 1
            assert loaded["windows"][0]["test_sharpe"] == 0.82
            assert loaded["summary"]["win_rate"] == 1.0

    def test_filename_format(self):
        """Filename should be {strategy}_{symbol}.json."""
        output = {"strategy": "v46", "symbol": "I"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_results(output, output_dir=tmpdir)
            assert os.path.basename(filepath) == "v46_I.json"


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args(self):
        args = parse_args(["--strategy", "v12.py", "--symbol", "AG"])
        assert args.strategy == "v12.py"
        assert args.symbol == "AG"
        assert args.train_years == 5
        assert args.test_years == 1
        assert args.trials == 30

    def test_quick_mode(self):
        args = parse_args(["--strategy", "v12.py", "--symbol", "AG", "--quick"])
        assert args.trials == 10

    def test_custom_params(self):
        args = parse_args([
            "--strategy", "v12.py", "--symbol", "AG",
            "--train-years", "3", "--test-years", "2",
            "--start", "2010", "--end", "2025",
            "--freq", "4h", "--trials", "50",
        ])
        assert args.train_years == 3
        assert args.test_years == 2
        assert args.start == 2010
        assert args.end == 2025
        assert args.freq == "4h"
        assert args.trials == 50


# ---------------------------------------------------------------------------
# Integration-style Tests (with mocking)
# ---------------------------------------------------------------------------

class TestWalkForwardWithMocks:
    """Tests that mock AlphaForge dependencies to verify walk-forward logic."""

    def _make_mock_strategy_file(self, tmpdir):
        """Create a minimal mock strategy .py file."""
        code = '''
class MockStrategy:
    name = "mock"
    warmup = 20
    freq = "daily"
    period: int = 14
    mult: float = 2.5
'''
        path = os.path.join(str(tmpdir), "mock_strategy.py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def test_load_strategy_class(self, tmp_path):
        """Loading a strategy class from file should find the class."""
        path = self._make_mock_strategy_file(tmp_path)
        cls, name = load_strategy_class_from_path(path)

        assert name == "MockStrategy"
        assert hasattr(cls, "warmup")
        assert cls.warmup == 20

    def test_load_strategy_not_found(self):
        """Loading from non-existent path should raise."""
        with pytest.raises(FileNotFoundError):
            load_strategy_class_from_path("/nonexistent/strategy.py")

    def test_load_strategy_no_class(self, tmp_path):
        """Loading from file with no strategy class should raise."""
        path = os.path.join(str(tmp_path), "empty.py")
        with open(path, "w") as f:
            f.write("# nothing here\nx = 42\n")

        with pytest.raises(ValueError, match="No strategy class"):
            load_strategy_class_from_path(path)

    @patch("strategies.walk_forward.optimize_window")
    @patch("strategies.walk_forward.run_single_backtest")
    @patch("strategies.walk_forward.auto_discover_params")
    @patch("strategies.walk_forward.load_strategy_class_from_path")
    def test_walk_forward_with_mock_backtest(
        self, mock_load, mock_discover, mock_backtest, mock_optimize, tmp_path
    ):
        """Full walk-forward with mocked optimization and backtest."""
        # Mock strategy class
        mock_cls = MagicMock()
        mock_cls.warmup = 20
        mock_cls.freq = "daily"
        mock_load.return_value = (mock_cls, "MockStrategy")

        # Mock param discovery
        mock_discover.return_value = [
            {"name": "period", "low": 5, "high": 30, "step": 2, "dtype": "int"},
        ]

        # Mock optimization — return fixed params for each window
        mock_optimize.return_value = {
            "best_params": {"period": 14},
            "best_value": 3.5,
        }

        # Mock backtest — return varying Sharpe per call
        sharpe_sequence = [0.82, 1.45, -0.31]
        call_count = [0]

        def mock_bt_side_effect(*args, **kwargs):
            idx = call_count[0] % len(sharpe_sequence)
            call_count[0] += 1
            return {
                "sharpe": sharpe_sequence[idx],
                "max_drawdown": -0.05,
                "n_trades": 12,
                "total_return": 0.15,
                "profit_concentration": 0.4,
                "monthly_win_rate": 0.6,
            }

        mock_backtest.side_effect = mock_bt_side_effect

        output = run_walk_forward(
            strategy_path="fake/v12.py",
            symbol="AG",
            start_year=2018,
            end_year=2024,
            train_years=3,
            test_years=1,
            freq="daily",
            n_trials=10,
            verbose=False,
        )

        assert output["strategy"] == "v12"
        assert output["symbol"] == "AG"
        assert len(output["windows"]) == 3
        assert output["summary"]["n_valid"] == 3
        assert output["summary"]["win_count"] == 2  # 0.82 and 1.45 are positive

    @patch("strategies.walk_forward.optimize_window")
    @patch("strategies.walk_forward.auto_discover_params")
    @patch("strategies.walk_forward.load_strategy_class_from_path")
    def test_walk_forward_optimization_failure(
        self, mock_load, mock_discover, mock_optimize
    ):
        """Windows where optimization fails should be marked accordingly."""
        mock_cls = MagicMock()
        mock_cls.warmup = 20
        mock_cls.freq = "daily"
        mock_load.return_value = (mock_cls, "MockStrategy")
        mock_discover.return_value = [
            {"name": "period", "low": 5, "high": 30, "step": 2, "dtype": "int"},
        ]

        # All optimizations fail
        mock_optimize.return_value = None

        output = run_walk_forward(
            strategy_path="fake/v12.py",
            symbol="AG",
            start_year=2018,
            end_year=2022,
            train_years=3,
            test_years=1,
            verbose=False,
        )

        assert output["summary"]["n_valid"] == 0
        for w in output["windows"]:
            assert w["status"] == "optimization_failed"

    @patch("strategies.walk_forward.optimize_window")
    @patch("strategies.walk_forward.run_single_backtest")
    @patch("strategies.walk_forward.auto_discover_params")
    @patch("strategies.walk_forward.load_strategy_class_from_path")
    def test_walk_forward_no_trades_in_test(
        self, mock_load, mock_discover, mock_backtest, mock_optimize
    ):
        """Window where test backtest returns failure (e.g. no trades)."""
        mock_cls = MagicMock()
        mock_cls.warmup = 20
        mock_cls.freq = "daily"
        mock_load.return_value = (mock_cls, "MockStrategy")
        mock_discover.return_value = [
            {"name": "period", "low": 5, "high": 30, "step": 2, "dtype": "int"},
        ]

        mock_optimize.return_value = {
            "best_params": {"period": 14},
            "best_value": 3.0,
        }

        # Backtest fails (insufficient data / no trades)
        mock_backtest.return_value = {
            "sharpe": -999.0,
            "max_drawdown": None,
            "n_trades": None,
            "total_return": None,
            "profit_concentration": None,
            "monthly_win_rate": None,
        }

        output = run_walk_forward(
            strategy_path="fake/v12.py",
            symbol="AG",
            start_year=2019,
            end_year=2024,
            train_years=3,
            test_years=1,
            verbose=False,
        )

        for w in output["windows"]:
            assert w["status"] == "test_failed"
        assert output["summary"]["n_valid"] == 0
