"""
Tests for the unified optimization result schema.

Validates:
1. build_result_entry produces all required fields
2. detect_strategy_status correctly identifies dead/error/active
3. is_strategy_dead checks existing results files
4. Existing result JSON files contain required fields (or reports what's missing)
"""

import json
import os
import tempfile

import pytest

import sys
from pathlib import Path

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest as _  # noqa: F401

from strategies.optimizer_core import (
    build_result_entry,
    detect_strategy_status,
    is_strategy_dead,
    UNIFIED_SCHEMA_FIELDS,
)


# =====================================================================
# 1. build_result_entry tests
# =====================================================================

class TestBuildResultEntry:

    def test_all_required_fields_present(self):
        entry = build_result_entry(
            version="v1", freq="daily", best_params={"a": 1},
            sharpe=1.5, score=7.2, n_trials=100, phase="two_phase",
            robustness={"is_robust": True, "neighbor_mean": 1.3,
                        "neighbor_std": 0.2, "above_threshold_pct": 0.8},
            elapsed=42.5, status="active",
        )
        for field in UNIFIED_SCHEMA_FIELDS:
            assert field in entry, f"Missing required field: {field}"

    def test_sharpe_rounding(self):
        entry = build_result_entry(
            version="v1", freq="daily", best_params={},
            sharpe=1.23456789,
        )
        assert entry["best_sharpe"] == 1.2346

    def test_score_rounding(self):
        entry = build_result_entry(
            version="v1", freq="daily", best_params={},
            score=7.891011,
        )
        assert entry["best_score"] == 7.891

    def test_none_sharpe_and_score(self):
        entry = build_result_entry(version="v1", freq="daily", best_params={})
        assert entry["best_sharpe"] is None
        assert entry["best_score"] is None

    def test_defaults(self):
        entry = build_result_entry(version="v1", freq="daily", best_params={})
        assert entry["n_trials"] == 0
        assert entry["phase"] == "two_phase"
        assert entry["status"] == "active"
        assert entry["robustness"] is None
        assert entry["elapsed_seconds"] == 0.0

    def test_extra_fields_preserved(self):
        entry = build_result_entry(
            version="v1", freq="daily", best_params={},
            early_stopped=True, multi_seed=True, cross_seed_std=0.15,
        )
        assert entry["early_stopped"] is True
        assert entry["multi_seed"] is True
        assert entry["cross_seed_std"] == 0.15

    def test_elapsed_rounding(self):
        entry = build_result_entry(
            version="v1", freq="daily", best_params={},
            elapsed=123.456789,
        )
        assert entry["elapsed_seconds"] == 123.5


# =====================================================================
# 2. detect_strategy_status tests
# =====================================================================

class TestDetectStrategyStatus:

    def test_active_normal(self):
        result = {"best_sharpe": 1.5, "best_score": 7.0, "best_params": {"a": 1}}
        assert detect_strategy_status(result) == "active"

    def test_error_sharpe(self):
        result = {"best_sharpe": -999.0, "best_params": {"a": 1}}
        assert detect_strategy_status(result) == "error"

    def test_error_score(self):
        result = {"best_score": -999.0, "best_params": {"a": 1}}
        assert detect_strategy_status(result) == "error"

    def test_import_error(self):
        result = {"error": "No module named 'missing_lib'", "best_params": {}}
        assert detect_strategy_status(result) == "import_error"

    def test_generic_error(self):
        result = {"error": "division by zero", "best_params": {}}
        assert detect_strategy_status(result) == "error"

    def test_dead_empty_params(self):
        result = {"best_sharpe": None, "best_score": None, "best_params": {}}
        assert detect_strategy_status(result) == "dead"

    def test_dead_no_params(self):
        result = {"best_sharpe": 0.5}
        assert detect_strategy_status(result) == "dead"

    def test_active_with_zero_sharpe(self):
        result = {"best_sharpe": 0.0, "best_params": {"a": 1}}
        assert detect_strategy_status(result) == "active"

    def test_error_very_negative_sharpe(self):
        result = {"best_sharpe": -950.0, "best_params": {"a": 1}}
        assert detect_strategy_status(result) == "error"

    def test_import_error_keyword_detection(self):
        result = {"error": "ImportError: cannot import name 'Foo'", "best_params": {}}
        assert detect_strategy_status(result) == "import_error"


# =====================================================================
# 3. is_strategy_dead tests
# =====================================================================

class TestIsStrategyDead:

    def test_nonexistent_file(self):
        assert is_strategy_dead("/nonexistent/path.json", "v1") is False

    def test_empty_file(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text("[]")
        assert is_strategy_dead(str(f), "v1") is False

    def test_version_not_found(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps([{"version": "v2", "status": "active"}]))
        assert is_strategy_dead(str(f), "v1") is False

    def test_active_strategy(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps([{"version": "v1", "status": "active"}]))
        assert is_strategy_dead(str(f), "v1") is False

    def test_dead_strategy(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps([{"version": "v1", "status": "dead"}]))
        assert is_strategy_dead(str(f), "v1") is True

    def test_error_strategy(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps([{"version": "v1", "status": "error"}]))
        assert is_strategy_dead(str(f), "v1") is True

    def test_import_error_strategy(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps([{"version": "v1", "status": "import_error"}]))
        assert is_strategy_dead(str(f), "v1") is True

    def test_no_status_field(self, tmp_path):
        """Old entries without status field should not be considered dead."""
        f = tmp_path / "results.json"
        f.write_text(json.dumps([{"version": "v1", "best_sharpe": 1.5}]))
        assert is_strategy_dead(str(f), "v1") is False

    def test_corrupt_json(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text("{invalid json")
        assert is_strategy_dead(str(f), "v1") is False


# =====================================================================
# 4. Existing results file validation
# =====================================================================

RESULTS_FILES = [
    os.path.join(QBASE_ROOT, "strategies", "strong_trend", "optimization_results.json"),
    os.path.join(QBASE_ROOT, "strategies", "all_time", "ag", "optimization_results.json"),
    os.path.join(QBASE_ROOT, "strategies", "medium_trend", "optimization_results.json"),
    os.path.join(QBASE_ROOT, "strategies", "boss", "optimization_results.json"),
]

# Fields that MUST be present in new results but may be missing in legacy files.
# This test reports what's missing rather than failing hard.
LEGACY_REQUIRED = {"version", "best_params"}


class TestExistingResultsFiles:

    @pytest.mark.parametrize("filepath", RESULTS_FILES)
    def test_file_is_valid_json(self, filepath):
        if not os.path.exists(filepath):
            pytest.skip(f"File not found: {filepath}")
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, list), f"{filepath} should contain a JSON array"
        assert len(data) > 0, f"{filepath} is empty"

    @pytest.mark.parametrize("filepath", RESULTS_FILES)
    def test_legacy_required_fields(self, filepath):
        if not os.path.exists(filepath):
            pytest.skip(f"File not found: {filepath}")
        with open(filepath) as f:
            data = json.load(f)
        for entry in data:
            for field in LEGACY_REQUIRED:
                assert field in entry, (
                    f"{filepath}: entry {entry.get('version', '?')} missing '{field}'"
                )

    @pytest.mark.parametrize("filepath", RESULTS_FILES)
    def test_report_missing_unified_fields(self, filepath):
        """Reports which unified schema fields are missing in each file.

        This is informational — existing files may lack new fields until
        re-optimization adds them. The test passes but prints a summary.
        """
        if not os.path.exists(filepath):
            pytest.skip(f"File not found: {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        missing_summary = {}
        for entry in data:
            version = entry.get("version", "?")
            missing = UNIFIED_SCHEMA_FIELDS - set(entry.keys())
            if missing:
                missing_summary[version] = missing

        if missing_summary:
            # Print summary but don't fail — this is expected for legacy data
            basename = os.path.basename(os.path.dirname(filepath))
            total = len(data)
            with_missing = len(missing_summary)
            # Collect all missing field names across all entries
            all_missing = set()
            for fields in missing_summary.values():
                all_missing.update(fields)
            print(
                f"\n  {basename}: {with_missing}/{total} entries missing "
                f"unified fields: {sorted(all_missing)}"
            )
