#!/usr/bin/env python3
"""
All-Time Strategy Failure Analysis
====================================
Analyzes optimization/test results to identify failure patterns and
generate actionable recommendations.

Usage:
    python strategies/all_time/ag/analyze_failures.py                # AG directory
    python strategies/all_time/ag/analyze_failures.py --dir strategies/all_time/i  # I directory
    python strategies/all_time/ag/analyze_failures.py --dir /absolute/path/to/dir
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Category definitions: version ranges → category name
# Supports both 100-strategy and 200-strategy layouts
CATEGORIES_100 = [
    ("Trend",           1,  20),
    ("Mean Reversion", 21,  40),
    ("Breakout",       41,  60),
    ("Multi-TF",       61,  80),
    ("Adaptive",       81, 100),
]

CATEGORIES_200 = [
    ("Trend",           1,  40),
    ("Mean Reversion", 41,  80),
    ("Breakout",       81, 120),
    ("Multi-TF",      121, 160),
    ("Adaptive",      161, 200),
]


def get_categories(max_version: int) -> list:
    """Select category mapping based on number of strategies."""
    if max_version <= 100:
        return CATEGORIES_100
    return CATEGORIES_200


def version_to_int(v: str) -> int:
    """Extract integer from version string like 'v1', 'v123'."""
    return int(v.lstrip("v"))


def get_category(version_num: int, categories: list) -> str:
    """Map a version number to its category name."""
    for name, lo, hi in categories:
        if lo <= version_num <= hi:
            return name
    return "Unknown"


def load_results(directory: Path) -> tuple:
    """Load optimization and test results from a directory.

    Returns:
        (opt_data, test_data) — either may be None if file not found.
        opt_data: list of dicts with 'version', 'best_sharpe', 'best_params'
        test_data: list of dicts with 'version', 'sharpe', etc.
    """
    opt_data = None
    test_data = None

    # Try optimization_results.json first, then optimization_coarse.json
    for fname in ["optimization_results.json", "optimization_coarse.json"]:
        fpath = directory / fname
        if fpath.exists():
            with open(fpath) as f:
                opt_data = json.load(f)
            break

    # Try test_results.json
    test_path = directory / "test_results.json"
    if test_path.exists():
        with open(test_path) as f:
            test_data = json.load(f)

    return opt_data, test_data


def build_strategy_records(opt_data, test_data):
    """Merge optimization and test data into unified records.

    Each record has:
        version (int), sharpe (float), best_params (dict),
        opt_sharpe (float), freq (str), and optionally
        total_return, max_drawdown, n_trades, calmar.
    """
    records = {}

    if opt_data:
        for entry in opt_data:
            vnum = version_to_int(entry["version"])
            records[vnum] = {
                "version": vnum,
                "opt_sharpe": entry.get("best_sharpe", 0.0),
                "best_params": entry.get("best_params", {}),
                "freq": entry.get("freq", "unknown"),
            }

    if test_data:
        for entry in test_data:
            vnum = version_to_int(entry["version"])
            if vnum not in records:
                records[vnum] = {
                    "version": vnum,
                    "best_params": {},
                    "freq": entry.get("freq", "unknown"),
                }
            rec = records[vnum]
            rec["sharpe"] = entry.get("sharpe", 0.0)
            rec["total_return"] = entry.get("total_return", 0.0)
            rec["max_drawdown"] = entry.get("max_drawdown", 0.0)
            rec["n_trades"] = entry.get("n_trades", 0)
            rec["calmar"] = entry.get("calmar", 0.0)
            if "freq" in entry:
                rec["freq"] = entry["freq"]

    # For entries with only opt data, use opt_sharpe as sharpe
    for rec in records.values():
        if "sharpe" not in rec:
            rec["sharpe"] = rec.get("opt_sharpe", 0.0)

    return list(records.values())


def group_by_category(records, categories):
    """Group records by strategy category."""
    groups = defaultdict(list)
    for rec in records:
        cat = get_category(rec["version"], categories)
        groups[cat].append(rec)
    return groups


def analyze_failure_patterns(records):
    """Extract common patterns among failing strategies.

    Returns list of (pattern_description, fail_count, total_count, fail_pct).
    """
    patterns = []

    # Collect strategies that have params with atr-related keys
    atr_key_names = ["atr_stop_mult", "atr_trail_mult", "trail_mult", "chand_mult"]

    # Pattern 1: Narrow ATR stop (< 3.0)
    narrow_stop = [r for r in records if _get_atr_mult(r) is not None]
    if narrow_stop:
        narrow_fail = [r for r in narrow_stop if _get_atr_mult(r) < 3.0]
        narrow_fail_neg = [r for r in narrow_fail if r["sharpe"] <= 0]
        narrow_total_neg = len(narrow_fail)
        if narrow_total_neg > 0:
            patterns.append((
                f"atr_stop/trail_mult < 3.0",
                len(narrow_fail_neg),
                narrow_total_neg,
                100.0 * len(narrow_fail_neg) / narrow_total_neg if narrow_total_neg else 0,
            ))

    # Pattern 2: Wide ATR stop (>= 3.0)
    if narrow_stop:
        wide = [r for r in narrow_stop if _get_atr_mult(r) >= 3.0]
        wide_neg = [r for r in wide if r["sharpe"] <= 0]
        if wide:
            patterns.append((
                f"atr_stop/trail_mult >= 3.0",
                len(wide_neg),
                len(wide),
                100.0 * len(wide_neg) / len(wide) if wide else 0,
            ))

    # Pattern 3: High-frequency strategies (1h) tend to fail more
    freq_groups = defaultdict(list)
    for r in records:
        freq_groups[r.get("freq", "unknown")].append(r)
    for freq, recs in sorted(freq_groups.items()):
        neg = [r for r in recs if r["sharpe"] <= 0]
        patterns.append((
            f"freq={freq}",
            len(neg),
            len(recs),
            100.0 * len(neg) / len(recs) if recs else 0,
        ))

    # Pattern 4: Too many parameters (> 5)
    many_params = [r for r in records if len(r.get("best_params", {})) > 5]
    if many_params:
        many_fail = [r for r in many_params if r["sharpe"] <= 0]
        patterns.append((
            f"params > 5",
            len(many_fail),
            len(many_params),
            100.0 * len(many_fail) / len(many_params),
        ))

    few_params = [r for r in records if 0 < len(r.get("best_params", {})) <= 5]
    if few_params:
        few_fail = [r for r in few_params if r["sharpe"] <= 0]
        patterns.append((
            f"params <= 5",
            len(few_fail),
            len(few_params),
            100.0 * len(few_fail) / len(few_params),
        ))

    # Pattern 5: Zero trades
    zero_trades = [r for r in records if r.get("n_trades", -1) == 0]
    if zero_trades:
        patterns.append((
            "n_trades == 0 (dead strategies)",
            len(zero_trades),
            len(zero_trades),
            100.0,
        ))

    # Pattern 6: Overtrading (> 500 trades)
    overtrade = [r for r in records if r.get("n_trades", 0) > 500]
    if overtrade:
        ot_fail = [r for r in overtrade if r["sharpe"] <= 0]
        patterns.append((
            f"n_trades > 500 (overtrading)",
            len(ot_fail),
            len(overtrade),
            100.0 * len(ot_fail) / len(overtrade),
        ))

    # Pattern 7: Indicator frequency — which param names appear most in failures
    fail_params = defaultdict(int)
    pass_params = defaultdict(int)
    for r in records:
        bucket = fail_params if r["sharpe"] <= 0 else pass_params
        for k in r.get("best_params", {}):
            bucket[k] += 1

    return patterns


def _get_atr_mult(record):
    """Extract ATR multiplier from params (various key names)."""
    params = record.get("best_params", {})
    for key in ["atr_stop_mult", "atr_trail_mult", "trail_mult", "chand_mult"]:
        if key in params:
            return params[key]
    return None


def generate_recommendations(records, groups, categories):
    """Generate actionable recommendations based on analysis."""
    recs = []

    total = len(records)
    positive = sum(1 for r in records if r["sharpe"] > 0)
    negative = total - positive

    if negative > total * 0.5:
        recs.append("High failure rate — consider tightening strategy generation criteria.")

    # Check ATR pattern
    atr_records = [r for r in records if _get_atr_mult(r) is not None]
    if atr_records:
        narrow = [r for r in atr_records if _get_atr_mult(r) < 3.0]
        narrow_fail = sum(1 for r in narrow if r["sharpe"] <= 0)
        if narrow and narrow_fail / len(narrow) > 0.5:
            recs.append(
                f"Narrow ATR stops (< 3.0) have {narrow_fail}/{len(narrow)} failures — "
                f"enforce minimum atr_mult >= 3.0 in strategy generation."
            )

    # Check frequency
    freq_groups = defaultdict(list)
    for r in records:
        freq_groups[r.get("freq", "unknown")].append(r)
    for freq, frecs in freq_groups.items():
        fail_rate = sum(1 for r in frecs if r["sharpe"] <= 0) / len(frecs) if frecs else 0
        if fail_rate > 0.6 and freq in ("1h", "5min", "10min", "30min"):
            recs.append(
                f"Frequency '{freq}' has {fail_rate:.0%} failure rate — "
                f"prefer daily or 4h for this instrument."
            )

    # Check overtrading
    overtrade = [r for r in records if r.get("n_trades", 0) > 500]
    if overtrade:
        ot_fail = sum(1 for r in overtrade if r["sharpe"] <= 0)
        if ot_fail / len(overtrade) > 0.5:
            recs.append(
                f"Overtrading (>500 trades) has {ot_fail}/{len(overtrade)} failures — "
                f"add trade frequency caps or stricter entry filters."
            )

    # Check dead strategies
    dead = sum(1 for r in records if r.get("n_trades", -1) == 0)
    if dead > 0:
        recs.append(
            f"{dead} strategies produced zero trades — "
            f"review entry conditions for overly restrictive filters."
        )

    # Category-specific
    for cat_name, lo, hi in categories:
        cat_recs = groups.get(cat_name, [])
        if cat_recs:
            fail_rate = sum(1 for r in cat_recs if r["sharpe"] <= 0) / len(cat_recs)
            if fail_rate > 0.7:
                recs.append(
                    f"Category '{cat_name}' (v{lo}-v{hi}) has {fail_rate:.0%} failure rate — "
                    f"this approach may not suit this instrument."
                )

    if not recs:
        recs.append("No major failure patterns detected — results look reasonable.")

    return recs


def run_analysis(directory: Path):
    """Main analysis pipeline."""
    opt_data, test_data = load_results(directory)

    if opt_data is None and test_data is None:
        print(f"ERROR: No optimization_results.json, optimization_coarse.json, "
              f"or test_results.json found in {directory}")
        return

    records = build_strategy_records(opt_data, test_data)
    if not records:
        print("ERROR: No strategy records found.")
        return

    max_version = max(r["version"] for r in records)
    categories = get_categories(max_version)

    total = len(records)
    positive = sum(1 for r in records if r["sharpe"] > 0)
    negative = total - positive

    dir_name = directory.name.upper()
    parent_name = directory.parent.name if directory.parent.name != "all_time" else ""
    label = f"{dir_name}" if not parent_name else f"{dir_name}"

    print(f"\n{'='*60}")
    print(f"  {label} All-Time Failure Analysis")
    print(f"{'='*60}")
    print(f"\n{label}: {total} strategies, {positive} positive Sharpe, {negative} negative/zero")
    print(f"Overall pass rate: {100*positive/total:.1f}%")

    # By category
    groups = group_by_category(records, categories)
    print(f"\nBy Category:")
    for cat_name, lo, hi in categories:
        cat_recs = groups.get(cat_name, [])
        if not cat_recs:
            continue
        cat_pos = sum(1 for r in cat_recs if r["sharpe"] > 0)
        cat_total = len(cat_recs)
        fail_pct = 100 * (cat_total - cat_pos) / cat_total if cat_total else 0
        avg_sharpe = sum(r["sharpe"] for r in cat_recs) / cat_total
        print(f"  {cat_name:20s} (v{lo:>3d}-v{hi:>3d}): "
              f"{cat_pos:>3d}/{cat_total:<3d} positive ({fail_pct:5.1f}% failure) "
              f"avg_sharpe={avg_sharpe:+.3f}")

    # Top performers
    top = sorted(records, key=lambda r: r["sharpe"], reverse=True)[:10]
    print(f"\nTop 10 by Sharpe:")
    for r in top:
        freq_str = r.get("freq", "?")
        cat = get_category(r["version"], categories)
        extra = ""
        if "total_return" in r:
            extra = f" ret={r['total_return']:+.2%} dd={r['max_drawdown']:.1%} trades={r.get('n_trades', '?')}"
        print(f"  v{r['version']:<4d} sharpe={r['sharpe']:+.4f} freq={freq_str:<6s} [{cat}]{extra}")

    # Failure patterns
    patterns = analyze_failure_patterns(records)
    print(f"\nCommon Failure Patterns:")
    for desc, fail_count, total_count, fail_pct in patterns:
        print(f"  {desc:40s}: {fail_count:>3d}/{total_count:<3d} failed ({fail_pct:5.1f}%)")

    # Recommendations
    recommendations = generate_recommendations(records, groups, categories)
    print(f"\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print()


def main():
    parser = argparse.ArgumentParser(description="All-Time Strategy Failure Analysis")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to the strategy directory (default: directory containing this script)",
    )
    args = parser.parse_args()

    if args.dir:
        directory = Path(args.dir)
        if not directory.is_absolute():
            directory = Path.cwd() / directory
    else:
        directory = Path(__file__).resolve().parent

    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    run_analysis(directory)


if __name__ == "__main__":
    main()
