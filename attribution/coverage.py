"""Regime Coverage Matrix — analyse portfolio-level regime exposure.

Takes RegimeAttributionResults (one per strategy) and produces:
1. Markdown coverage matrix (strategy x regime with performance + status icons)
2. JSON file for programmatic use
3. RED FLAG detection when a regime has < 2 profitable strategies

Usage (standalone):
    python -m attribution.coverage  (not typically called directly; use batch.py)
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401


# =========================================================================
# Status helpers
# =========================================================================

def _pnl_icon(total_pnl_pct: float) -> str:
    """Return status icon based on total PnL percentage."""
    if total_pnl_pct > 10:
        return "\u2705"   # green check — strong positive
    elif total_pnl_pct > 0:
        return "\u26a0\ufe0f"  # warning — marginally positive
    else:
        return "\u274c"   # red X — negative


def _is_profitable(total_pnl_pct: float) -> bool:
    """Consider a strategy profitable in a regime if total PnL > 0."""
    return total_pnl_pct > 0


# =========================================================================
# Regime dimension definitions
# =========================================================================

TREND_REGIMES = {
    "strong": "Strong Trend",
    "weak": "Weak Trend",
    "none": "No Trend",
}

VOLATILITY_REGIMES = {
    "high": "High Vol",
    "normal": "Normal Vol",
    "low": "Low Vol",
}

ACTIVITY_REGIMES = {
    "active": "Active",
    "normal": "Normal",
    "quiet": "Quiet",
}


# =========================================================================
# Core Matrix Generation
# =========================================================================

def _build_dimension_matrix(
    regime_results: dict,
    weights: dict,
    dimension: str,
    regime_labels: dict,
) -> tuple[list[list[str]], dict[str, dict], list[str]]:
    """Build a coverage matrix for one regime dimension.

    Args:
        regime_results: {version: RegimeAttributionResult}
        weights: {version: weight_float}
        dimension: "by_trend", "by_volatility", or "by_activity"
        regime_labels: {key: display_label}

    Returns:
        (table_rows, coverage_data, red_flags)
    """
    versions = sorted(regime_results.keys(), key=lambda v: weights.get(v, 0), reverse=True)
    regime_keys = list(regime_labels.keys())

    # Track profitable strategy counts per regime
    profitable_counts = {rk: 0 for rk in regime_keys}
    total_with_data = {rk: 0 for rk in regime_keys}

    table_rows = []
    coverage_data = {}

    for ver in versions:
        result = regime_results[ver]
        regime_dict = getattr(result, dimension, {})
        weight_pct = weights.get(ver, 0) * 100

        row_cells = []
        ver_data = {}

        for rk in regime_keys:
            stats = regime_dict.get(rk)
            if stats and stats.n_trades > 0:
                pnl = stats.total_pnl_pct
                icon = _pnl_icon(pnl)
                cell = f"{icon} {pnl:+.1f}%"
                total_with_data[rk] += 1
                if _is_profitable(pnl):
                    profitable_counts[rk] += 1
                ver_data[rk] = {
                    "total_pnl_pct": round(pnl, 2),
                    "n_trades": stats.n_trades,
                    "win_rate": round(stats.win_rate, 1),
                    "profitable": _is_profitable(pnl),
                }
            else:
                cell = "\u2014"
                ver_data[rk] = None

        row_cells = [f"{ver} ({weight_pct:.0f}%)"] + [
            (f"{_pnl_icon(regime_dict.get(rk).total_pnl_pct)} "
             f"{regime_dict.get(rk).total_pnl_pct:+.1f}%"
             if regime_dict.get(rk) and regime_dict[rk].n_trades > 0
             else "\u2014")
            for rk in regime_keys
        ]
        table_rows.append(row_cells)
        coverage_data[ver] = ver_data

    # Coverage summary row
    n_strategies = len(versions)
    summary_cells = ["**Portfolio Coverage**"] + [
        f"{profitable_counts[rk]}/{n_strategies}"
        for rk in regime_keys
    ]
    table_rows.append(summary_cells)

    # Red flags: regimes with < 2 profitable strategies
    red_flags = []
    for rk in regime_keys:
        if profitable_counts[rk] < 2 and total_with_data[rk] > 0:
            label = regime_labels[rk]
            count = profitable_counts[rk]
            red_flags.append(
                f"{label}: only {count} profitable strateg{'y' if count == 1 else 'ies'}"
            )

    return table_rows, coverage_data, red_flags


def generate_coverage_matrix(
    regime_results: dict,
    weights: dict,
    symbol: str,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Generate regime coverage matrix as Markdown + JSON.

    Args:
        regime_results: {version: RegimeAttributionResult}
        weights: {version: weight_float}
        symbol: Trading symbol (for filenames).
        output_dir: Output directory (default: research_log/attribution/).

    Returns:
        (md_path, json_path) — paths to generated files.
    """
    if output_dir is None:
        output_dir = str(Path(QBASE_ROOT) / "research_log" / "attribution")
    os.makedirs(output_dir, exist_ok=True)

    all_red_flags = []
    all_coverage_data = {}
    lines = []

    lines.append(f"# Portfolio Regime Coverage Matrix — {symbol}")
    lines.append("")
    lines.append(f"Strategies: {len(regime_results)} | "
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # === Trend Dimension ===
    trend_rows, trend_data, trend_flags = _build_dimension_matrix(
        regime_results, weights, "by_trend", TREND_REGIMES,
    )
    all_coverage_data["trend"] = trend_data
    all_red_flags.extend(trend_flags)

    lines.append("## By Trend Strength (ADX)")
    lines.append("")
    _write_matrix_table(lines, list(TREND_REGIMES.values()), trend_rows)

    # === Volatility Dimension ===
    vol_rows, vol_data, vol_flags = _build_dimension_matrix(
        regime_results, weights, "by_volatility", VOLATILITY_REGIMES,
    )
    all_coverage_data["volatility"] = vol_data
    all_red_flags.extend(vol_flags)

    lines.append("## By Volatility (ATR Percentile)")
    lines.append("")
    _write_matrix_table(lines, list(VOLATILITY_REGIMES.values()), vol_rows)

    # === Activity Dimension ===
    act_rows, act_data, act_flags = _build_dimension_matrix(
        regime_results, weights, "by_activity", ACTIVITY_REGIMES,
    )
    all_coverage_data["activity"] = act_data
    all_red_flags.extend(act_flags)

    lines.append("## By Volume Activity")
    lines.append("")
    _write_matrix_table(lines, list(ACTIVITY_REGIMES.values()), act_rows)

    # === Red Flags ===
    lines.append("## Red Flags")
    lines.append("")
    if all_red_flags:
        for flag in all_red_flags:
            lines.append(f"- **RED FLAG**: {flag}")
    else:
        lines.append("No red flags detected. All regimes have >= 2 profitable strategies.")
    lines.append("")

    # === Per-strategy summary ===
    lines.append("## Strategy Summary")
    lines.append("")
    lines.append("| Strategy | Weight | Sharpe | Trades | Best Regime | Worst Regime |")
    lines.append("|----------|:------:|:------:|:------:|------------|-------------|")
    for ver in sorted(regime_results.keys(), key=lambda v: weights.get(v, 0), reverse=True):
        r = regime_results[ver]
        w = weights.get(ver, 0) * 100
        lines.append(
            f"| {ver} | {w:.0f}% | {r.total_sharpe:.2f} | {r.total_trades} | "
            f"{r.best_regime} | {r.worst_regime} |"
        )
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    # Write Markdown
    md_path = os.path.join(output_dir, f"portfolio_{symbol}_coverage.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # Write JSON
    json_data = {
        "symbol": symbol,
        "n_strategies": len(regime_results),
        "generated": datetime.now().isoformat(),
        "coverage": all_coverage_data,
        "red_flags": all_red_flags,
        "strategy_weights": {v: round(w, 4) for v, w in weights.items()},
    }
    json_path = os.path.join(output_dir, f"portfolio_{symbol}_coverage.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return md_path, json_path


def _write_matrix_table(lines: list, headers: list[str], rows: list[list[str]]):
    """Write a Markdown table from headers and rows."""
    header_line = "| Strategy | " + " | ".join(headers) + " |"
    sep_line = "|----------|" + "|".join([":---:" for _ in headers]) + "|"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


# =========================================================================
# Utility: extract red flags from coverage JSON
# =========================================================================

def get_red_flags(coverage_json_path: str) -> list[str]:
    """Load coverage JSON and return red flags list."""
    with open(coverage_json_path) as f:
        data = json.load(f)
    return data.get("red_flags", [])
