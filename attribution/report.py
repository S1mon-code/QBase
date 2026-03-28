"""Attribution report generator — Markdown reports + orchestrator."""
import os
from datetime import datetime


def generate_attribution_report(signal_result, regime_result, output_path: str) -> str:
    """Generate Markdown attribution report combining signal and regime results."""
    lines = []

    version = signal_result.strategy_version
    symbol = signal_result.symbol
    period = signal_result.period

    lines.append(f"# {version} Attribution Report — {symbol} ({period})")
    lines.append("")

    # === Signal Attribution ===
    lines.append("## Signal Attribution (Ablation Test)")
    lines.append("")
    lines.append(f"**Baseline Sharpe: {signal_result.baseline_sharpe:.3f}** | "
                 f"Trades: {signal_result.baseline_trades}")
    lines.append("")

    if signal_result.contributions:
        lines.append("| Indicator | Role | Without It | Contribution | % of Total |")
        lines.append("|-----------|------|:----------:|:------------:|:----------:|")
        for name, c in sorted(signal_result.contributions.items(),
                               key=lambda x: -x[1]['contribution']):
            lines.append(
                f"| {name} | {c['role']} | "
                f"{c['ablated_sharpe']:.3f} | "
                f"{c['contribution']:+.3f} | "
                f"{c['pct_contribution']:.1f}% |"
            )
        lines.append("")
        lines.append(f"**Dominant indicator**: {signal_result.dominant_indicator}")
        if signal_result.redundant_indicators:
            lines.append(f"**Redundant indicators**: {', '.join(signal_result.redundant_indicators)}")
        else:
            lines.append("**Redundant indicators**: None")
    else:
        lines.append("*No ablation data available.*")

    # Auto insights
    lines.append("")
    lines.append("### Interpretation")
    if signal_result.contributions:
        for name, c in signal_result.contributions.items():
            if c['pct_contribution'] > 60:
                lines.append(f"- Strategy is heavily dependent on **{name}** "
                             f"({c['pct_contribution']:.0f}% contribution). "
                             f"Consider if this is a feature or a risk.")
            elif c['pct_contribution'] < 5:
                lines.append(f"- **{name}** adds minimal value ({c['pct_contribution']:.1f}%). "
                             f"Consider removing for simplicity.")
    lines.append("")

    # === Regime Attribution ===
    lines.append("## Regime Attribution")
    lines.append("")
    lines.append(f"Total round-trip trades analyzed: {regime_result.total_trades}")
    lines.append("")

    # By Trend
    _write_regime_table(lines, "By Trend Strength (ADX)", regime_result.by_trend,
                        {'strong': 'Strong (ADX>25)', 'weak': 'Weak (15-25)',
                         'none': 'None (<15)', 'unknown': 'Unknown'})

    # By Volatility
    _write_regime_table(lines, "By Volatility (ATR Percentile)", regime_result.by_volatility,
                        {'high': 'High (>75th)', 'normal': 'Normal (25-75th)',
                         'low': 'Low (<25th)', 'unknown': 'Unknown'})

    # By Activity
    _write_regime_table(lines, "By Volume Activity", regime_result.by_activity,
                        {'active': 'Active (>1.5x)', 'normal': 'Normal (0.7-1.5x)',
                         'quiet': 'Quiet (<0.7x)', 'unknown': 'Unknown'})

    # Cross analysis
    if regime_result.cross_trend_vol:
        lines.append("### Cross Analysis: Trend x Volatility")
        lines.append("")
        lines.append("| | High Vol | Normal Vol | Low Vol |")
        lines.append("|--|:--:|:--:|:--:|")
        for tl in ['strong', 'weak', 'none']:
            label = {'strong': '**Strong Trend**', 'weak': '**Weak Trend**', 'none': '**No Trend**'}[tl]
            cells = []
            for vl in ['high', 'normal', 'low']:
                stats = regime_result.cross_trend_vol.get((tl, vl))
                if stats and stats.n_trades > 0:
                    cells.append(f"{stats.win_rate:.0f}% ({stats.n_trades})")
                else:
                    cells.append("—")
            lines.append(f"| {label} | {' | '.join(cells)} |")
        lines.append("")

    # Best/worst
    if regime_result.best_regime:
        lines.append(f"**Best regime**: {regime_result.best_regime}")
    if regime_result.worst_regime:
        lines.append(f"**Worst regime**: {regime_result.worst_regime}")
    lines.append("")

    # Regime insights
    if regime_result.by_trend:
        wr_values = {k: v.win_rate for k, v in regime_result.by_trend.items() if v.n_trades >= 2}
        if wr_values:
            best_wr = max(wr_values.values())
            worst_wr = min(wr_values.values())
            if best_wr - worst_wr > 20:
                lines.append(f"- Strategy has strong regime dependency: "
                             f"win rate ranges from {worst_wr:.0f}% to {best_wr:.0f}% "
                             f"across trend regimes.")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    # Write file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path


def _write_regime_table(lines, title, regime_dict, label_map):
    """Write a regime stats table to lines."""
    lines.append(f"### {title}")
    lines.append("")
    if not regime_dict:
        lines.append("*No data.*")
        lines.append("")
        return
    lines.append("| Regime | Trades | Win Rate | Avg PnL | Total PnL |")
    lines.append("|--------|:------:|:--------:|:-------:|:---------:|")
    for key, label in label_map.items():
        stats = regime_dict.get(key)
        if stats and stats.n_trades > 0:
            lines.append(
                f"| {label} | {stats.n_trades} | {stats.win_rate:.1f}% | "
                f"{stats.avg_pnl_pct:+.2f}% | {stats.total_pnl_pct:+.2f}% |"
            )
    lines.append("")


def run_full_attribution(test_results, load_strategy_fn, test_sets, output_dir=None):
    """Orchestrator: run attribution for all strategies with positive test Sharpe.

    Args:
        test_results: list of dicts from validate_all(), each with
            'version', 'params', 'test_AG', 'test_EC', etc.
        load_strategy_fn: callable(version) -> strategy class
        test_sets: dict like {"AG": ("2025-01-01", "2026-03-01"), ...}
        output_dir: directory for reports (default: research_log/attribution/)
    """
    from attribution.signal import run_signal_attribution
    from attribution.regime import run_regime_attribution

    if output_dir is None:
        from pathlib import Path
        output_dir = str(Path(__file__).resolve().parents[1] / "research_log" / "attribution")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for r in test_results:
        ver = r['version']
        params = r.get('params', {})

        for symbol, (start, end) in test_sets.items():
            sharpe = r.get(f'test_{symbol}')
            if sharpe is not None and sharpe > 0:
                try:
                    strategy_cls = load_strategy_fn(ver)
                    print(f"  Attribution: {ver} on {symbol}...", end=" ", flush=True)

                    signal_result = run_signal_attribution(
                        strategy_cls, params, symbol, start, end,
                    )
                    regime_result = run_regime_attribution(
                        strategy_cls, params, symbol, start, end,
                    )

                    out_path = os.path.join(output_dir, f"{ver}_{symbol}.md")
                    generate_attribution_report(signal_result, regime_result, out_path)
                    print(f"done ({out_path})")
                    count += 1
                except Exception as e:
                    print(f"FAILED ({e})")
                break

    print(f"\nAttribution reports generated: {count}/{len(test_results)}")
