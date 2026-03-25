"""
Portfolio Builder for Strong Trend Strategies — LC (碳酸锂)
============================================================
Runs all 50 strategies on LC 2025-06 → 2026-03 (+168% rally),
builds HRP + Risk Parity portfolios, saves detailed results.

Usage:
    python strategies/strong_trend/portfolio_builder_lc.py
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

QBASE_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.portfolio import (
    PortfolioConfig, StrategyAllocation, PortfolioBacktester,
)
from alphaforge.report import HTMLReportGenerator

from strategies.strong_trend.optimizer import (
    create_strategy_with_params, _map_freq, _resample_bars, DATA_DIR,
)

# =========================================================================
# Configuration
# =========================================================================

SYMBOL = "LC"
START = "2025-06-01"
END = "2026-03-01"
TOTAL_CAPITAL = 3_000_000

SHARPE_THRESHOLD = 0.3       # Lower than AG — LC is a single test set
CORR_THRESHOLD = 0.7

RESULTS_PATH = str(Path(QBASE_ROOT) / "strategies" / "strong_trend" / "optimization_results.json")
OUTPUT_DIR = str(Path(QBASE_ROOT) / "strategies" / "strong_trend" / "portfolio")


# =========================================================================
# Step 1: Run individual backtests
# =========================================================================

def run_single_test(version, params):
    """Run one strategy on LC, return (sharpe, maxdd, return, trades, equity_series)."""
    try:
        strategy = create_strategy_with_params(version, params)
        freq = strategy.freq
        loader = MarketDataLoader(DATA_DIR)
        spec_manager = ContractSpecManager()

        load_freq, resample_factor = _map_freq(freq)
        bars = loader.load(SYMBOL, freq=load_freq, start=START, end=END)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None

        if resample_factor > 1:
            bars = _resample_bars(bars, resample_factor)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None

        engine = EventDrivenBacktester(
            spec_manager=spec_manager,
            initial_capital=1_000_000,
            slippage_ticks=1.0,
        )
        result = engine.run(strategy, {SYMBOL: bars}, warmup=strategy.warmup)

        sharpe = result.sharpe
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None

        # Daily equity curve
        eq = result.equity_curve
        daily_eq = None
        if eq is not None and len(eq) > 0:
            if isinstance(eq, pd.Series):
                daily_eq = eq.resample("D").last().dropna()
            else:
                daily_eq = pd.Series(eq)

        return {
            "sharpe": float(sharpe),
            "total_return": float(result.total_return) if hasattr(result, "total_return") else None,
            "max_drawdown": float(result.max_drawdown) if hasattr(result, "max_drawdown") else None,
            "n_trades": int(result.n_trades) if hasattr(result, "n_trades") else None,
            "win_rate": float(result.win_rate) if hasattr(result, "win_rate") else None,
            "equity": daily_eq,
        }
    except Exception as e:
        print(f"  {version} error: {e}")
        return None


def collect_all_results():
    """Run all 50 strategies on LC."""
    with open(RESULTS_PATH) as f:
        opt_results = json.load(f)

    print("=" * 70)
    print(f"STEP 1: Running 50 strategies on {SYMBOL} ({START} → {END})")
    print("=" * 70)

    all_data = []
    for r in opt_results:
        ver = r["version"]
        params = r["best_params"]
        train_sharpe = r["best_sharpe"]

        res = run_single_test(ver, params)

        if res:
            strategy = create_strategy_with_params(ver, params)
            print(f"  {ver:>4s}: Sharpe={res['sharpe']:>7.3f}  "
                  f"Return={res['total_return']:>7.2%}  "
                  f"MaxDD={res['max_drawdown']:>7.2%}  "
                  f"Trades={res['n_trades'] or 0}")
            all_data.append({
                "version": ver,
                "params": params,
                "train_sharpe": train_sharpe,
                "freq": strategy.freq,
                **res,
            })
        else:
            print(f"  {ver:>4s}: FAILED")

    # Sort by Sharpe
    all_data.sort(key=lambda x: x["sharpe"], reverse=True)
    return all_data


# =========================================================================
# Step 2-4: Filter + HRP (reuse from AG builder)
# =========================================================================

def compute_correlation_matrix(data_list):
    """Compute daily return correlation from equity curves."""
    returns_dict = {}
    for d in data_list:
        eq = d.get("equity")
        if eq is not None and len(eq) > 10:
            if isinstance(eq, pd.Series):
                ret = eq.pct_change().dropna()
            else:
                ret = pd.Series(eq).pct_change().dropna()
            if len(ret) > 5:
                returns_dict[d["version"]] = ret

    if len(returns_dict) < 2:
        return None, returns_dict

    df = pd.DataFrame(returns_dict).fillna(0)
    return df.corr(), returns_dict


def filter_by_correlation(data_list, corr_matrix, threshold=CORR_THRESHOLD):
    """Remove highly correlated, keep higher Sharpe."""
    if corr_matrix is None:
        return data_list

    sorted_data = sorted(data_list, key=lambda x: x["sharpe"], reverse=True)
    kept = []
    removed = set()

    for d in sorted_data:
        ver = d["version"]
        if ver in removed or ver not in corr_matrix.columns:
            continue
        kept.append(d)
        for d2 in sorted_data:
            v2 = d2["version"]
            if v2 == ver or v2 in removed or v2 not in corr_matrix.columns:
                continue
            c = abs(corr_matrix.loc[ver, v2])
            if c > threshold:
                removed.add(v2)
                print(f"  Remove {v2} (corr={c:.3f} with {ver})")

    return kept


def hrp_weights(returns_dict, versions):
    """HRP weight allocation."""
    cols = [v for v in versions if v in returns_dict]
    if len(cols) < 2:
        return {v: 1.0 / len(versions) for v in versions}

    df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)
    cov = df.cov()
    corr = df.corr()

    dist = np.sqrt(0.5 * (1 - corr))
    link = linkage(squareform(dist.values, checks=False), method="single")
    sorted_cols = [cols[i] for i in leaves_list(link)]

    def _cluster_var(cov_mat, items):
        sub = cov_mat.loc[items, items]
        w = 1.0 / np.diag(sub)
        w = w / w.sum()
        return float(np.dot(w, np.dot(sub.values, w)))

    def _bisect(cov_mat, items):
        weights = pd.Series(1.0, index=items)
        clusters = [items]
        while clusters:
            new_c = []
            for cl in clusters:
                if len(cl) <= 1:
                    continue
                mid = len(cl) // 2
                left, right = cl[:mid], cl[mid:]
                vl, vr = _cluster_var(cov_mat, left), _cluster_var(cov_mat, right)
                alpha = 1.0 - vl / (vl + vr)
                weights[left] *= alpha
                weights[right] *= (1 - alpha)
                if len(left) > 1: new_c.append(left)
                if len(right) > 1: new_c.append(right)
            clusters = new_c
        return weights / weights.sum()

    w = _bisect(cov, sorted_cols)
    result = {v: float(w.get(v, 0)) for v in versions}
    total = sum(result.values())
    return {k: v / total for k, v in result.items()} if total > 0 else result


def risk_parity_weights(returns_dict, versions):
    """Risk parity: weight ~ 1/vol."""
    vols = {}
    for v in versions:
        if v in returns_dict:
            vol = returns_dict[v].std()
            if vol > 0:
                vols[v] = vol
    if not vols:
        return {v: 1.0 / len(versions) for v in versions}
    inv = {v: 1.0 / vol for v, vol in vols.items()}
    total = sum(inv.values())
    return {v: inv.get(v, 0) / total for v in versions}


def run_portfolio_backtest(kept_data, weights):
    """Run PortfolioBacktester on LC."""
    allocations = []
    for d in kept_data:
        ver = d["version"]
        w = weights.get(ver, 0)
        if w <= 0.001:
            continue
        allocations.append(StrategyAllocation(
            name=f"strong_trend_{ver}",
            strategy_file=f"strategies/strong_trend/{ver}.py",
            symbols=SYMBOL, freq=d["freq"], weight=w, params=d["params"],
        ))

    config = PortfolioConfig(
        total_capital=TOTAL_CAPITAL, allocations=allocations, rebalance="none",
    )
    runner = PortfolioBacktester(
        spec_manager=ContractSpecManager(), data_dir=DATA_DIR,
    )
    return runner.run(config, start=START, end=END)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    # Step 1
    all_data = collect_all_results()
    valid = [d for d in all_data if d["sharpe"] is not None]

    print(f"\n  Valid results: {len(valid)}/50")
    print(f"  Positive Sharpe: {sum(1 for d in valid if d['sharpe'] > 0)}")

    # Step 2: Sharpe filter
    filtered = [d for d in valid if d["sharpe"] >= SHARPE_THRESHOLD]
    print(f"\nSTEP 2: Sharpe filter (>={SHARPE_THRESHOLD}): {len(filtered)}/{len(valid)} pass")

    # Step 3: Correlation
    print(f"\nSTEP 3: Correlation analysis")
    corr, returns_dict = compute_correlation_matrix(filtered)
    kept = filter_by_correlation(filtered, corr)
    print(f"  Kept: {len(kept)} strategies")

    if corr is not None:
        upper = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_corr = upper.mean()
        print(f"  Avg correlation (pre-filter): {avg_corr:.3f}")

    # Recompute correlation for kept strategies only
    corr_kept, returns_kept = compute_correlation_matrix(kept)
    if corr_kept is not None:
        upper_kept = corr_kept.values[np.triu_indices_from(corr_kept.values, k=1)]
        avg_corr_kept = upper_kept.mean()
        print(f"  Avg correlation (post-filter): {avg_corr_kept:.3f}")

    # Step 4: HRP weights
    versions_kept = [d["version"] for d in kept]
    hrp_w = hrp_weights(returns_kept or returns_dict, versions_kept)
    rp_w = risk_parity_weights(returns_kept or returns_dict, versions_kept)

    print(f"\nSTEP 4: HRP Weights")
    for v in sorted(hrp_w, key=hrp_w.get, reverse=True):
        if hrp_w[v] > 0.001:
            sh = next((d["sharpe"] for d in kept if d["version"] == v), 0)
            print(f"  {v:>4s}: {hrp_w[v]*100:>5.1f}%  (Sharpe = {sh:.3f})")

    # Step 5: Portfolio backtests
    print(f"\nSTEP 5: Portfolio backtests on {SYMBOL}")

    hrp_report = None
    rp_report = None

    try:
        hrp_report = run_portfolio_backtest(kept, hrp_w)
        print(f"  HRP:  Sharpe={hrp_report.combined_result.sharpe:.3f}  "
              f"Return={hrp_report.combined_result.total_return:.2%}  "
              f"MaxDD={hrp_report.combined_result.max_drawdown:.2%}")
    except Exception as e:
        print(f"  HRP failed: {e}")

    try:
        rp_report = run_portfolio_backtest(kept, rp_w)
        print(f"  RP:   Sharpe={rp_report.combined_result.sharpe:.3f}  "
              f"Return={rp_report.combined_result.total_return:.2%}  "
              f"MaxDD={rp_report.combined_result.max_drawdown:.2%}")
    except Exception as e:
        print(f"  RP failed: {e}")

    # Save results
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    lc_result = {
        "symbol": SYMBOL,
        "period": f"{START} → {END}",
        "n_strategies": len(kept),
        "sharpe_threshold": SHARPE_THRESHOLD,
        "corr_threshold": CORR_THRESHOLD,
        "total_capital": TOTAL_CAPITAL,
        "individual_results": [],
        "strategies": {},
    }

    # All 50 individual results
    for d in all_data:
        lc_result["individual_results"].append({
            "version": d["version"],
            "sharpe": round(d["sharpe"], 3),
            "total_return": round(d.get("total_return") or 0, 4),
            "max_drawdown": round(d.get("max_drawdown") or 0, 4),
            "n_trades": d.get("n_trades"),
            "win_rate": round(d.get("win_rate") or 0, 3),
            "freq": d["freq"],
        })

    # Portfolio strategies
    for d in kept:
        ver = d["version"]
        lc_result["strategies"][ver] = {
            "weight_hrp": round(hrp_w.get(ver, 0), 4),
            "weight_rp": round(rp_w.get(ver, 0), 4),
            "sharpe": round(d["sharpe"], 3),
            "total_return": round(d.get("total_return") or 0, 4),
            "max_drawdown": round(d.get("max_drawdown") or 0, 4),
            "freq": d["freq"],
        }

    if hrp_report:
        lc_result["hrp_sharpe"] = round(hrp_report.combined_result.sharpe, 3)
        lc_result["hrp_return"] = round(hrp_report.combined_result.total_return, 4)
        lc_result["hrp_maxdd"] = round(hrp_report.combined_result.max_drawdown, 4)
    if rp_report:
        lc_result["rp_sharpe"] = round(rp_report.combined_result.sharpe, 3)
        lc_result["rp_return"] = round(rp_report.combined_result.total_return, 4)
        lc_result["rp_maxdd"] = round(rp_report.combined_result.max_drawdown, 4)
    if corr_kept is not None:
        lc_result["avg_correlation"] = round(avg_corr_kept, 3)

    weights_path = str(Path(OUTPUT_DIR) / "weights_lc.json")
    with open(weights_path, "w") as f:
        json.dump(lc_result, f, indent=2)
    print(f"\nResults saved to {weights_path}")

    # HTML report
    if hrp_report:
        try:
            report_path = str(Path(QBASE_ROOT) / "reports" / "portfolio_hrp_lc.html")
            reporter = HTMLReportGenerator()
            reporter.generate_portfolio_report(
                [s.result for s in hrp_report.slot_results],
                hrp_report.combined_result, report_path,
            )
            print(f"HTML report: {report_path}")
        except Exception as e:
            print(f"  HTML report failed: {e}")

    # Summary
    best = max(valid, key=lambda x: x["sharpe"])
    print(f"\n{'='*70}")
    print(f"PORTFOLIO SUMMARY — {SYMBOL} ({START} → {END})")
    print(f"{'='*70}")
    print(f"  Individual: {len(valid)} valid, {sum(1 for d in valid if d['sharpe']>0)} positive Sharpe")
    print(f"  Portfolio strategies: {len(kept)}")
    if corr_kept is not None:
        print(f"  Avg correlation: {avg_corr_kept:.3f}")
    if hrp_report:
        print(f"  HRP:  Sharpe={hrp_report.combined_result.sharpe:.3f}  "
              f"Return={hrp_report.combined_result.total_return:.2%}  "
              f"MaxDD={hrp_report.combined_result.max_drawdown:.2%}")
    if rp_report:
        print(f"  RP:   Sharpe={rp_report.combined_result.sharpe:.3f}  "
              f"Return={rp_report.combined_result.total_return:.2%}  "
              f"MaxDD={rp_report.combined_result.max_drawdown:.2%}")
    print(f"  Best single: {best['version']} Sharpe={best['sharpe']:.3f}")
