"""
Portfolio Builder — Strategy Selection + HRP Weighting + Bootstrap Validation
=============================================================================
Full pipeline: run all strategies → activity filter → exhaustive/greedy selection
(with drawdown overlap penalty) → Ledoit-Wolf shrinkage + Sharpe-weighted HRP →
weight cap → LOO validation → bootstrap → role annotation → tail risk metrics.

Requires an optimizer module in the strategy directory with:
  create_strategy_with_params, load_strategy_class, _map_freq, _resample_bars,
  DATA_DIR, STRATEGY_CLASSES

Usage:
    # Strong trend (default)
    python portfolio/builder.py --symbol AG --start 2025-01-01 --end 2026-03-01

    # With activity filter (for all_time strategies where some may not trade)
    python portfolio/builder.py --symbol AG --start 2022-01-01 --end 2026-02-24 --min-activity 0.001

    # With validation symbol
    python portfolio/builder.py --symbol AG --start 2025-01-01 --end 2026-03-01 \
        --validation-symbol EC --validation-start 2023-07-01 --validation-end 2024-09-01
"""
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import date
from itertools import combinations

warnings.filterwarnings("ignore")

QBASE_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, QBASE_ROOT)
import conftest  # noqa: F401

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester
from alphaforge.engine.portfolio import (
    PortfolioConfig, StrategyAllocation, PortfolioBacktester,
)
from alphaforge.report import HTMLReportGenerator

from strategies.optimizer_core import (
    create_strategy_with_params, map_freq, resample_bars, run_single_backtest,
)
from config import get_data_dir

DATA_DIR = get_data_dir()


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Portfolio Builder — strategy selection + HRP weighting + bootstrap validation"
    )
    parser.add_argument("--symbol", default="AG", help="Primary symbol (default: AG)")
    parser.add_argument("--start", default="2025-01-01", help="Test start date (default: 2025-01-01)")
    parser.add_argument("--end", default="2026-03-01", help="Test end date (default: 2026-03-01)")
    parser.add_argument("--validation-symbol", default="EC", help="Validation symbol (default: EC)")
    parser.add_argument("--validation-start", default="2023-07-01", help="Validation start (default: 2023-07-01)")
    parser.add_argument("--validation-end", default="2024-09-01", help="Validation end (default: 2024-09-01)")
    parser.add_argument("--capital", type=int, default=3_000_000, help="Total capital (default: 3000000)")
    parser.add_argument("--penalty-weight", type=float, default=0.1, help="Drawdown overlap penalty weight (default: 0.1)")
    parser.add_argument("--max-weight", type=float, default=0.20, help="Max single strategy weight (default: 0.20)")
    parser.add_argument("--min-activity", type=float, default=0.0,
                        help="Minimum abs(return) to be considered active (default: 0.0 = no filter, use 0.001 for 0.1%%)")
    parser.add_argument("--strategy-dir", default=None,
                        help="Strategy directory (default: strategies/strong_trend/)")
    parser.add_argument("--results-file", default=None,
                        help="Optimization results JSON (default: strategy_dir/optimization_results.json)")
    parser.add_argument("--output", default=None,
                        help="Output weights JSON path (default: strategy_dir/portfolio/weights_{symbol}.json)")
    parser.add_argument("--stability-test", type=int, default=0, metavar="N",
                        help="Run N stability test iterations after building (default: 0 = off, recommended: 50)")
    parser.add_argument("--stability-subsample", type=float, default=0.8,
                        help="Subsample fraction for stability test (default: 0.8)")
    return parser.parse_args()


# =========================================================================
# Step 1: Run individual backtests
# =========================================================================

def run_single_test(strategy_cls, params, symbol, start, end):
    """Run one strategy on one symbol, return dict with metrics + daily equity Series."""
    try:
        strategy = create_strategy_with_params(strategy_cls, params)
        freq = strategy.freq
        loader = MarketDataLoader(DATA_DIR)
        spec_manager = ContractSpecManager()

        load_freq, resample_factor = map_freq(freq)
        bars = loader.load(symbol, freq=load_freq, start=start, end=end)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None

        if resample_factor > 1:
            bars = resample_bars(bars, resample_factor)
        if bars is None or len(bars) < strategy.warmup + 20:
            return None

        engine = EventDrivenBacktester(
            spec_manager=spec_manager,
            initial_capital=1_000_000,
            slippage_ticks=1.0,
        )
        result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)

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
            "total_return": float(result.total_return) if hasattr(result, "total_return") else 0.0,
            "max_drawdown": float(result.max_drawdown) if hasattr(result, "max_drawdown") else 0.0,
            "n_trades": int(result.n_trades) if hasattr(result, "n_trades") else 0,
            "win_rate": float(result.win_rate) if hasattr(result, "win_rate") else 0.0,
            "equity": daily_eq,
        }
    except Exception as e:
        print(f"  {version} error: {e}")
        return None


def collect_all_results(symbol, start, end, val_symbol, val_start, val_end,
                        strategy_dir=None, results_file=None):
    """Run all strategies on primary + validation symbol.

    Args:
        strategy_dir: directory containing vN.py files (default: strategies/strong_trend/)
        results_file: path to optimization_results.json (default: strategy_dir/optimization_results.json)
    """
    if strategy_dir is None:
        strategy_dir = str(Path(QBASE_ROOT) / "strategies" / "strong_trend")
    strategy_dir = Path(strategy_dir)

    if results_file is None:
        results_file = strategy_dir / "optimization_results.json"
    results_file = Path(results_file)

    with open(results_file) as f:
        opt_results = json.load(f)

    # Filter out error results and empty params
    opt_results = [r for r in opt_results if "error" not in r and r.get("best_params")]

    print("=" * 70)
    print(f"STEP 1: Running {len(opt_results)} strategies on {symbol} ({start} -> {end})")
    if val_symbol:
        print(f"        + validation on {val_symbol} ({val_start} -> {val_end})")
    print(f"        Strategy dir: {strategy_dir}")
    print("=" * 70)

    # Dynamic strategy loading (works for any strategy directory)
    import importlib.util
    from alphaforge.strategy.base import TimeSeriesStrategy

    def _load_cls(version):
        filepath = strategy_dir / f"{version}.py"
        if not filepath.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"strat_{version}", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, TimeSeriesStrategy)
                    and attr is not TimeSeriesStrategy):
                return attr
        return None

    all_data = []
    for r in opt_results:
        ver = r["version"]
        params = r["best_params"]
        train_sharpe = r["best_sharpe"]

        strategy_cls = _load_cls(ver)
        if strategy_cls is None:
            print(f"  {ver:>5s}: SKIP (file not found)")
            continue

        # Primary symbol test
        res = run_single_test(strategy_cls, params, symbol, start, end)

        # Validation symbol test
        val_res = None
        if val_symbol:
            val_res = run_single_test(strategy_cls, params, val_symbol, val_start, val_end)

        primary_sharpe = res["sharpe"] if res else None
        val_sharpe = val_res["sharpe"] if val_res else None

        strategy = create_strategy_with_params(strategy_cls, params)
        freq = strategy.freq

        print(f"  {ver:>5s}: Train={train_sharpe:>7.3f}  "
              f"{symbol}={primary_sharpe or 0:>7.3f}  "
              f"{val_symbol or 'N/A'}={val_sharpe or 0:>7.3f}  "
              f"freq={freq}")

        all_data.append({
            "version": ver,
            "params": params,
            "train_sharpe": train_sharpe,
            "primary_sharpe": primary_sharpe,
            "primary_return": res["total_return"] if res else None,
            "primary_maxdd": res["max_drawdown"] if res else None,
            "primary_equity": res["equity"] if res else None,
            "val_sharpe": val_sharpe,
            "val_return": val_res["total_return"] if val_res else None,
            "val_maxdd": val_res["max_drawdown"] if val_res else None,
            "n_trades": res["n_trades"] if res else None,
            "win_rate": res["win_rate"] if res else None,
            "freq": freq,
        })

    return all_data


# =========================================================================
# Drawdown Overlap Penalty
# =========================================================================

def mean_drawdown_overlap(returns_df):
    """Average pairwise drawdown overlap (Jaccard similarity).

    Drawdown = equity below -1% from peak.
    """
    n = returns_df.shape[1]
    if n < 2:
        return 0.0
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            eq_i = (1 + returns_df.iloc[:, i]).cumprod()
            eq_j = (1 + returns_df.iloc[:, j]).cumprod()
            dd_i = eq_i / eq_i.cummax() - 1
            dd_j = eq_j / eq_j.cummax() - 1
            in_dd_i = dd_i < -0.01
            in_dd_j = dd_j < -0.01
            intersection = (in_dd_i & in_dd_j).sum()
            union = (in_dd_i | in_dd_j).sum()
            overlaps.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(overlaps)) if overlaps else 0.0


def calc_portfolio_sharpe_with_penalty(subset_returns_df, penalty_weight=0.1):
    """Portfolio Sharpe with drawdown overlap penalty (equal weights for selection)."""
    n_strats = subset_returns_df.shape[1]
    if n_strats == 0:
        return -999.0
    # Equal weight during selection phase
    combined = subset_returns_df.mean(axis=1)
    if combined.std() == 0:
        return 0.0
    base_sharpe = combined.mean() / combined.std() * np.sqrt(252)
    overlap = mean_drawdown_overlap(subset_returns_df)
    return base_sharpe * (1 - penalty_weight * overlap)


# =========================================================================
# Step 2: Strategy Selection (Exhaustive or Bidirectional Greedy)
# =========================================================================

def build_returns_df(all_data):
    """Build aligned daily returns DataFrame from equity curves."""
    returns_dict = {}
    for d in all_data:
        eq = d.get("primary_equity")
        if eq is not None and len(eq) > 10:
            if isinstance(eq, pd.Series):
                ret = eq.pct_change().dropna()
            else:
                ret = pd.Series(eq).pct_change().dropna()
            if len(ret) > 5:
                returns_dict[d["version"]] = ret
    return returns_dict


def exhaustive_search(pool_versions, returns_dict, penalty_weight=0.1):
    """Iterate all 2^N subsets of size >= 3 to find max portfolio Sharpe."""
    n = len(pool_versions)
    print(f"  Exhaustive search over {n} strategies (2^{n} = {2**n} subsets)")

    # Build aligned returns df for pool
    cols = [v for v in pool_versions if v in returns_dict]
    df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)

    best_sharpe = -np.inf
    best_subset = None
    total_evaluated = 0

    for size in range(3, len(cols) + 1):
        for subset in combinations(cols, size):
            sub_df = df[list(subset)]
            s = calc_portfolio_sharpe_with_penalty(sub_df, penalty_weight)
            total_evaluated += 1
            if s > best_sharpe:
                best_sharpe = s
                best_subset = list(subset)

    print(f"  Evaluated {total_evaluated} subsets")
    print(f"  Best subset: {len(best_subset)} strategies, Sharpe={best_sharpe:.4f}")
    return best_subset, best_sharpe


def bidirectional_greedy(pool_versions, returns_dict, penalty_weight=0.1):
    """Forward-add then backward-delete, repeat until stable."""
    cols = [v for v in pool_versions if v in returns_dict]
    df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)

    # Sort by individual Sharpe descending for starting order
    individual_sharpes = {}
    for v in cols:
        ret = df[v]
        if ret.std() > 0:
            individual_sharpes[v] = ret.mean() / ret.std() * np.sqrt(252)
        else:
            individual_sharpes[v] = 0.0
    sorted_pool = sorted(cols, key=lambda v: individual_sharpes[v], reverse=True)

    # Phase 1: Forward greedy
    selected = [sorted_pool[0]]
    for v in sorted_pool[1:]:
        candidate = selected + [v]
        sub_df_candidate = df[candidate]
        sub_df_current = df[selected]
        s_candidate = calc_portfolio_sharpe_with_penalty(sub_df_candidate, penalty_weight)
        s_current = calc_portfolio_sharpe_with_penalty(sub_df_current, penalty_weight)
        if s_candidate > s_current:
            selected.append(v)

    if len(selected) < 3:
        # Ensure minimum 3
        for v in sorted_pool:
            if v not in selected:
                selected.append(v)
            if len(selected) >= 3:
                break

    # Iterate forward-add + backward-delete until stable
    stable = False
    iteration = 0
    while not stable and iteration < 10:
        iteration += 1
        stable = True

        # Backward delete
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                without = [x for x in selected if x != v]
                if len(without) < 3:
                    continue
                s_without = calc_portfolio_sharpe_with_penalty(df[without], penalty_weight)
                s_current = calc_portfolio_sharpe_with_penalty(df[selected], penalty_weight)
                if s_without > s_current:
                    selected.remove(v)
                    improved = True
                    stable = False
                    print(f"    Backward: removed {v}")
                    break  # restart inner loop

        # Forward add
        for v in sorted_pool:
            if v in selected:
                continue
            candidate = selected + [v]
            s_candidate = calc_portfolio_sharpe_with_penalty(df[candidate], penalty_weight)
            s_current = calc_portfolio_sharpe_with_penalty(df[selected], penalty_weight)
            if s_candidate > s_current:
                selected.append(v)
                stable = False
                print(f"    Forward: added {v}")

    current_sharpe = calc_portfolio_sharpe_with_penalty(df[selected], penalty_weight)
    print(f"  Bidirectional greedy: {len(selected)} strategies, Sharpe={current_sharpe:.4f}")
    return selected, current_sharpe


def select_strategies(all_data, returns_dict, penalty_weight=0.1):
    """Step 2: Select strategy subset maximizing portfolio Sharpe (with penalty).

    Uses exhaustive search if pool <= 20, bidirectional greedy otherwise.
    No Sharpe threshold filter. No correlation threshold filter.
    """
    # Pool = all strategies with valid returns
    pool = [d["version"] for d in all_data if d["version"] in returns_dict]
    print(f"\nSTEP 2: Strategy Selection (pool = {len(pool)} strategies)")

    if len(pool) <= 20:
        print("  Method: EXHAUSTIVE (pool <= 20)")
        selected, best_sharpe = exhaustive_search(pool, returns_dict, penalty_weight)
    else:
        print("  Method: BIDIRECTIONAL GREEDY (pool > 20)")
        selected, best_sharpe = bidirectional_greedy(pool, returns_dict, penalty_weight)

    print(f"  Selected: {selected}")
    return selected, best_sharpe


# =========================================================================
# Step 3: Ledoit-Wolf Shrinkage + Sharpe-weighted HRP
# =========================================================================

def hrp_weights_from_cov(cov_shrunk, corr_shrunk):
    """HRP weight allocation using external covariance and correlation matrices."""
    cols = list(cov_shrunk.columns)
    if len(cols) < 2:
        return {v: 1.0 / len(cols) for v in cols}

    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr_shrunk))
    # Ensure diagonal is zero (numerical precision)
    np.fill_diagonal(dist.values, 0.0)
    dist_condensed = squareform(dist.values, checks=False)

    # Hierarchical clustering
    link = linkage(dist_condensed, method="single")
    sort_ix = leaves_list(link)
    sorted_cols = [cols[i] for i in sort_ix]

    # Recursive bisection
    def _get_cluster_var(cov_mat, cluster_items):
        cov_sub = cov_mat.loc[cluster_items, cluster_items]
        diag = np.diag(cov_sub.values)
        # Avoid division by zero
        diag = np.where(diag > 0, diag, 1e-10)
        w_ivp = 1.0 / diag
        w_ivp = w_ivp / w_ivp.sum()
        return float(np.dot(w_ivp, np.dot(cov_sub.values, w_ivp)))

    def _recursive_bisect(cov_mat, sorted_items):
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = _get_cluster_var(cov_mat, left)
                var_right = _get_cluster_var(cov_mat, right)

                total_var = var_left + var_right
                if total_var == 0:
                    alpha = 0.5
                else:
                    alpha = 1.0 - var_left / total_var

                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            clusters = new_clusters

        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    w = _recursive_bisect(cov_shrunk, sorted_cols)
    return {v: float(w.get(v, 0)) for v in cols}


def compute_shrunk_hrp_sharpe_weights(selected, returns_dict, strategy_sharpes):
    """Step 3: Ledoit-Wolf covariance shrinkage + Sharpe-weighted HRP.

    Returns final weights dict (before capping).
    """
    cols = [v for v in selected if v in returns_dict]
    df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)
    returns_matrix = df

    # Ledoit-Wolf shrinkage
    lw = LedoitWolf().fit(returns_matrix.values)
    cov_shrunk = pd.DataFrame(lw.covariance_, index=cols, columns=cols)

    # Convert to correlation
    std = np.sqrt(np.diag(cov_shrunk.values))
    std = np.where(std > 0, std, 1e-10)
    corr_shrunk = cov_shrunk.values / np.outer(std, std)
    # Clamp to [-1, 1]
    corr_shrunk = np.clip(corr_shrunk, -1.0, 1.0)
    np.fill_diagonal(corr_shrunk, 1.0)
    corr_shrunk = pd.DataFrame(corr_shrunk, index=cols, columns=cols)

    # HRP using shrunk covariance
    w_hrp = hrp_weights_from_cov(cov_shrunk, corr_shrunk)

    # Sharpe weighting
    sharpe_factor = {}
    for v in cols:
        s = strategy_sharpes.get(v, 0.0)
        sharpe_factor[v] = max(0.1, s)

    w_final = {v: w_hrp[v] * sharpe_factor[v] for v in cols}
    total = sum(w_final.values())
    if total > 0:
        w_final = {v: w / total for v, w in w_final.items()}
    else:
        w_final = {v: 1.0 / len(cols) for v in cols}

    return w_final, returns_matrix, cov_shrunk, corr_shrunk


# =========================================================================
# Step 4: Weight Cap
# =========================================================================

def apply_weight_cap(weights, max_weight=0.20):
    """Iteratively clip and redistribute weights exceeding max_weight."""
    weights = dict(weights)  # copy
    for _ in range(100):  # safety limit
        if not any(w > max_weight + 1e-9 for w in weights.values()):
            break
        excess_total = 0.0
        for v, w in list(weights.items()):
            if w > max_weight:
                excess_total += w - max_weight
                weights[v] = max_weight
        under_cap = {v: w for v, w in weights.items() if w < max_weight - 1e-9}
        under_total = sum(under_cap.values())
        if under_total > 0:
            for v in under_cap:
                weights[v] += excess_total * (under_cap[v] / under_total)
    # Final normalize
    total = sum(weights.values())
    if total > 0:
        weights = {v: w / total for v, w in weights.items()}
    return weights


# =========================================================================
# Step 5: Leave-One-Out Validation
# =========================================================================

def leave_one_out(selected, returns_dict, strategy_sharpes, penalty_weight=0.1, max_weight=0.20):
    """Iteratively remove strategies that hurt portfolio Sharpe.

    After removal, re-run Steps 3-4 to get final weights.
    """
    print(f"\nSTEP 5: Leave-One-Out validation (starting with {len(selected)} strategies)")

    # Current portfolio Sharpe (with penalty, equal weights for selection metric)
    cols = [v for v in selected if v in returns_dict]
    df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)
    current_sharpe = calc_portfolio_sharpe_with_penalty(df, penalty_weight)

    improved = True
    while improved:
        improved = False
        for v in list(selected):
            without = [x for x in selected if x != v]
            if len(without) < 3:
                continue
            cols_w = [x for x in without if x in returns_dict]
            df_w = pd.DataFrame({x: returns_dict[x] for x in cols_w}).fillna(0)
            sharpe_without = calc_portfolio_sharpe_with_penalty(df_w, penalty_weight)
            if sharpe_without > current_sharpe:
                selected.remove(v)
                current_sharpe = sharpe_without
                improved = True
                print(f"  Removed {v}: Sharpe improved to {current_sharpe:.4f}")
                break  # restart

    print(f"  After LOO: {len(selected)} strategies, Sharpe={current_sharpe:.4f}")

    # Re-run Step 3-4 with remaining strategies
    weights, returns_matrix, cov_shrunk, corr_shrunk = compute_shrunk_hrp_sharpe_weights(
        selected, returns_dict, strategy_sharpes
    )
    weights = apply_weight_cap(weights, max_weight)

    return selected, weights, returns_matrix, cov_shrunk, corr_shrunk


# =========================================================================
# Step 6: Bootstrap Validation
# =========================================================================

def bootstrap_validation(portfolio_returns, selected, returns_dict, strategy_sharpes,
                         n_bootstrap=1000, n_weight_samples=100):
    """Bootstrap Sharpe CI and weight stability."""
    print(f"\nSTEP 6: Bootstrap validation ({n_bootstrap} Sharpe samples, {n_weight_samples} weight samples)")

    # Sharpe bootstrap
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        sample = portfolio_returns.sample(frac=1.0, replace=True)
        s_std = sample.std()
        if s_std > 0:
            s = sample.mean() / s_std * np.sqrt(252)
        else:
            s = 0.0
        bootstrap_sharpes.append(float(s))

    ci_lower = float(np.percentile(bootstrap_sharpes, 2.5))
    ci_upper = float(np.percentile(bootstrap_sharpes, 97.5))
    sharpe_mean = float(np.mean(bootstrap_sharpes))

    print(f"  Sharpe mean={sharpe_mean:.3f}  95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]  "
          f"width={ci_upper - ci_lower:.3f}")

    # Weight stability: subsample 90% of returns, re-run HRP
    cols = [v for v in selected if v in returns_dict]
    df_full = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)
    weight_samples = {v: [] for v in cols}

    for _ in range(n_weight_samples):
        n_rows = len(df_full)
        idx = np.random.choice(n_rows, size=int(n_rows * 0.9), replace=False)
        sub_returns = df_full.iloc[idx]

        try:
            # Re-run shrunk HRP on subsample
            lw = LedoitWolf().fit(sub_returns.values)
            cov_s = pd.DataFrame(lw.covariance_, index=cols, columns=cols)
            std_s = np.sqrt(np.diag(cov_s.values))
            std_s = np.where(std_s > 0, std_s, 1e-10)
            corr_s = cov_s.values / np.outer(std_s, std_s)
            corr_s = np.clip(corr_s, -1.0, 1.0)
            np.fill_diagonal(corr_s, 1.0)
            corr_s = pd.DataFrame(corr_s, index=cols, columns=cols)

            w_hrp = hrp_weights_from_cov(cov_s, corr_s)

            # Sharpe weighting
            for v in cols:
                sf = max(0.1, strategy_sharpes.get(v, 0.0))
                w_hrp[v] *= sf
            total_w = sum(w_hrp.values())
            if total_w > 0:
                w_hrp = {v: w / total_w for v, w in w_hrp.items()}

            for v in cols:
                weight_samples[v].append(w_hrp.get(v, 0.0))
        except Exception:
            continue

    # Compute CV per strategy
    weight_stability = {}
    for v in cols:
        samples = weight_samples[v]
        if len(samples) > 1:
            mean_w = float(np.mean(samples))
            std_w = float(np.std(samples))
            cv = std_w / mean_w if mean_w > 0 else 0.0
            weight_stability[v] = {"mean": round(mean_w, 4), "std": round(std_w, 4), "cv": round(cv, 3)}
        else:
            weight_stability[v] = {"mean": 0.0, "std": 0.0, "cv": 0.0}

    return {
        "n_samples": n_bootstrap,
        "sharpe_mean": round(sharpe_mean, 3),
        "sharpe_ci_lower": round(ci_lower, 3),
        "sharpe_ci_upper": round(ci_upper, 3),
        "ci_width": round(ci_upper - ci_lower, 3),
        "weight_stability": weight_stability,
    }


# =========================================================================
# Step 7: Role Annotation
# =========================================================================

def annotate_roles(selected, returns_dict, strategy_sharpes, final_weights):
    """Assign roles: core, satellite, or hedge (non-binding, for report only)."""
    cols = [v for v in selected if v in returns_dict]
    df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)

    sharpes = [strategy_sharpes.get(v, 0.0) for v in cols]
    median_sharpe = float(np.median(sharpes))

    roles = {}
    for v in cols:
        # Correlation of v's returns with portfolio returns excluding v
        others = [x for x in cols if x != v]
        if not others:
            roles[v] = "core"
            continue
        # Portfolio returns excluding v (equal weight)
        port_ex = df[others].mean(axis=1)
        corr_with_rest = float(df[v].corr(port_ex)) if port_ex.std() > 0 else 0.0

        if strategy_sharpes.get(v, 0.0) >= median_sharpe and corr_with_rest > 0.2:
            roles[v] = "core"
        elif strategy_sharpes.get(v, 0.0) > 0 and corr_with_rest < 0.4:
            roles[v] = "satellite"
        else:
            roles[v] = "hedge"

    return roles


# =========================================================================
# Step 8: Tail Risk Metrics
# =========================================================================

def compute_tail_risk(portfolio_returns):
    """Compute CVaR, tail ratio, max single day loss, omega ratio."""
    if len(portfolio_returns) < 10:
        return {}

    q05 = float(portfolio_returns.quantile(0.05))
    q95 = float(portfolio_returns.quantile(0.95))

    cvar_95 = float(portfolio_returns[portfolio_returns <= q05].mean()) if q05 != 0 else 0.0
    tail_ratio = abs(q95) / abs(q05) if q05 != 0 else 0.0
    max_single_day_loss = float(portfolio_returns.min())

    # Omega ratio (threshold=0)
    gains = float(portfolio_returns[portfolio_returns > 0].sum())
    losses = abs(float(portfolio_returns[portfolio_returns <= 0].sum()))
    omega = gains / losses if losses > 0 else float('inf')

    return {
        "cvar_95": round(cvar_95, 6),
        "tail_ratio": round(tail_ratio, 3),
        "max_single_day_loss": round(max_single_day_loss, 6),
        "omega_ratio": round(omega, 3),
    }


# =========================================================================
# DD Duration
# =========================================================================

def compute_dd_duration(equity_series):
    """Max consecutive days the portfolio equity was below its previous peak."""
    if equity_series is None or len(equity_series) < 2:
        return 0
    eq = np.array(equity_series, dtype=float)
    peak = np.maximum.accumulate(eq)
    in_dd = eq < peak
    max_dur = 0
    current_dur = 0
    for dd in in_dd:
        if dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return int(max_dur)


# =========================================================================
# Portfolio Backtest via AlphaForge
# =========================================================================

def run_portfolio_backtest(all_data, weights, symbol, start, end, capital):
    """Run combined portfolio backtest via PortfolioBacktester."""
    allocations = []
    for d in all_data:
        ver = d["version"]
        w = weights.get(ver, 0)
        if w <= 0.001:
            continue
        allocations.append(
            StrategyAllocation(
                name=f"strong_trend_{ver}",
                strategy_file=str(Path(QBASE_ROOT) / "strategies" / "strong_trend" / f"{ver}.py"),
                symbols=symbol,
                freq=d["freq"],
                weight=w,
                params=d["params"],
            )
        )

    if not allocations:
        return None

    config = PortfolioConfig(
        total_capital=capital,
        allocations=allocations,
        rebalance="none",
    )

    runner = PortfolioBacktester(
        spec_manager=ContractSpecManager(),
        data_dir=DATA_DIR,
    )
    report = runner.run(config, start=start, end=end)
    return report


# =========================================================================
# Main Pipeline
# =========================================================================

def main():
    args = parse_args()

    symbol = args.symbol.upper()
    start = args.start
    end = args.end
    val_symbol = args.validation_symbol.upper() if args.validation_symbol else None
    val_start = args.validation_start
    val_end = args.validation_end
    capital = args.capital
    penalty_weight = args.penalty_weight
    max_weight = args.max_weight
    min_activity = args.min_activity

    print(f"\n{'='*70}")
    print(f"PORTFOLIO BUILDER — {symbol} ({start} -> {end})")
    print(f"Capital: {capital:,}  Penalty: {penalty_weight}  Max Weight: {max_weight:.0%}")
    if min_activity > 0:
        print(f"Activity filter: abs(return) > {min_activity:.1%}")
    print(f"{'='*70}\n")

    # ---- Step 1: Run all strategies ----
    all_data = collect_all_results(symbol, start, end, val_symbol, val_start, val_end,
                                    strategy_dir=args.strategy_dir,
                                    results_file=args.results_file)
    valid = [d for d in all_data if d["primary_sharpe"] is not None]
    n_total = len(all_data)
    print(f"\n  Valid results: {len(valid)}/{n_total}")
    print(f"  Positive Sharpe: {sum(1 for d in valid if (d['primary_sharpe'] or 0) > 0)}")

    # ---- Activity filter ----
    if min_activity > 0:
        active = [d for d in all_data
                  if d["primary_sharpe"] is not None
                  and abs(d["primary_return"] or 0) > min_activity]
        n_filtered = len(all_data) - len(active)
        print(f"  Activity filter: {len(active)} active, {n_filtered} filtered out")
        build_data = active
    else:
        build_data = all_data

    # ---- Build returns dict ----
    returns_dict = build_returns_df(build_data)
    if len(returns_dict) < 3:
        print("ERROR: Fewer than 3 strategies with valid equity curves. Cannot build portfolio.")
        return

    # Strategy Sharpes (primary symbol)
    strategy_sharpes = {}
    for d in build_data:
        if d["primary_sharpe"] is not None:
            strategy_sharpes[d["version"]] = d["primary_sharpe"]

    # ---- Step 2: Strategy Selection ----
    selected, selection_sharpe = select_strategies(all_data, returns_dict, penalty_weight)

    # ---- Step 3: Shrunk HRP + Sharpe weighting ----
    print(f"\nSTEP 3: Ledoit-Wolf Shrinkage + Sharpe-weighted HRP")
    weights, returns_matrix, cov_shrunk, corr_shrunk = compute_shrunk_hrp_sharpe_weights(
        selected, returns_dict, strategy_sharpes
    )
    print("  Pre-cap weights:")
    for v in sorted(weights, key=weights.get, reverse=True):
        if weights[v] > 0.001:
            print(f"    {v:>4s}: {weights[v]*100:>5.1f}%  (Sharpe={strategy_sharpes.get(v, 0):.3f})")

    # ---- Step 4: Weight Cap ----
    print(f"\nSTEP 4: Weight Cap at {max_weight:.0%}")
    weights = apply_weight_cap(weights, max_weight)
    print("  Post-cap weights:")
    for v in sorted(weights, key=weights.get, reverse=True):
        if weights[v] > 0.001:
            print(f"    {v:>4s}: {weights[v]*100:>5.1f}%")

    # ---- Step 5: Leave-One-Out ----
    selected, weights, returns_matrix, cov_shrunk, corr_shrunk = leave_one_out(
        list(selected), returns_dict, strategy_sharpes, penalty_weight, max_weight
    )

    print("\n  Final weights after LOO:")
    for v in sorted(weights, key=weights.get, reverse=True):
        if weights[v] > 0.001:
            print(f"    {v:>4s}: {weights[v]*100:>5.1f}%")

    # ---- Run PortfolioBacktester ----
    print(f"\n{'='*70}")
    print(f"Running PortfolioBacktester on {symbol} ({start} -> {end})")
    print(f"{'='*70}")

    # Filter all_data to selected strategies for portfolio backtest
    selected_data = [d for d in all_data if d["version"] in selected]
    portfolio_report = None
    try:
        portfolio_report = run_portfolio_backtest(selected_data, weights, symbol, start, end, capital)
        if portfolio_report:
            cr = portfolio_report.combined_result
            print(f"  Portfolio Sharpe: {cr.sharpe:.3f}")
            print(f"  Portfolio Return: {cr.total_return:.2%}")
            print(f"  Portfolio MaxDD:  {cr.max_drawdown:.2%}")
    except Exception as e:
        print(f"  Portfolio backtest failed: {e}")

    # ---- Extract daily returns from PortfolioBacktester ----
    portfolio_daily_returns = None
    portfolio_equity = None
    if portfolio_report and portfolio_report.combined_result.equity_curve is not None:
        eq = portfolio_report.combined_result.equity_curve
        if isinstance(eq, pd.Series) and hasattr(eq.index, 'freq'):
            portfolio_equity = eq.resample("D").last().dropna()
        elif isinstance(eq, pd.Series) and isinstance(eq.index, pd.DatetimeIndex):
            portfolio_equity = eq.resample("D").last().dropna()
        else:
            # Non-dated equity — convert to Series but mark as non-daily
            portfolio_equity = pd.Series(np.array(eq, dtype=float))
        portfolio_daily_returns = portfolio_equity.pct_change().dropna()

    # If no portfolio report equity, fall back to manual calculation
    if portfolio_daily_returns is None or len(portfolio_daily_returns) < 5:
        print("  Falling back to manual daily returns calculation")
        cols = [v for v in selected if v in returns_dict]
        df = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)
        w_arr = np.array([weights.get(v, 0) for v in cols])
        portfolio_daily_returns = (df.values * w_arr).sum(axis=1)
        portfolio_daily_returns = pd.Series(portfolio_daily_returns, index=df.index)

    # ---- Step 6: Bootstrap ----
    bootstrap_result = bootstrap_validation(
        portfolio_daily_returns, selected, returns_dict, strategy_sharpes
    )

    # ---- Step 7: Role Annotation ----
    print(f"\nSTEP 7: Role Annotation")
    roles = annotate_roles(selected, returns_dict, strategy_sharpes, weights)
    for v in sorted(weights, key=weights.get, reverse=True):
        if weights.get(v, 0) > 0.001:
            print(f"  {v:>4s}: {roles.get(v, 'N/A'):>10s}  weight={weights[v]*100:>5.1f}%")

    # ---- Step 8: Tail Risk ----
    print(f"\nSTEP 8: Tail Risk Metrics")
    tail_risk = compute_tail_risk(portfolio_daily_returns)
    for k, v in tail_risk.items():
        print(f"  {k}: {v}")

    # ---- Compute additional portfolio metrics ----
    # Avg correlation (from selected strategies)
    cols = [v for v in selected if v in returns_dict]
    df_sel = pd.DataFrame({v: returns_dict[v] for v in cols}).fillna(0)
    corr_matrix = df_sel.corr()
    if len(cols) > 1:
        upper = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        # Filter NaN values before averaging (can happen with zero-variance columns)
        valid_upper = upper[~np.isnan(upper)]
        avg_corr = float(np.mean(valid_upper)) if len(valid_upper) > 0 else 0.0
    else:
        avg_corr = 0.0

    # Avg drawdown overlap
    avg_dd_overlap = mean_drawdown_overlap(df_sel)

    # DD duration — always compute from daily returns to ensure unit = trading days
    if portfolio_daily_returns is not None and len(portfolio_daily_returns) > 0:
        daily_eq = (1 + portfolio_daily_returns).cumprod()
        dd_duration = compute_dd_duration(daily_eq)
    else:
        dd_duration = 0

    # Portfolio metrics from backtester (preferred) or manual
    if portfolio_report:
        cr = portfolio_report.combined_result
        port_sharpe = float(cr.sharpe)
        port_return = float(cr.total_return)
        port_maxdd = float(cr.max_drawdown)
        # Annual return
        n_days = len(portfolio_daily_returns)
        if n_days > 0 and port_return > -1:
            annual_return = (1 + port_return) ** (252 / max(n_days, 1)) - 1
        else:
            annual_return = 0.0
        calmar = abs(annual_return / port_maxdd) if port_maxdd != 0 else 0.0
    else:
        # Manual fallback
        if len(portfolio_daily_returns) > 0 and portfolio_daily_returns.std() > 0:
            port_sharpe = float(portfolio_daily_returns.mean() / portfolio_daily_returns.std() * np.sqrt(252))
        else:
            port_sharpe = 0.0
        port_return = float((1 + portfolio_daily_returns).prod() - 1) if len(portfolio_daily_returns) > 0 else 0.0
        eq_manual = (1 + portfolio_daily_returns).cumprod()
        port_maxdd = float((eq_manual / eq_manual.cummax() - 1).min()) if len(eq_manual) > 0 else 0.0
        n_days = len(portfolio_daily_returns)
        annual_return = (1 + port_return) ** (252 / max(n_days, 1)) - 1 if port_return > -1 else 0.0
        calmar = abs(annual_return / port_maxdd) if port_maxdd != 0 else 0.0

    # ---- Build output JSON ----
    # Determine method name
    pool_with_returns = [d["version"] for d in all_data if d["version"] in returns_dict]
    method = "exhaustive_hrp_sharpe" if len(pool_with_returns) <= 20 else "greedy_hrp_sharpe"

    output = {
        "meta": {
            "method": method,
            "symbol": symbol,
            "test_start": start,
            "test_end": end,
            "validation_symbol": val_symbol or "",
            "validation_start": val_start or "",
            "validation_end": val_end or "",
            "total_capital": capital,
            "n_strategies_total": n_total,
            "min_activity": min_activity,
            "n_strategies_selected": len(selected),
            "build_date": str(date.today()),
        },
        "strategies": {},
        "portfolio_metrics": {
            "sharpe": round(port_sharpe, 3),
            "total_return": round(port_return, 4),
            "max_drawdown": round(port_maxdd, 4),
            "annual_return": round(annual_return, 4),
            "calmar": round(calmar, 3),
            "avg_correlation": round(avg_corr, 3) if not np.isnan(avg_corr) else 0.0,
            "avg_dd_overlap": round(avg_dd_overlap, 3),
            **{k: v for k, v in tail_risk.items()},
        },
        "bootstrap": bootstrap_result,
        "individual_results": [],
        "daily_returns": [round(float(x), 8) for x in portfolio_daily_returns.values],
        "dd_duration_days": dd_duration,
        "correlation_matrix": {},
    }

    # Strategies detail
    for v in selected:
        d = next((x for x in all_data if x["version"] == v), None)
        if d is None:
            continue
        output["strategies"][v] = {
            "weight": round(weights.get(v, 0), 4),
            "sharpe": round(d["primary_sharpe"] or 0, 3),
            "total_return": round(d["primary_return"] or 0, 4),
            "max_drawdown": round(d["primary_maxdd"] or 0, 4),
            "freq": d["freq"],
            "role": roles.get(v, "unknown"),
            "params": d["params"],
        }

    # Individual results (all 50)
    for d in all_data:
        output["individual_results"].append({
            "version": d["version"],
            "sharpe": round(d["primary_sharpe"] or 0, 3),
            "total_return": round(d["primary_return"] or 0, 4),
            "max_drawdown": round(d["primary_maxdd"] or 0, 4),
            "freq": d["freq"],
        })

    # Correlation matrix (selected strategies only)
    corr_dict = {}
    for v1 in cols:
        corr_dict[v1] = {}
        for v2 in cols:
            val = corr_matrix.loc[v1, v2]
            corr_dict[v1][v2] = round(float(val), 4) if not np.isnan(val) else 0.0
    output["correlation_matrix"] = corr_dict

    # ---- Save JSON ----
    if args.output:
        weights_path = args.output
        output_dir = Path(weights_path).parent
    elif args.strategy_dir:
        output_dir = Path(args.strategy_dir) / "portfolio"
        weights_path = str(output_dir / f"weights_{symbol.lower()}.json")
    else:
        output_dir = Path(QBASE_ROOT) / "strategies" / "strong_trend" / "portfolio"
        weights_path = str(output_dir / f"weights_{symbol.lower()}.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {weights_path}")

    # ---- HTML Report ----
    if portfolio_report:
        try:
            reports_dir = Path(QBASE_ROOT) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = str(reports_dir / f"portfolio_hrp_{symbol.lower()}.html")
            reporter = HTMLReportGenerator()
            reporter.generate_portfolio_report(
                [s.result for s in portfolio_report.slot_results],
                portfolio_report.combined_result,
                report_path,
            )
            print(f"HTML report: {report_path}")
        except Exception as e:
            print(f"  HTML report failed: {e}")

    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print(f"PORTFOLIO SUMMARY — {symbol} ({start} -> {end})")
    print(f"{'='*70}")
    print(f"  Strategies selected: {len(selected)}/{len(valid)} valid")
    print(f"  Method: {method}")
    print(f"  Portfolio Sharpe:  {port_sharpe:.3f}")
    print(f"  Portfolio Return:  {port_return:.2%}")
    print(f"  Portfolio MaxDD:   {port_maxdd:.2%}")
    print(f"  Annual Return:     {annual_return:.2%}")
    print(f"  Calmar Ratio:      {calmar:.3f}")
    print(f"  Avg Correlation:   {avg_corr:.3f}")
    print(f"  Avg DD Overlap:    {avg_dd_overlap:.3f}")
    print(f"  DD Duration:       {dd_duration} days")
    print(f"  Bootstrap Sharpe:  {bootstrap_result['sharpe_mean']:.3f} "
          f"[{bootstrap_result['sharpe_ci_lower']:.3f}, {bootstrap_result['sharpe_ci_upper']:.3f}]")
    for k, v in tail_risk.items():
        print(f"  {k}: {v}")

    print(f"\n  Strategy weights:")
    for v in sorted(weights, key=weights.get, reverse=True):
        if weights[v] > 0.001:
            role = roles.get(v, "?")
            sharpe = strategy_sharpes.get(v, 0)
            print(f"    {v:>4s}: {weights[v]*100:>5.1f}%  role={role:<10s}  Sharpe={sharpe:.3f}")

    best_single = max(valid, key=lambda x: x["primary_sharpe"] or -999)
    print(f"\n  Best single ({symbol}): {best_single['version']} Sharpe={best_single['primary_sharpe']:.3f}")
    print(f"  Portfolio / Best single: {port_sharpe / max(best_single['primary_sharpe'], 0.001):.2f}x")

    # ---- Optional: Stability Test ----
    if args.stability_test > 0:
        from portfolio.stability_test import (
            run_stability_test, print_stability_report, save_stability_result,
        )
        # Build full aligned returns DataFrame for stability test
        pool_with_returns = [d["version"] for d in all_data if d["version"] in returns_dict]
        full_returns_df = pd.DataFrame(
            {v: returns_dict[v] for v in pool_with_returns}
        ).fillna(0)

        stability_result = run_stability_test(
            returns_df=full_returns_df,
            pool_versions=pool_with_returns,
            strategy_sharpes=strategy_sharpes,
            n_runs=args.stability_test,
            frac=args.stability_subsample,
            penalty_weight=penalty_weight,
            max_weight=max_weight,
            seed=42,
        )
        stability_result["symbol"] = symbol

        print_stability_report(stability_result)

        stability_path = save_stability_result(stability_result, symbol)
        print(f"\n  Stability results saved to {stability_path}")


if __name__ == "__main__":
    main()
