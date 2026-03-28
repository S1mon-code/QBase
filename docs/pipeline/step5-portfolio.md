# Step 5: Portfolio 构建（Portfolio Construction）

> 从多个验证+归因完成的策略中，构建最优组合。代码：[`portfolio/builder.py`](../../portfolio/builder.py)，评分：[`portfolio/scorer.py`](../../portfolio/scorer.py)

## 核心原则

**组合的唯一目标：最大化组合风险调整收益（Sharpe + CVaR 约束），同时控制回撤。**

不是把好策略堆在一起，而是找到一组**互补**的策略。一个负 Sharpe 但与其他策略负相关的策略，可能比一个正 Sharpe 但高度相关的策略更有价值。

---

## 数据分割（三段式 — 新 Portfolio 必须遵守）

构建 Portfolio 本身也是一种"模型选择"，必须防止选择偏差：

```
全部数据
├── 训练集 (60%)    → 策略参数优化（Optuna）
├── 验证集 (20%)    → 策略选择 + 权重优化（Step 1-5）
└── 测试集 (20%)    → 最终评估（不改任何东西，只看结果）
```

**所有新 Portfolio 必须使用三段式分割。** 旧 Portfolio（如 Strong Trend AG）沿用现有分割，但后续迭代时应迁移到三段式。

**绝不在测试集上做任何决策。** 验证集上选策略、算权重；测试集只用于最终报告。如果测试集表现远差于验证集，说明过拟合了，需要回去简化。

---

## 构建流程总览

```
开发 N 个策略 → 全部在验证集回测 → 穷举/贪心组合优化（含回撤重叠分析）
→ 协方差收缩 + Sharpe 加权 HRP → 权重上限裁剪
→ Leave-one-out 检验 → Bootstrap 稳健性验证
→ 测试集最终评估 → 最终 portfolio
```

---

## Step 1: 组合优化（策略选择）

**不使用 Sharpe 门槛过滤，不使用硬性相关性阈值。** 以组合 Sharpe 最大化为目标选择策略子集。

### 方法选择

| 策略池大小 | 方法 | 理由 |
|-----------|------|------|
| N ≤ 20 | **穷举搜索** | 2^20 ≈ 100 万种组合，几分钟内完成，保证全局最优 |
| 20 < N ≤ 50 | **双向贪心**（先加后删交替迭代） | 比单向贪心更接近全局最优 |
| N > 50 | **遗传算法/模拟退火** | 近似全局最优 |

### 穷举搜索（推荐，N ≤ 20）

```python
from itertools import combinations

best_sharpe = -np.inf
best_subset = None

for size in range(3, len(pool) + 1):  # 至少 3 个策略
    for subset in combinations(pool, size):
        portfolio_sharpe = calc_portfolio_sharpe(subset)
        if portfolio_sharpe > best_sharpe:
            best_sharpe = portfolio_sharpe
            best_subset = subset
```

### 双向贪心（N > 20 时的后备方案）

```python
# Phase 1: 前向贪心 — 逐个加入
pool = sort_by_sharpe_descending(all_strategies)
selected = [pool[0]]
for strategy in pool[1:]:
    candidate = selected + [strategy]
    if calc_portfolio_sharpe(candidate) > calc_portfolio_sharpe(selected):
        selected.append(strategy)

# Phase 2: 后向删除 — 逐个尝试移除
improved = True
while improved:
    improved = False
    for s in list(selected):
        without = [x for x in selected if x != s]
        if len(without) >= 3 and calc_portfolio_sharpe(without) > calc_portfolio_sharpe(selected):
            selected.remove(s)
            improved = True

# Phase 3: 重复 Phase 1-2 直到稳定（通常 2-3 轮）
```

**关键：负 Sharpe + 负相关性的策略会被自然纳入。** 比如一个 Sharpe=-0.3 但与组合相关性 -0.5 的策略，加入后可能让组合 MaxDD 从 -10% 降到 -6%，整体 Sharpe 反而提升。

### 回撤重叠惩罚

组合 Sharpe 计算时使用回撤重叠惩罚：

```python
def calc_portfolio_sharpe(subset, penalty_weight=0.1):
    """组合 Sharpe，带回撤重叠惩罚。"""
    base_sharpe = weighted_sharpe(subset)  # 标准组合 Sharpe

    # 回撤重叠率：策略两两之间回撤期的 Jaccard 相似度
    overlap = mean_drawdown_overlap(subset)
    # overlap ∈ [0, 1]，0=完全不重叠，1=完全重叠

    return base_sharpe * (1 - penalty_weight * overlap)
```

回撤重叠比收益率相关性更重要 — 两个策略日收益相关性 0.3 但回撤在同一周发生，组合在那一周就会爆。

---

## Step 2: 协方差收缩 + Sharpe 加权 HRP

### Step 2a: 协方差矩阵收缩估计

原始样本协方差在 N 接近 T 时噪音极大（17 个策略、几百个日收益观测）。必须做收缩：

```python
from sklearn.covariance import LedoitWolf

# Ledoit-Wolf 收缩估计（自动选择最优收缩系数）
lw = LedoitWolf().fit(returns_matrix)
cov_shrunk = pd.DataFrame(lw.covariance_, index=strategies, columns=strategies)
corr_shrunk = cov_to_corr(cov_shrunk)

# 用收缩后的协方差做 HRP
w_hrp = hrp_weights(cov_shrunk, corr_shrunk)
```

### Step 2b: Sharpe 加权

```python
# HRP 基础权重（基于收缩协方差）
w_hrp = hrp_weights(cov_shrunk, corr_shrunk)

# Sharpe 调整因子
sharpe_factor = {v: max(0.1, sharpe) for v, sharpe in strategy_sharpes.items()}
# 对负 Sharpe 策略（因负相关被纳入）给予最低权重 0.1

# 混合权重 = HRP × Sharpe 调整，然后归一化
w_final = {v: w_hrp[v] * sharpe_factor[v] for v in strategies}
w_final = normalize(w_final)  # 归一化到 sum=1
```

**效果：** 高 Sharpe + 低相关的策略获得最多权重。低 Sharpe 的"对冲策略"获得适当但不过多的权重。

### 角色化分析（参考，不强制约束权重）

可以在报告中将策略标注角色，辅助理解组合结构，但不作为硬性约束：

| 角色 | 特征 | 参考权重 |
|------|------|---------|
| **核心层** | Sharpe ≥ 中位数，与组合正相关 | 通常占 50-60% |
| **卫星层** | Sharpe > 0，与核心层低相关 | 通常占 25-35% |
| **对冲层** | 与组合负/零相关（含负 Sharpe） | 通常占 10-15% |

如果 Sharpe 加权 HRP 的结果明显偏离上述参考范围（如对冲层占 40%），可以作为复查信号，但不自动调整。

---

## Step 3: 权重上限裁剪

**单策略最大权重 20%。** 超出部分按比例重新分配给其他策略：

```python
MAX_WEIGHT = 0.20

while any(w > MAX_WEIGHT for w in weights.values()):
    excess = {v: w - MAX_WEIGHT for v, w in weights.items() if w > MAX_WEIGHT}
    for v in excess:
        weights[v] = MAX_WEIGHT
    # 将多余权重按比例分配给未达上限的策略
    total_excess = sum(excess.values())
    under_cap = {v: w for v, w in weights.items() if w < MAX_WEIGHT}
    for v in under_cap:
        weights[v] += total_excess * (weights[v] / sum(under_cap.values()))
```

**为什么 20%：** 实测中 v41 在 AG 上拿了 24%、v31 在 LC 上拿了 22%，单策略集中度过高。20% 上限保证至少 5 个策略共同承担风险，同时给优质策略足够空间。

---

## Step 4: Leave-one-out 边际检验

构建完成后，逐个移除策略验证其边际贡献：

```python
for strategy in portfolio:
    without = portfolio - {strategy}
    sharpe_without = calc_portfolio_sharpe(without)

    if sharpe_without > current_sharpe:
        # 去掉这个策略反而更好 → 移除它
        portfolio.remove(strategy)
        current_sharpe = sharpe_without
        print(f"Removed {strategy}: portfolio Sharpe improved to {sharpe_without}")
```

**这是最后的清理步骤。** 穷举/贪心构建时选入的策略，在其他策略也加入后可能变成冗余。Leave-one-out 检验能发现并清理这些冗余。

---

## Step 5: Bootstrap 稳健性验证

对最终组合做 Bootstrap 检验，量化结果的可信度：

```python
n_bootstrap = 1000
bootstrap_sharpes = []

for _ in range(n_bootstrap):
    # 有放回重抽样日收益
    sample = portfolio_daily_returns.sample(frac=1.0, replace=True)
    bootstrap_sharpes.append(calc_sharpe(sample))

ci_lower = np.percentile(bootstrap_sharpes, 2.5)
ci_upper = np.percentile(bootstrap_sharpes, 97.5)
print(f"Portfolio Sharpe: {portfolio_sharpe:.3f}  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**通过标准：**
- 95% CI 下界 > 0（Sharpe 显著大于零）
- CI 宽度 < 1.5（结果足够稳定）
- 如果 CI 跨零，说明结果不可靠，需要简化组合或增加数据

### 权重稳定性测试

```python
# 随机去掉 10% 数据，重新计算权重，重复 100 次
weight_samples = []
for _ in range(100):
    subset = portfolio_returns.sample(frac=0.9)
    w = full_pipeline(subset)  # 重新跑 Step 2-3
    weight_samples.append(w)

# 每个策略权重的变异系数 (CV)
for strategy in portfolio:
    weights_list = [ws[strategy] for ws in weight_samples]
    cv = np.std(weights_list) / np.mean(weights_list)
    print(f"  {strategy}: mean={np.mean(weights_list):.3f}  CV={cv:.2f}")
    # CV > 0.5 的策略权重不稳定，考虑降权或移除
```

---

## Step 6: Portfolio 质量要求

**Portfolio 评分必须达到 B+ (75分) 以上才可使用。**

### 评分维度（4 维 12 指标）

| 维度 | 权重 | 包含指标 | 关注点 |
|------|:----:|---------|--------|
| **收益风险比** | 40% | Sharpe, Calmar, MaxDD, 回撤持续时间, **CVaR 95%** | 赚多少、亏多少、亏多久 |
| **组合质量** | 25% | 平均相关性, **回撤重叠率**, Portfolio vs Best Single, 正 Sharpe 比例 | 分散有效吗、组合有增值吗 |
| **实操性** | 20% | 策略数量, 最大单策略权重, 频率多样性, **Bootstrap CI 宽度** | 能不能实际跑起来 |
| **稳定性** | 15% | 收益一致性, 权益曲线稳定性, **Tail Ratio** | 是不是靠运气 |

### 各指标评分标准 (0-10)

**Sharpe Ratio:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| ≥3.0 | 10 | 卓越 |
| 2.0 | 6.7 | 优秀 |
| 1.0 | 3.3 | 及格 |
| ≤0 | 0 | 不合格 |

**Calmar Ratio (年化收益/最大回撤):**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| ≥10 | 10 | 极低风险高收益 |
| 5 | 5 | 中等 |
| 1 | 1 | 勉强 |

**Max Drawdown:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| <3% | 10 | 极低风险 |
| <5% | 8.5 | 低风险 |
| <10% | 6 | 中等 |
| <20% | 3 | 高风险 |
| >30% | 0 | 不可接受 |

**平均相关性:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| <0.2 | 10 | 完美分散 |
| <0.3 | 8.5 | 优秀 |
| <0.5 | 5.5 | 及格 |
| >0.7 | 1 | 无分散价值 |

**Portfolio / 最佳单策略 Sharpe 比:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| ≥1.15 | 9 | 组合显著增值 |
| ≥1.0 | 8 | 组合优于单策略 |
| ≥0.8 | 6 | 可接受 |
| <0.6 | 1 | 组合没有价值 |

**最大单策略权重:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| <10% | 10 | 充分分散 |
| <15% | 8 | 良好 |
| <25% | 5 | 集中风险 |
| >30% | 2 | 过度集中 |

### 尾部风险指标（必须纳入评估）

| 指标 | 含义 | 最低要求 | 理想目标 |
|------|------|---------|---------|
| CVaR 95% | 最差 5% 情况平均损失 | > -3% (日) | > -1.5% (日) |
| Tail Ratio | 右尾95分位 / |左尾5分位| | > 0.8 | > 1.2 |
| 最大单日亏损 | 极端事件暴露 | > -5% | > -3% |
| Omega Ratio | 收益分布完整比较 | > 1.5 | > 2.0 |

### 等级映射

| 分数 | 等级 | 含义 |
|:----:|:----:|------|
| 90+ | A+ | 卓越 — 可直接进入模拟盘 |
| 85-89 | A | 优秀 — 小幅优化后可上线 |
| 80-84 | A- | 良好 — 需要补充 walk-forward 验证 |
| 75-79 | B+ | 中上 — 核心指标过关，部分维度待提升 |
| 70-74 | B | 中等 — 可用但有明显弱点 |
| 60-69 | C | 勉强 — 需要重大改进 |
| <60 | D/F | 不合格 — 重新构建 |

### 质量门槛汇总（统一两级制）

**Tier 1: 策略级门槛（进入 Portfolio 的最低标准）**

| 指标 | 要求 |
|------|------|
| 验证集 Sharpe | > 0 |
| 归因分析 | 已完成（信号归因 + 行情归因） |
| 交易次数 | 达到频率门槛（daily≥10, 4h≥20, 1h≥30, 5min≥80） |
| Walk-Forward Win Rate | ≥ 50%（对关键策略） |

**Tier 2: Portfolio 级门槛（组合上线的最低标准）**

| 指标 | 最低要求 | 理想目标 |
|------|---------|---------|
| 综合评分 | ≥ 75 (B+) | ≥ 85 (A) |
| Regime 覆盖 | 无 RED FLAG（不能所有策略都依赖同一 regime） | 2/3 regime 有正 PnL 策略 |
| Bootstrap 95% CI 下界 | > 0 | > 0.5 |
| Portfolio Sharpe / 最佳单策略 | ≥ 0.8 | ≥ 1.0 |
| Max Drawdown | < -15% | < -5% |
| 策略间平均相关性 | < 0.5 | < 0.3 |
| 回撤重叠率 | < 0.5 | < 0.3 |
| 最大单策略权重 | ≤ 20% | ≤ 10% |

### 评分不达标的迭代指引

如果评分低于 B+，必须在 `research_log/` 中分析原因并迭代：

- 组合 Sharpe 低于单策略？→ 检查 Sharpe 加权和角色分配是否合理
- 回撤过大？→ 检查回撤重叠率，补充对冲层策略
- 相关性过高？→ 扩大策略池（加入不同频率/指标类型）
- 某策略权重过高？→ 降低权重上限
- Bootstrap CI 跨零？→ 数据不足或过拟合，简化组合

---

## Step 7: Portfolio 止损标准

| 级别 | 触发条件 | 动作 |
|------|---------|------|
| **预警** | 组合回撤达 **-10%** | 记录日志，检查各策略状态，不自动操作 |
| **减仓** | 组合回撤达 **-15%** | 所有策略仓位减半，暂停加仓 |
| **熔断** | 组合回撤达 **-20%** | 全部平仓，停止交易，人工审查后重启 |
| **单日熔断** | 单日亏损达权益 **-5%** | 当日全部平仓，次日恢复 |

**熔断后的恢复规则：**

- 减仓后：回撤收窄至 -10% 以内，恢复正常仓位
- 全面熔断后：必须人工审查并确认市场环境正常后才能重启
- 连续 2 次触发 -20% 熔断：停止该 portfolio，重新优化权重

---

## Step 8: 重平衡（待定）

重平衡机制后续讨论确定。

---

## 活跃度过滤（所有 portfolio 必用）

**构建 portfolio 时不允许有僵尸策略。** 僵尸策略（测试期几乎不交易、收益接近 0）会导致：

1. 与其他策略的相关性为 NaN（方差为 0 无法计算 correlation）
2. 被错误地当作"对冲策略"选入 portfolio（因为 NaN 相关性被视为 0 相关）
3. 稀释资金利用率，占用权重但不产生收益

**所有策略类型都必须加活跃度过滤：**

```bash
# 所有 portfolio 构建都加 --min-activity
python portfolio/builder.py --symbol AG --min-activity 0.001  # strong_trend
python strategies/all_time/ag/build_portfolio.py --min-activity 0.001  # all_time
```

```python
# 过滤逻辑
active = [d for d in all_data
          if d["primary_sharpe"] is not None
          and abs(d["primary_return"] or 0) > min_activity]
```

**经验值：** 所有策略类型统一使用 `--min-activity 0.001`（abs(return) > 0.1%）。

**教训（AG v26 事件）：** v26 (TTM Squeeze + ADX + Force Index) 在 AG 测试期仅 4 笔交易、收益 -0.19%，日收益率几乎全为 0。被 portfolio builder 选为 "hedge"（因 NaN 相关性），占 9.3% 权重。加入活跃度过滤后被正确剔除，portfolio 从 7 策略变 6 策略，Sharpe 反而从 3.05 提升到 3.08。

---

## Portfolio 构建工具使用

**通用工具位置：** `portfolio/builder.py` + `portfolio/scorer.py`

### Strong Trend Portfolio（品种无关策略）

```bash
python portfolio/builder.py --symbol AG --start 2025-01-01 --end 2026-03-01
python portfolio/scorer.py strategies/strong_trend/portfolio/weights_ag.json
```

### All-Time AG Portfolio（品种专用策略）

```bash
python strategies/all_time/ag/build_portfolio.py
python portfolio/scorer.py strategies/all_time/ag/portfolio/weights_ag.json
```

`build_portfolio.py` 是一个薄包装器，将 `portfolio/builder.py` 的策略加载接口替换为 `all_time/ag/optimizer.py`，并设置正确的默认参数（测试期 2022-2026、活跃度过滤 0.1%）。

### 更多用法

```bash
# 带活跃度过滤（all_time 策略用）
python portfolio/builder.py --symbol AG --start 2022-01-01 --end 2026-02-24 --min-activity 0.001

# 带验证品种
python portfolio/builder.py --symbol AG --start 2025-01-01 --end 2026-03-01 \
    --validation-symbol EC --validation-start 2023-07-01 --validation-end 2024-09-01

# 调整参数
python portfolio/builder.py --symbol AG --max-weight 0.15 --penalty-weight 0.2 --capital 5000000

# 评分目录下所有组合
python portfolio/scorer.py --dir strategies/strong_trend/portfolio/
```

---

## 多频率策略混合

Portfolio 中的策略可能混合 5min、1h、4h、daily 等不同频率：

### 收益率对齐

所有策略的权益曲线统一 resample 到**日线级别**后再做相关性计算和赋权。

```python
daily_equity = resample_to_daily(strategy_equity_curve)
daily_returns = daily_equity.pct_change()
```

### 仓位管理注意

- 高频策略（5min/10min）日内可能多次开平仓，日线策略可能持仓数天
- 两者同时运行时，总保证金占用可能在盘中瞬间叠加
- 风控的总敞口限制（80%）按**实时合计**，不是按日线合计

### 策略容量约束

- 高频策略（5min/10min）在低流动性品种上有容量限制
- 分配资金不能超过策略容量上限（以滑点敏感性测试为准）
- 滑点敏感性：2x 滑点下 Sharpe 下降 >30% 的策略，权重上限额外收紧至 10%

### 信号冲突处理

- 不同频率策略可能对同一品种产生相反信号（如 daily 做多，5min 做空）
- 在 `all_time/` 中这是允许的（多空均可），但需要注意净头寸
- 在 `strong_trend/` 中不会冲突（都是只做多）

---

## All-Time AG 策略架构

**100 个策略，5 类，多空全天候：**

| 类别 | 版本 | 数量 | 方向 | 特点 |
|------|------|:---:|:---:|------|
| 趋势跟踪 | v1-v20 | 20 | 多空 | Supertrend/ADX/Aroon/EMA/Ichimoku 等 |
| 均值回归 | v21-v40 | 20 | 多空 | RSI/Stochastic/BB/Z-Score/CCI 等 |
| 突破 | v41-v60 | 20 | 多空 | Donchian/TTM Squeeze/Keltner/NR7 等 |
| 多周期+量价 | v61-v80 | 20 | 多空 | 4h+1h 多周期, OBV/MFI/Klinger 等 |
| 自适应/混合 | v81-v100 | 20 | 多空 | Regime 切换/多指标投票/双策略模式 |

**与 strong_trend 的区别：**

| 维度 | strong_trend | all_time/ag |
|------|-------------|-------------|
| 方向 | 只做多 | **多空均可** |
| 品种 | 品种无关（跨品种训练） | **AG 专用**（单品种训练） |
| 行情覆盖 | 只做强趋势行情 | **全天候（趋势+震荡+突破）** |
| 训练数据 | J/ZC/JM/I/NI/SA 多品种 | AG 2012-2021 全历史 |
| 测试数据 | AG/EC 强趋势时段 | AG 2022-2026（含各种行情） |
| 策略数量 | 50 | 100 |
| optimizer | 手动定义参数空间 | **自动检测参数空间** |

**优化器特性（`optimizer.py`）：**
- 调用 `optimizer_core` 的复合目标函数、两阶段优化、稳健性检查
- 自动从策略类的 type annotations 检测参数及范围
- 支持批量优化：`--strategy v1-v20 --trials 150`

---

## All-Time Iron Ore (I) 策略架构

**200 个策略，5 类×40，多空全天候，最低频率 1h：**

| 类别 | 版本 | 数量 | 频率分布 | 指标特点 |
|------|------|:---:|---------|---------|
| 趋势跟踪 | v1-v40 | 40 | daily(10), 4h(15), 1h(15) | Supertrend/ADX/Aroon/Ichimoku/PSAR/HMA/EMA Ribbon 等 |
| 均值回归 | v41-v80 | 40 | daily(5), 4h(15), 1h(20) | Bollinger/Keltner/Z-score/RSI/Stochastic/CCI/Fisher 等 |
| 突破 | v81-v120 | 40 | daily(10), 4h(15), 1h(15) | Donchian/TTM Squeeze/NR7/Range Expansion/Chandelier 等 |
| 多周期 | v121-v160 | 40 | 1h+4h(20), 4h+daily(20) | 大周期定方向+小周期精确入场，lookahead-free 映射 |
| 自适应/混合 | v161-v200 | 40 | daily(5), 4h(20), 1h(15) | HMM regime/Kalman/投票(≥2/3)/双模切换(趋势↔均值回归) |

**品种特性：**
- 数据：2013-10 ~ 2026-02（12.4 年 1min 数据）
- 乘数 100，保证金 12%，涨跌停 ±8%
- 流动性 A 级（>100 万手/日）
- 板块：黑色系（J/JM/RB 强相关）
- 驱动：供给侧政策、产能冲击、房地产周期

**数据分割：**
- 训练集：2013-10 ~ 2021-12-31（8.3 年）
- 测试集：2022-01-01 ~ 2026-02

**优化结果（粗调 50 trials/strategy）：**
- 200 策略中 89 个正 Sharpe（44%），最高 1.77（v35 TEMA+PPO 1h）
- 自适应类最稳定：33/40 正 Sharpe，0 失败
- 均值回归和突破类失败率较高（信号在铁矿上偏稀疏）
- 1h 频率策略整体表现优于 daily

**用法：**

```bash
# 粗调
python strategies/all_time/i/optimizer.py --strategy all --trials 50 --phase coarse

# 精调
python strategies/all_time/i/optimizer.py --strategy v35,v96,v124 --trials 100 --phase fine

# 多种子（top 候选）
python strategies/all_time/i/optimizer.py --strategy v35 --trials 50 --multi-seed
```

---

## 当前 Portfolio 成果

| Portfolio | 品种 | Sharpe | Return | MaxDD | 策略数 | 说明 |
|-----------|------|:------:|:------:|:-----:|:-----:|------|
| **Strong Trend (通用 C)** | AG | **2.58** | **66.13%** | -12.80% | 5 | 通用组合，同一权重跑任何品种 |
| **Strong Trend (通用 C)** | LC | **2.37** | **19.19%** | -3.99% | 5 | 同上 |
| **AG All-Time** | AG | 优化完成 | — | — | 158 valid | 待构建 Portfolio |
| **Medium Trend** | 多品种 | 优化中 | — | — | 200 策略 | V4 迁移完成，全频率优化中 |

**运行评分：**

```bash
python portfolio/scorer.py
```

---

## Regime 覆盖验证（建议）

> **来自归因分析的启示：** 组合中如果所有策略都依赖强趋势 regime，弱趋势下会集体失效。

建议流程：

1. 收集每个策略的 regime 归因结果
2. 构建 regime 覆盖矩阵
3. 确保组合在至少 2/3 的 regime 类型下有正 PnL 的策略
4. 如发现 regime 盲区，补充相应策略

**现已有自动化工具支持：**

```bash
# 批量归因：对 Portfolio 中所有策略运行信号+行情归因
python -m attribution.batch --portfolio <weights_file>

# Regime 覆盖矩阵：自动检测 RED FLAG（某 regime 下无正 PnL 策略）
python -m attribution.coverage --portfolio <weights_file>

# 回撤归因：分析回撤期间各策略和 regime 的贡献
python -m attribution.drawdown --portfolio <weights_file>
```

这些工具应在 Portfolio 构建完成后、最终评估前运行，作为 Tier 2 质量门槛的一部分。

---

## 权重文件格式

```json
{
  "symbol": "AG",
  "period": "2025-01-01 ~ 2026-03-01",
  "capital": 3000000,
  "strategies": {
    "v12": {"weight": 0.20, "freq": "daily", "role": "trend_capture"},
    "v11": {"weight": 0.18, "freq": "daily", "role": "volume_signal"},
    ...
  },
  "metrics": {
    "sharpe": 2.58,
    "max_drawdown": -0.085,
    "calmar": 4.2,
    "bootstrap_ci_95": [1.8, 3.3]
  }
}
```

---

## 鲁棒性测试（Portfolio 上线前必做）

Portfolio 构建完成后、最终评估前，必须通过以下鲁棒性测试：

### 滑点敏感性测试（必须）

测试 Portfolio 中每个策略在不同滑点水平下的表现衰减：

```bash
python tests/robustness/slippage_test.py --strategy <strategy> --symbol <symbol>
```

**判定标准：**

| 判定 | 含义 | 动作 |
|------|------|------|
| **LOW** | 2x 滑点下 Sharpe 下降 < 15% | 正常使用 |
| **MODERATE** | 2x 滑点下 Sharpe 下降 15-30% | 可用，但权重上限收紧至 10% |
| **HIGH** | 2x 滑点下 Sharpe 下降 > 30% | 从 Portfolio 中移除或大幅降权 |

### Monte Carlo 压力测试（必须）

对 Portfolio 整体做 1000 次 Bootstrap 重采样，评估在各种随机序列下的稳健性：

```bash
python tests/robustness/stress_test.py --strategy <strategy> --symbol <symbol>
```

**判定标准：**

| 判定 | 含义 | 动作 |
|------|------|------|
| **ROBUST** | 95% CI 下界 > 0，CVaR 可控 | 可以上线 |
| **ACCEPTABLE** | 95% CI 下界接近 0，CVaR 偏高 | 可用，需要更严格的 Portfolio 止损 |
| **FRAGILE** | 95% CI 跨零或 CVaR 极端 | 不能上线，需要简化或重构 |

### 选择稳定性测试（推荐）

测试 Portfolio 策略选择在数据扰动下的稳定性，将每个策略分类为 CORE/SATELLITE/EDGE：

```bash
# 独立运行
python portfolio/stability_test.py --portfolio <weights_file>

# 或在 builder 中集成运行
python portfolio/builder.py --symbol AG --stability-test 100
```

**策略分类：**

| 分类 | 含义 | 说明 |
|------|------|------|
| **CORE** | 在 >80% 的扰动中被选入 | 组合的核心成员，权重可信 |
| **SATELLITE** | 在 50-80% 的扰动中被选入 | 有价值但不稳定，权重应保守 |
| **EDGE** | 在 <50% 的扰动中被选入 | 边缘策略，考虑移除或大幅降权 |

如果 Portfolio 中 EDGE 策略占比 > 30%，说明组合选择不稳定，需要扩大策略池或简化选择标准。

---

## Portfolio Checklist

1. 所有候选策略已完成 Step 1-4.5
2. 活跃度过滤（`--min-activity 0.001`）
3. 运行 `portfolio/builder.py` 生成权重
4. 检查 LOO — 去掉任一策略 Sharpe 不应暴跌
5. 检查 Bootstrap CI — 95% 下界 > 0
6. 检查 regime 覆盖 — 组合不应全依赖同一 regime
7. Portfolio Sharpe > 最佳单策略的 80%
8. **滑点敏感性测试** — 所有策略判定不为 HIGH（`slippage_test.py`）
9. **Monte Carlo 压力测试** — 整体判定不为 FRAGILE（`stress_test.py`）
10. **选择稳定性测试**（推荐） — EDGE 策略占比 < 30%（`stability_test.py`）
11. 评分 ≥ 75 分 (B+)（via scorer.py）
12. 保存权重到 `strategies/<category>/portfolio/weights_<symbol>.json`
13. Commit: `[portfolio] <category> <symbol> portfolio weights`
