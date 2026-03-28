# Step 4.5: 归因分析（Attribution Analysis）

> 测试集验证通过后，分析策略的 alpha 来源和行情依赖性。代码：[`attribution/`](../../attribution/)

## 核心原则

**每个进入 Portfolio 的策略必须通过归因分析。** 不是只看 Sharpe，还要理解 alpha 来源。

- 不做归因 → 你以为 v12 靠 Aroon+PPO+VolMom 三方合力赚钱
- 做了归因 → 发现 VolMom 贡献 51.7%，Aroon 贡献 0.1%，PPO 贡献 2.8%

归因决定：策略是否简化、Portfolio 权重是否合理、regime 覆盖是否完整。

---

## 两层自动归因

### 1. 信号归因（Signal Attribution — Ablation Test）

对每个指标做 ablation test：替换为中性值，重跑回测，看 Sharpe 下降多少。

**核心公式：**

```
Contribution = Baseline Sharpe - Ablated Sharpe
```

**贡献度解读：**

| 贡献占比 | 含义 | 行动 |
|---------|------|------|
| > 60% | 核心依赖 | 需要评估单点失效风险 |
| 5% - 60% | 有效贡献 | 保留 |
| < 5% | 冗余指标 | 考虑移除以降低复杂度 |

**中性值来源（两种方式）：**

1. **INDICATOR_CONFIG 声明（推荐）** — 策略显式声明每个指标的中性值，归因更准确
2. **自动发现** — 不声明时，自动发现指标数组（排除 `_atr`, `_avg_volume` 等非信号数组），用数组中位数作为中性值

### 2. 行情归因（Regime Attribution）

将每笔交易标注当时的行情状态，计算各 regime 下的胜率和 PnL。

**三个维度：**

| 维度 | 指标 | 分档 |
|------|------|------|
| 趋势强度 | ADX(14) | Strong (>25) / Weak (15-25) / None (<15) |
| 波动率 | ATR 百分位 | High (>75th) / Normal / Low (<25th) |
| 量能活跃度 | Volume / SMA(20) | Active (>1.5x) / Normal / Quiet (<0.7x) |

---

## INDICATOR_CONFIG 声明（推荐）

```python
class StrongTrendV12(TimeSeriesStrategy):
    INDICATOR_CONFIG = [
        {'name': 'Aroon Oscillator', 'array_attr': '_aroon_osc', 'neutral_value': 100, 'role': 'trend'},
        {'name': 'PPO Histogram',    'array_attr': '_ppo_hist',   'neutral_value': 1.0, 'role': 'momentum'},
        {'name': 'Volume Momentum',  'array_attr': '_vol_mom',    'neutral_value': 2.0, 'role': 'volume'},
    ]
```

不声明则自动发现（用数组中位数作为中性值）。声明后归因更准确。

---

## 运行方式

归因分析自动集成到 `validate_and_iterate.py`，在测试集验证完成后自动运行。

报告输出到 `research_log/attribution/{version}_{symbol}.md`。

也可单独运行（Standalone API）：

```python
from attribution.signal import run_signal_attribution
from attribution.regime import run_regime_attribution
from attribution.report import generate_attribution_report

signal = run_signal_attribution(strategy_cls, params, "AG", "2025-01-01", "2026-03-01")
regime = run_regime_attribution(strategy_cls, params, "AG", "2025-01-01", "2026-03-01")
generate_attribution_report(signal, regime, "research_log/attribution/v12_AG.md")
```

---

## v12 案例研究（AG 2025-01 ~ 2026-03）

来源：v12 (Aroon + PPO + Volume Momentum) 的归因报告。

### 信号归因结果

| 指标 | 去掉后 Sharpe | 贡献 | 占比 |
|------|:-:|:-:|:-:|
| Volume Momentum | 1.494 | +1.597 | **51.7%** |
| PPO Histogram | 3.004 | +0.086 | 2.8% |
| Aroon Oscillator | 3.087 | +0.003 | 0.1% |

**关键发现：**

- **Volume Momentum 是 v12 唯一真正的 alpha 来源**（51.7%）。去掉它 Sharpe 从 3.09 降到 1.49
- **Aroon 和 PPO 几乎无贡献** — 之前认为 "Aroon 是最强趋势指标" 的结论需要修正。Aroon 的价值不在于提供 alpha，而可能在于过滤噪音交易（但 ablation 显示这个过滤几乎无效）
- **PPO 贡献 2.8%** — 之前认为 "PPO 是最强动量指标" 同样需要修正。PPO 在 v12 中更像锦上添花而非核心
- **此前的 "量价/OI 指标是 alpha 核心来源" 结论得到量化验证** — 不只是"重要"，而是"几乎全部"

### 行情归因结果

| Trend Regime | 交易数 | 胜率 | Avg PnL |
|---|:---:|:---:|:---:|
| Strong (ADX>25) | 3 | **100%** | +173% |
| Weak (15-25) | 3 | **0%** | -6.8% |
| None (<15) | 3 | 100% | +5.2% |

| Volume Regime | 交易数 | 胜率 | Avg PnL |
|---|:---:|:---:|:---:|
| Active (>1.5x) | 3 | **100%** | +173% |
| Normal | 5 | 60% | -0.7% |
| Quiet (<0.7x) | 1 | 0% | -1.3% |

**关键发现：**

- **v12 在强趋势 + 高量能下赚所有钱** — 100% 胜率，平均 +173%
- **弱趋势下 100% 亏损** — Portfolio 必须有策略覆盖弱趋势 regime
- **行情依赖度极高** — 不是一个全天候策略，而是一个精确的强趋势捕手
- **ATR 波动率维度显示 Unknown** — 因为测试期只有 1 年（252 bar rolling percentile 需要更多历史），下次用更长的测试期或降低 rolling window

---

## 归因后的决策

### 策略简化

如果归因发现 redundant indicators：

```
归因结果：Aroon 贡献 0.1%，PPO 贡献 2.8%
→ 可以创建 v12-lite（只用 Volume Momentum + ATR 止损）
→ 参数从 5 个降到 2 个（vol_mom_period + atr_trail_mult）
→ 过拟合空间大幅缩小
```

但需要在更多品种/更长时间段上验证简化版，避免在 AG 单品种上的结论不泛化。

**启示：v12 可以极大简化** — 可能只需 Volume Momentum + ATR 止损。简化后参数从 5 个降到 2 个，大幅降低过拟合空间。

### Portfolio 影响

| 归因发现 | Portfolio 决策 |
|---------|---------------|
| 策略在所有 regime 盈利 | 高权重候选 |
| 仅在强趋势盈利 | 限制权重，搭配其他 regime 策略 |
| 依赖单一指标 | 降低权重（单点失效风险） |
| 多指标均有贡献 | 增加权重（信号多样性） |

### v12 对 Portfolio C 的启示

1. **v12 在 Portfolio C 中占 25% 权重 — 这可能过高**，因为它本质上是一个单维度策略（Volume Momentum + 强趋势才有效）
2. **需要确认其他 4 个策略 (v8, v11, v31, v34) 是否在弱趋势中盈利** — 如果它们也依赖强趋势，Portfolio 在弱趋势下会集体失效
3. **归因分析应该成为 Portfolio 构建的前置条件** — 不只看 Sharpe 和相关性，还要看策略在哪些 regime 下有效，确保 Portfolio 覆盖所有 regime

---

## 归因报告格式

保存到 `research_log/attribution/`：

```markdown
# vN <品种> 归因报告 (日期)

## 信号归因
- baseline Sharpe: X.XX
- dominant: <指标名> (XX.X%)
- redundant: <指标名> (X.X%)
- 结论：<可简化 / 多指标均有贡献>

## 行情归因
- best regime: <regime 描述> (XX% WR, +XX%)
- worst regime: <regime 描述> (XX% WR, -XX%)
- 结论：<全天候 / 特定行情策略>

## 建议
- <策略简化建议>
- <Portfolio 权重建议>
- <regime 覆盖建议>
```

---

## 归因 Checklist

1. 测试集验证通过（Step 4）
2. 运行信号归因 → 识别 dominant 和 redundant 指标
3. 运行行情归因 → 识别 best/worst regime
4. 生成报告保存到 `research_log/attribution/`
5. 如有 redundant 指标，考虑策略简化
6. 更新 Portfolio 权重建议
7. Commit: `[attribution] vN <品种> attribution report`
