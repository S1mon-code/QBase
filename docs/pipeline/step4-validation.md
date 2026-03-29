# Step 4: 测试集验证（Test Set Validation）

---

## 核心原则

**测试集是只读的。不能因为测试集结果反过来修改任何参数或逻辑。**

---

## 验证规则

1. **参数锁定** — 用训练集优化出的参数原封不动跑测试集，不管结果好坏都不回去改
2. **不淘汰策略** — 测试集 Sharpe 为负不淘汰（portfolio 层面负相关策略有对冲价值），唯一淘汰条件仍然是交易次数不达标
3. **如果看了测试集结果再调参，测试集就变成了第二个训练集，失去验证意义**

---

## 数据分割

### 强趋势策略

训练品种和测试品种**不重叠**（跨品种验证）：

| 品种 | 训练集 | 测试集 |
|------|--------|--------|
| AG | J, ZC, JM, I, NI, SA (2016-2024) | AG 2025-01 ~ 2026-03 |
| EC | 同上 | EC 2023-07 ~ 2024-09 |

### All-Time 策略

时间切割，训练和测试**不重叠**（跨时间验证）：

| 品种 | 训练集 | 测试集 |
|------|--------|--------|
| AG | AG 2015-01 ~ 2024-06 | AG 2024-07 ~ 2026-03 |
| I | I 2015-01 ~ 2024-06 | I 2024-07 ~ 2026-03 |

---

## 详细结果记录

每个策略的测试集验证结果必须完整记录到 `research_log/` 中：

### 收益指标

- 总收益率、年化收益率
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Profit Factor（总利润 / 总亏损）

### 风险指标

- 最大回撤（幅度 + 持续天数）
- CVaR 95%（最差 5% 情况的平均损失）
- 最大单日亏损
- 年化波动率

### 交易统计

- 总交易次数
- 胜率
- 平均盈利 / 平均亏损（盈亏比）
- 最大单笔盈利 / 最大单笔亏损
- 平均持仓时间（bar 数）
- 最长持仓 / 最短持仓
- 做多次数 / 做空次数（all_time 策略）

### 与训练集对比

- 测试集 Sharpe / 训练集 Sharpe 比值
- 训练集 vs 测试集的 MaxDD 对比
- 训练集 vs 测试集的交易次数对比
- 训练集 vs 测试集的平均持仓时间对比

### 行为一致性检查

- 平均持仓时间是否与训练集一致（差异 > 3 倍需标注）
- 交易频率是否与训练集一致（差异 > 3 倍需标注）
- 做多/做空比例是否与训练集一致（all_time 策略）
- 如有异常行为，在 research_log 中标注原因分析

---

## 推荐验证方法：Walk-Forward Validation

Walk-Forward 是最严格的样本外验证方法。它模拟真实交易中"用历史优化参数→未来检验"的过程。

### 三段式数据分割（新策略必须遵守）

```
全部数据
├── 训练集 (60%)    → 策略参数优化（Optuna）
├── 验证集 (20%)    → 策略选择 + 权重优化
└── 测试集 (20%)    → 最终评估（只读，不改任何东西）
```

### Walk-Forward 滚动验证

对关键策略做滚动窗口验证（推荐 5年训练 → 1年测试）：

```bash
python strategies/walk_forward.py \
    --strategy strategies/strong_trend/v12.py \
    --symbol AG \
    --train-years 5 --test-years 1 \
    --start 2015 --end 2026 \
    --freq daily

# 快速测试模式 (10 trials/window)
python strategies/walk_forward.py --strategy v12.py --symbol AG --quick
```

每个窗口独立优化参数（30 trials coarse-only），然后在下一年测试。关注：

- **Mean Sharpe** — 所有窗口的平均测试 Sharpe
- **Win Rate** — 正 Sharpe 窗口占比（要求 ≥ 50%）
- **Worst Window** — 最差窗口的 Sharpe（识别 regime 盲区）

结果保存到 `research_log/walk_forward/{strategy}_{symbol}.json`。

---

## Industrial 模式验证（必须）

**任何策略进入 Portfolio 前，必须通过 Industrial 模式验证。** 这是 Basic 模式验证之外的额外步骤。

### 验证流程

1. 用 Step 3 锁定的参数，在 Industrial 模式下重跑测试集回测
2. 记录 Industrial Sharpe、与 Basic Sharpe 的衰减比
3. 根据衰减程度判定策略可靠性

### Industrial 验证配置

```python
config = BacktestConfig(
    initial_capital=10_000_000,
    volume_adaptive_spread=True,
    dynamic_margin=True,
    time_varying_spread=True,
    rollover_window_bars=20,
    asymmetric_impact=True,
    detect_locked_limit=True,
    margin_check_mode="daily",
    margin_call_grace_bars=3,
)
```

### 判定矩阵

| Industrial 衰减 | 判定 | 动作 |
|:---------------:|------|------|
| < 10% | 正常 | 通过，正常入 Portfolio |
| 10-30% | 可接受 | 通过，记录衰减比 |
| 30-50% | 警告 | 需要在 Industrial 模式下重新优化 |
| **> 50%** | **不可靠** | **策略不入 Portfolio — alpha 来自不真实成交假设** |

### 实测参考数据

| 策略 | 频率 | Basic Sharpe | Industrial Sharpe | 衰减 | 成交量变化 |
|------|------|:-----------:|:-----------------:|:----:|:----------:|
| v12 | daily | 3.09 | 2.84 | -8.1% | 18→18 |
| v9 | 1h | 2.15 | 1.00 | -53.7% | 261→43 |

### 验证记录模板补充

在标准验证记录模板中增加以下字段：

```markdown
### Industrial 模式验证
- **Industrial Sharpe**: X.XX
- **Basic Sharpe**: X.XX
- **衰减**: X.X%
- **Industrial 交易次数**: N（Basic: M）
- **判定**: 正常 / 可接受 / 警告 / 不可靠
```

---

## 结果解读（仅标注，不淘汰）

| 训练集 vs 测试集 | 解读 | 标注 |
|------|------|------|
| 测试集 Sharpe >= 训练集 50% | 正常，泛化能力可以 | 无需特别标注 |
| 测试集 Sharpe 远低于训练集 | 可能过拟合 | 标注"疑似过拟合" |
| 测试集 Sharpe 反而更好 | 可能是运气或市场环境恰好匹配 | 标注"测试集偏高，需观察" |
| 测试集交易行为异常 | 策略逻辑在新数据上不稳定 | 标注"行为异常" + 具体描述 |

---

## 记录格式模板

```markdown
## YYYY-MM-DD <品种> <策略名> 测试集验证

### 策略信息
- **版本**: vN
- **指标**: 指标1(参数) + 指标2(参数) + 指标3(参数)
- **频率**: daily / 4h / 1h / ...
- **优化参数**: param1=X, param2=Y, param3=Z

### 训练集结果（参考）
- **期间**: YYYY-MM-DD ~ YYYY-MM-DD
- **Sharpe**: X.XX | **MaxDD**: X.X% | **Trades**: N
- **年化收益**: X.X% | **胜率**: X.X%

### 测试集结果
- **期间**: YYYY-MM-DD ~ YYYY-MM-DD
- **收益**: 总收益=X.X%, 年化=X.X%
- **风险调整**: Sharpe=X.XX, Sortino=X.XX, Calmar=X.XX
- **风险**: MaxDD=X.X% (持续N天), CVaR95=X.X%, 最大单日亏损=X.X%
- **交易**: N笔, 胜率=X.X%, 盈亏比=X.XX, 平均持仓=N bars
- **Omega**: X.XX | **Profit Factor**: X.XX

### 训练集/测试集对比
- **Sharpe 比值**: 测试/训练 = X.XX
- **行为一致性**: 正常 / 异常（描述）

### 标注
- 无 / 疑似过拟合 / 测试集偏高 / 行为异常

### 结论
一句话总结
```
