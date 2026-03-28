# 风险管理框架

## 单笔风险

每笔交易最大风险 = 账户权益的 **2%**。这是仓位计算的基础。

### 仓位公式

```python
# 标准仓位公式
lots = (equity * 0.02) / (stop_distance * contract_multiplier)

# 示例：AG 白银，权益 100万，ATR=50，止损 3.5×ATR
# lots = 1,000,000 * 0.02 / (50 * 3.5 * 15) = 7.6 → 7 手
```

### 保证金上限

仓位还需受保证金上限约束：

```python
margin_per_lot = price * spec.multiplier * spec.margin_rate
max_lots_by_margin = int(equity * 0.30 / margin_per_lot)  # 30% 上限
lots = min(risk_lots, max_lots_by_margin)
```

完整仓位计算：

```python
risk_per_trade = equity * 0.02          # 每笔交易风险 <= 2% 权益
risk_per_lot = ATR * trail_mult * contract_multiplier
lots = int(risk_per_trade / risk_per_lot)
lots = max(1, min(lots, max_lots_by_margin))
```

---

## 总敞口限制

| 限制 | 数值 | 说明 |
|------|------|------|
| 单策略最大持仓 | 权益的 **30%** 保证金占用 | 含所有加仓层 |
| 单品种最大持仓 | 权益的 **40%** 保证金占用 | 同品种多策略合计 |
| 总账户最大持仓 | 权益的 **80%** 保证金占用 | 留 20% 现金缓冲 |
| 单笔最大亏损 | 权益的 **2%** | 通过仓位公式控制 |
| 单日最大亏损 | 权益的 **5%** | 触发后当日停止交易 |

---

## 合约参数 API

**动态获取，不要硬编码。** 从 AlphaForge 的 `ContractSpecManager` 获取品种参数：

```python
from alphaforge.data.contract_specs import ContractSpecManager

specs = ContractSpecManager()
spec = specs.get("AG")
spec.multiplier      # 15
spec.tick_size        # 1.0
spec.margin_rate      # 0.08
spec.commission_open  # 0.00005 (ratio)

# 仓位计算中使用
margin_per_lot = price * spec.multiplier * spec.margin_rate
max_lots_by_margin = int(equity * 0.30 / margin_per_lot)  # 30% 上限
lots = min(risk_lots, max_lots_by_margin)
```

---

## 止损设计

### ATR 跟踪止损

按策略类型选择 ATR 倍数：

| 策略类型 | 推荐止损 | ATR multiplier |
|---------|---------|:-:|
| 强趋势 | ATR trailing stop | **> 4.0** |
| 中等趋势 | ATR trailing stop | 3.0 - 4.0 |
| 均值回归 | 固定 ATR 止损 | 2.0 - 3.0 |

### 关键经验（50 策略实验总结）

- **宽止损是强趋势中最重要的单一因素**
- Top 10 策略中 9 个 `atr_trail_mult` > 4.0
- 强趋势中 15-25% 回撤是正常的
- v46 仅 Supertrend + 宽止损排名 16/50

### 换仓日处理

```python
if context.is_rollover:
    return  # 换仓日不开新仓，避免异常价格
```

---

## Portfolio 止损（4 级）

| 级别 | 触发条件 | 动作 |
|------|---------|------|
| **预警** | 组合回撤达 **-10%** | 记录日志，检查各策略状态，不自动操作 |
| **减仓** | 组合回撤达 **-15%** | 所有策略仓位减半，暂停加仓 |
| **熔断** | 组合回撤达 **-20%** | 全部平仓，停止交易，人工审查后重启 |
| **单日熔断** | 单日亏损达权益 **-5%** | 当日全部平仓，次日恢复 |

### 恢复规则

- **减仓后**：回撤收窄至 -10% 以内，恢复正常仓位
- **全面熔断后**：必须人工审查并确认市场环境正常后才能重启
- **连续 2 次触发 -20% 熔断**：停止该 portfolio，重新优化权重

---

## 策略退化与退役标准

策略上线后需要持续监控，触发以下条件时考虑退役：

| 条件 | 动作 |
|------|------|
| 滚动 6 个月 Sharpe < 0 | 标记为"观察"，降低权重至原来的 50% |
| 连续 3 个月亏损 | 标记为"观察"，降低权重至原来的 50% |
| 滚动 12 个月 Sharpe < -0.5 | 从 portfolio 中移除 |
| 最大回撤超过回测最大回撤的 1.5 倍 | 立即移除，调查原因 |
| 市场结构性变化（如品种规则改变、交易时间变更） | 重新回测验证，不通过则移除 |

### 退役流程

1. 从 portfolio 权重中移除
2. 在 `research_log/` 中记录退役原因和日期
3. 策略文件保留（不删除），但在文件头标注 `# RETIRED: YYYY-MM-DD 原因`
4. 重新运行 portfolio 构建，重新分配权重

---

## 权重限制

| 参数 | 默认值 | 说明 |
|------|:-----:|------|
| 单策略最大权重 | **20%** | `--max-weight 0.20` |
| 单策略最小权重 | 隐含 ~5% | HRP 自然分配 |
| 策略数量 | 3-8 | 穷举选择范围 |

**为什么单策略最大 20%？** 避免 portfolio 过度依赖单一策略。即使该策略完全失效，对组合的影响不超过 20%。配合 HRP 权重分配，自然形成 5%-20% 的分布。

### 同向持仓限制

- 同一品种最多同时运行 **3 个策略** 同方向持仓
- 避免 "所有策略同时看多 AG" 导致过度集中

### 品种敞口限制

- 单品种名义敞口 ≤ 总资金的 30%
- 高相关品种组（如 J+JM+I 黑色系）合计敞口 ≤ 50%

### 频率分散

- 组合中至少包含 2 种频率的策略（如 daily + 4h）
- 评分系统中 "频率多样性" 占实操性维度的一部分

---

## 风险度量指标

| 指标 | 计算方式 | 健康范围 |
|------|---------|---------|
| Sharpe | 年化收益 / 年化波动 | > 1.0 |
| Calmar | 年化收益 / MaxDD | > 2.0 |
| CVaR 95% | 最差 5% 日收益的均值 | > -2% |
| MaxDD Duration | 最大回撤恢复天数 | < 30 天 |
| Bootstrap CI | 1000 次重采样 Sharpe 分布 | 95% 下界 > 0 |
| Profit Concentration | Top 10% 天 PnL / 总利润 | < 0.5 |
| Simulated CVaR 95% | Monte Carlo 压力测试最差 5% 均损 | > -3% (日) |
| Simulated MaxDD | Monte Carlo 回撤分布中位数 | < 回测 MaxDD × 1.5 |
| Slippage Sensitivity | 2x 滑点下 Sharpe 衰减 | < 30% (LOW/MODERATE) |

**利润集中度**是最容易被忽略的风险指标。如果 80% 利润来自 10% 的交易日，去掉那几天就没有 alpha。策略看起来 Sharpe 很高但实际非常脆弱。

---

## 鲁棒性测试工具

### 滑点敏感性测试

评估策略在不同滑点水平（1x/2x/3x 基准滑点）下的 Sharpe 衰减，判定为 LOW/MODERATE/HIGH：

```bash
python tests/robustness/slippage_test.py --strategy <strategy> --symbol <symbol>
```

- **LOW** — Sharpe 下降 < 15%，策略对滑点不敏感
- **MODERATE** — Sharpe 下降 15-30%，需要收紧权重上限至 10%
- **HIGH** — Sharpe 下降 > 30%，策略不适合实盘或需大幅降权

**与仓位管理的关系：** HIGH 敏感性的策略在低流动性品种上尤其危险，应结合流动性分级（见下方）综合评估。

### Monte Carlo 压力测试

1000 次 Bootstrap 重采样，输出模拟 CVaR、模拟最大回撤分布，判定为 ROBUST/ACCEPTABLE/FRAGILE：

```bash
python tests/robustness/stress_test.py --strategy <strategy> --symbol <symbol>
```

**关键输出指标：**

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| 模拟 CVaR 95% | 压力场景下最差 5% 平均损失 | > -3% (日) |
| 模拟 MaxDD 中位数 | Bootstrap 回撤分布中位数 | < 回测 MaxDD 的 1.5x |
| 模拟 MaxDD 95 分位 | 极端场景回撤 | < -20% |

### Portfolio 选择稳定性测试

通过数据扰动评估策略在 Portfolio 中的稳定性，分类为 CORE/SATELLITE/EDGE：

```bash
python portfolio/stability_test.py --portfolio <weights_file>
```

- **CORE** — >80% 扰动中被选入，是组合的可靠成员
- **SATELLITE** — 50-80% 扰动中被选入，权重应保守设置
- **EDGE** — <50% 扰动中被选入，考虑移除

**作为风险管理工具：** EDGE 策略过多的 Portfolio 在市场环境变化时容易解体。建议 CORE 策略权重合计 > 50%。

---

## 回撤归因工具

Portfolio 回撤发生时，使用回撤归因工具快速定位原因：

```bash
python -m attribution.drawdown --portfolio <weights_file>
```

**输出：**
- 按回撤事件分解，显示哪些策略在回撤中亏损最多
- 回撤主要发生在哪种 regime 下（强趋势/弱趋势/高波动等）
- 帮助决定是否需要调整权重、添加对冲策略或触发退役流程

结合 `attribution.decay`（alpha 衰减检测）可判断回撤是暂时的还是策略失效的信号。

---

## AlphaForge V6.0 风险管理增强

V6.0 新增多项回测级风险管理特性，使回测更贴近真实交易环境：

### 保证金管理增强

```python
config = BacktestConfig(
    margin_check_mode="daily",       # "bar": 每bar检查 | "daily": 每日结算时检查
    margin_call_grace_bars=5,        # 保证金追缴宽限期（bar数），0=立即强平
)
```

- **`margin_check_mode="daily"`** — 更贴近真实交易所结算机制。日内保证金不足不会立即强平，而是在每日结算时检查
- **`margin_call_grace_bars`** — 宽限期内允许策略自行减仓，避免因短暂波动被强平

### 涨跌停锁仓检测

```python
config = BacktestConfig(
    detect_locked_limit=True,        # 涨跌停锁仓检测
)
```

启用后，当持仓品种触及涨跌停且无法平仓时，引擎会标记为锁仓状态。策略可通过此信息调整风险管理行为。

### 隔夜跳空风险

```python
config = BacktestConfig(
    overnight_gap=True,              # 隔夜跳空风险模拟
)
```

模拟夜盘收盘到日盘开盘间的跳空风险，影响止损触发和盈亏计算。

### Iron Rules 参考

AlphaForge V6.0 定义了 13 条不可违反的回测铁律（Signal Delay、Tick Snap、FIFO、Margin Check、Locked Limit、Partial Fill、Single Direction、Night Session、Forced Liquidation、Nonlinear Impact、Bid-Ask Spread、Rollover Cost、Settlement Mark-to-Market）。详见 [AlphaForge API — Iron Rules](alphaforge-api.md#iron-rulesv60--13-条铁律)。

---

## 注意事项速查

- 所有策略必须使用预计算模式（on_init_arrays + context.bar_index 查表）
- 所有价格计算用复权价（context.get_full_close_array），下单用原始价（context.close_raw）
- 信号在下一个 bar 的 open 执行，不是当前 bar
- 同品种不可同时持有多空仓位
- 保证金不足会被拒绝开仓，权益低于维持保证金会被强平（V6 支持 `margin_call_grace_bars` 宽限）
- FIFO 平仓：先平昨仓（便宜），再平今仓（贵）
- 单笔成交量不超过该 bar 成交量的 10%
- V6: `margin_check_mode="daily"` 更贴近真实结算
- V6: `detect_locked_limit=True` 检测涨跌停锁仓
