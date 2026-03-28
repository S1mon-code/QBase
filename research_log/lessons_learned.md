# QBase 实战经验与教训

持续更新。每次从优化/回测中发现的规律性结论记录在此。

---

## 强趋势策略 — 50 策略完整总结 (2026-03-25)

来源：50 个策略 × 150 Optuna trials × 6 训练品种 (J, ZC, JM, I, NI, SA)，AG/EC 测试集验证。

### 止损
- **宽止损是最重要的单一因素**：Top 10 策略中 9 个 `atr_trail_mult` > 4.0
- v20 (Sharpe 1.88) 和 v18 (Sharpe 1.85) 是例外，`atr_trail_mult` ~2.1-2.2，但它们有其他机制弥补（Fractal level / NR7 squeeze 精确入场）
- 强趋势中 15-25% 的回撤是正常的，窄止损 (<3×ATR) 在 6 品种训练中一致被淘汰
- **v46 (仅 Supertrend + 宽止损) 排名 16/50**，说明好的止损 > 花哨的信号

### 指标选择
- **量价/OI 指标是 alpha 核心来源**：Top 10 中 8/10 使用量/OI 指标
  - Volume Momentum, VROC, OI Momentum, Volume Spike, Force Index
  - 中国期货散户占比高 → 量能信号比西方市场有价值
- **PPO (Percentage Price Oscillator) 是最强动量指标**：v12 (Sharpe 2.92) 和 v34 (2.45) 都用 PPO
- **Aroon 是最强趋势指标**：v12 (2.92) 用 Aroon，远超传统 ADX
- **非传统组合大幅胜出经典组合**：
  - v12 (Aroon+PPO) 2.92 vs v3 (海龟 Donchian+ADX) 0.55 → **5.3x 差距**
  - v34 (McGinley+PPO) 2.45 vs v4 (EMA+MACD) 0.05 → **49x 差距**
- **EMA 交叉在强趋势中基本无效**：v4 排名 49/50，交叉信号太慢且反复损耗
- **TTM Squeeze 在强趋势中无效**：v26 Sharpe=0，挤压信号适合震荡不适合强趋势

### 多周期策略
- **效果不及预期**：v41-v45 平均 Mean Test 0.62，远低于单周期平均 1.15
- **小周期噪音是主因**：v43 (5min) 唯一训练集负 Sharpe
- **日线级聚合最有效**：v44 (10min→daily) 排名最高 (1.33)
- **结论**：除非有明确理由，优先用单周期策略。多周期增加复杂度但未增加 alpha

### 极简策略
- **1-2 个指标够用**：v46 (仅 Supertrend) Sharpe 1.32, v47 (ADX+Chandelier) 1.09
- **堆砌指标 ≠ 更好**：很多 3 指标策略排名低于 1 指标策略
- **核心原则验证**：1 个强指标 > 3 个弱指标

### 行情状态识别
- **Hurst > 0.55 作为趋势过滤器有效**：v49 AG Sharpe 1.76
- **Yang-Zhang 波动率扩张 = 趋势启动信号**：v50 EC Sharpe 2.33
- **两者在单一品种上有效但泛化性弱**：需要更多测试品种验证

### 频率
- daily 占据 Top 10 中 9 席，唯一例外是 v9 (1h, AG 2.15)
- 4h 策略稳定但不突出 (v16 1.24, v31 1.55, v13 0.81)
- 频率选择重要性远低于指标选择和止损设置
- **结论**：强趋势策略默认用 daily，只有特定需求才用更高频

### 优化
- 多品种平均 Sharpe 作为目标函数有效防止过拟合
- 150 trials 足够，v4 用 200 trials 也没救回来
- 训练集 Sharpe 0.5-1.0 对应测试集 1.0-2.9，测试集 > 训练集是因为 AG/EC 趋势特别强
- **训练集 Sharpe > 0.8 ≠ 测试集更好**：v22 训练 0.87 但测试 0.41；v20 训练 0.37 但测试 1.88
- **一致性比平均值更重要**：v12 在 AG (3.09) 和 EC (2.75) 都很强，这比只在一个品种上极高更可靠

### 预计算架构
- on_init_arrays 迁移成功，50 个策略全部通过
- 优化速度显著提升：daily ~30s/策略，1h ~60s/策略，5min ~5min/策略
- 50 策略全量优化耗时 ~30 分钟（含 3 轮迭代验证）
- Optuna 兼容：参数变化时 on_init_arrays 自动重算，每 trial 只算 1 次指标

---

## 归因分析 — 首批发现 (2026-03-28)

来源：v12 (Aroon + PPO + Volume Momentum) 在 AG 2025-01 ~ 2026-03 的归因报告。

### 信号归因（Ablation Test）

| 指标 | 去掉后 Sharpe | 贡献 | 占比 |
|------|:-:|:-:|:-:|
| Volume Momentum | 1.494 | +1.597 | **51.7%** |
| PPO Histogram | 3.004 | +0.086 | 2.8% |
| Aroon Oscillator | 3.087 | +0.003 | 0.1% |

**关键发现：**
- **Volume Momentum 是 v12 唯一真正的 alpha 来源**（51.7%）。去掉它 Sharpe 从 3.09 降到 1.49
- **Aroon 和 PPO 几乎无贡献** — 之前认为 "Aroon 是最强趋势指标" 的结论需要修正。Aroon 的价值不在于提供 alpha，而可能在于过滤噪音交易（但 ablation 显示这个过滤几乎无效）
- **PPO 贡献 2.8%** — 之前认为 "PPO 是最强动量指标" 同样需要修正。PPO 在 v12 中更像锦上添花而非核心
- **此前的 "量价/OI 指标是 alpha 核心来源" 结论得到量化验证** — 不只是"重要"，而是 "几乎全部"
- **启示：v12 可以极大简化** — 可能只需 Volume Momentum + ATR 止损

### 行情归因（Regime Attribution）

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

### 对 Portfolio 的启示

1. **v12 在 Portfolio C 中占 25% 权重 — 这可能过高**，因为它本质上是一个单维度策略（Volume Momentum + 强趋势才有效）
2. **需要确认其他 4 个策略 (v8, v11, v31, v34) 是否在弱趋势中盈利** — 如果它们也依赖强趋势，Portfolio 在弱趋势下会集体失效
3. **归因分析应该成为 Portfolio 构建的前置条件** — 不只看 Sharpe 和相关性，还要看策略在哪些 regime 下有效，确保 Portfolio 覆盖所有 regime

### 对策略简化的启示

- 如果 Aroon 和 PPO 真的无贡献，可以创建一个 v12-lite（只用 Volume Momentum + ATR 止损），对比性能
- 简化后参数从 5 个降到 2 个（vol_mom_period + atr_trail_mult），大幅降低过拟合空间
- 但需要在更多品种 / 更长时间段上验证，避免在 AG 单品种上的结论不泛化

---

## Portfolio 构建 Checklist (待执行)

基于 50 策略结果，下一步：
1. 按测试集 Sharpe 取 Top 25 (Sharpe > 0.87)
2. 计算策略间日收益率相关性矩阵
3. 相关性 > 0.7 的保留 Sharpe 更高的
4. HRP 赋权 → Walk-forward 验证
5. Portfolio Sharpe 必须 > 最佳单策略 (v12) 的 80% = 2.34
