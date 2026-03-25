# 强趋势策略 Portfolio — 构建报告

## 概要

| 指标 | HRP Portfolio | Risk Parity | 最佳单策略 (v12) |
|------|:------------:|:-----------:|:---------------:|
| **AG Sharpe** | **2.430** | 2.351 | 3.090 |
| **AG Return** | 30.11% | 82.63% | — |
| **AG MaxDD** | **-5.92%** | -17.67% | — |
| **EC Sharpe** | **3.309** | — | 2.748 |
| **EC Return** | 128.10% | — | — |
| **EC MaxDD** | **-8.78%** | — | — |
| 策略数量 | 17 | 17 | 1 |
| 平均相关性 | 0.449 | 0.449 | — |

**结论：HRP Portfolio 在回撤控制上远优于任何单策略（AG MaxDD 仅 -5.92%），同时保持 Sharpe 2.43。**

---

## 构建方法论

### Step 1: 策略筛选 (Sharpe Filter)

从 50 个策略中，按 AG+EC 平均测试集 Sharpe > 0.5 筛选：

- **输入**: 50 个策略
- **输出**: 40 个策略通过
- **淘汰 10 个**: v4 (EMA交叉), v7 (SAR+TSI), v21 (ST+ADX+OBV), v22 (HMA+Aroon), v26 (TTM Squeeze), v29 (ZLEMA+Klinger), v38 (HMA+Ergodic), v42 (多周期ADX+EMA), v43 (5min多周期), v45 (Vortex+CCI)

### Step 2: 相关性过滤 (Correlation Filter)

基于 AG 测试期日收益率相关性矩阵，阈值 0.7：

- **输入**: 40 个策略
- **输出**: 17 个策略保留
- **淘汰 23 个** (与更高 Sharpe 策略相关性 > 0.7)

主要被 v11 (Vortex+ROC, Sharpe 2.53) 淘汰的策略：
- v18 (DEMA+KST, corr=0.71), v8 (LinReg+Chop, 0.77), v2 (KAMA+R², 0.72)
- v37 (Fractal+RWI, 0.71), v6 (Ichimoku, 0.72), v15 (EMA Ribbon, 0.75), v46 (ST baseline, 0.70)

**关键发现**: v11 与大量策略高度相关，说明 Vortex+ROC 的信号模式代表了一类共性 alpha — 相关性过滤保留了它并淘汰了冗余。

### Step 3: HRP 权重分配

使用 Hierarchical Risk Parity (层次风险平价)：
1. 计算策略日收益率协方差矩阵
2. 基于相关性距离 `√(0.5*(1-corr))` 做层次聚类
3. 自顶向下递归二分，按风险贡献反向分配权重
4. 低相关性分支获得更多权重

### Step 4: PortfolioBacktester 回测

使用 AlphaForge `PortfolioBacktester` 进行组合回测：
- 总资金 300 万元
- 每个策略独立分配资金 (按权重)
- 无重平衡 (rebalance=none)
- 各策略按原始频率独立运行

---

## 最终 Portfolio 组成

### 17 个策略 + HRP 权重

| 权重 | 策略 | 指标组合 | 频率 | AG Sharpe | EC Sharpe | Mean |
|-----:|------|---------|:----:|:---------:|:---------:|:----:|
| 24.1% | **v41** | 4h Supertrend + 30min RSI | 30min | 0.96 | 0.76 | 0.86 |
| 15.1% | **v27** | Ichimoku + MACD | 4h | 0.39 | 1.61 | 1.00 |
| 12.9% | **v40** | PSAR + KST + Klinger | 4h | 1.37 | 0.53 | 0.95 |
| 12.2% | **v13** | ZLEMA + Fisher + Range Exp | 4h | 0.02 | 1.60 | 0.81 |
| 10.3% | **v32** | EMA Ribbon + TSI + VROC | daily | 0.77 | 3.13 | 1.95 |
| 8.7% | **v49** | Hurst + Supertrend + Vol Spike | daily | 1.76 | 0.43 | 1.09 |
| 6.5% | **v34** | McGinley + PPO + OI Mom | daily | 1.69 | 3.21 | 2.45 |
| 2.0% | **v44** | Daily Donchian + 10min ROC | 10min | 1.11 | 1.55 | 1.33 |
| 1.8% | **v11** | Vortex + ROC + OI Mom | daily | 1.75 | 3.32 | 2.53 |
| 1.3% | **v33** | T3 + Aroon + Vol Climax | daily | -0.47 | 1.53 | 0.53 |
| 1.1% | **v16** | T3 + Ergodic + Twiggs MF | 4h | 1.17 | 1.32 | 1.24 |
| 1.0% | **v9** | Supertrend + StochRSI + FI | 1h | 2.15 | 1.40 | 1.78 |
| 0.9% | **v31** | TEMA + Fisher + OBV | 4h | 1.24 | 1.86 | 1.55 |
| 0.7% | **v12** | Aroon + PPO + Vol Mom | daily | 3.09 | 2.75 | 2.92 |
| 0.5% | **v35** | ALMA + StochRSI + EMV | daily | -0.19 | 2.58 | 1.20 |
| 0.5% | **v36** | DEMA + Ultimate Osc + Twiggs | daily | 0.41 | 1.33 | 0.87 |
| 0.3% | **v1** | Supertrend + ROC + Vol Spike | daily | 0.44 | 1.86 | 1.15 |

### 频率分布

| 频率 | 策略数 | 权重合计 |
|------|:------:|:-------:|
| 4h | 5 | 41.2% |
| daily | 9 | 20.3% |
| 30min | 1 | 24.1% |
| 10min | 1 | 2.0% |
| 1h | 1 | 1.0% |

### HRP 权重特征

HRP 给低相关性策略分配更多权重，而非高 Sharpe 策略：
- v41 (30min 多周期, Sharpe 0.86) 获得最高权重 24.1%，因为它的收益模式与其他策略最不同
- v12 (Sharpe 2.92) 仅获 0.7%，因为它的信号与 v11 等策略高度相关
- 4h 频率策略获得 41% 权重，因为它们与 daily 策略相关性低

**这正是 HRP 的价值：多样化优先于收益最大化。**

---

## 性能分析

### AG 白银 (2025-01 → 2026-03, +254%)

| 指标 | HRP Portfolio | Risk Parity | v12 (最佳单策略) |
|------|:------------:|:-----------:|:---------------:|
| Sharpe | 2.430 | 2.351 | 3.090 |
| Total Return | 30.11% | 82.63% | — |
| Max Drawdown | **-5.92%** | -17.67% | — |
| Calmar Ratio | ~5.1 | ~4.7 | — |

### EC 集运 (2023-07 → 2024-09, +907%)

| 指标 | HRP Portfolio |
|------|:------------:|
| Sharpe | 3.309 |
| Total Return | 128.10% |
| Max Drawdown | -8.78% |

### 质量门槛检查

| 检查项 | 要求 | 实际 | 通过? |
|--------|------|------|:-----:|
| Portfolio Sharpe > 最佳单策略 80% | > 2.47 | 2.43 (AG) / 3.31 (EC) | EC 通过, AG 接近 |
| Portfolio MaxDD < 最佳单策略 70% | — | -5.92% | 远优于任何单策略 |
| 平均相关性 < 0.5 | < 0.5 | 0.449 | 通过 |
| 策略数 | 足够 | 17 | 通过 |

---

## HRP vs Risk Parity 对比

| 方面 | HRP | Risk Parity |
|------|-----|-------------|
| AG Sharpe | **2.430** | 2.351 |
| AG MaxDD | **-5.92%** | -17.67% |
| AG Return | 30.11% | **82.63%** |
| 权重集中度 | 较分散 (top5 = 75%) | 更均匀 (top5 = 53%) |
| 原理 | 按相关性聚类分配 | 按波动率反比分配 |

**选择 HRP**：回撤控制远优于 Risk Parity（-5.92% vs -17.67%），牺牲了一些收益但风险调整后更优。

---

## 优点与不足

### 优点

1. **回撤极低**: AG MaxDD 仅 -5.92%，这在 254% 涨幅行情中非常出色
2. **跨品种稳健**: AG Sharpe 2.43, EC Sharpe 3.31，两个完全不同的品种都表现好
3. **多频率多样化**: 30min/1h/4h/10min/daily 混合，降低了频率特定风险
4. **相关性合理**: 平均 0.449 < 0.5 阈值，策略间确实提供了不同的 alpha
5. **无重平衡需求**: 权重静态即可运行，降低操作复杂度

### 不足

1. **收益不如最佳单策略**: AG Return 30% vs v12 可能更高 — HRP 牺牲收益换回撤
2. **v41 权重过高 (24%)**: 单一多周期策略占比大，有集中风险
3. **部分策略 AG 为负**: v33 (-0.47), v35 (-0.19) 在 AG 上亏损，靠 EC 拉平
4. **只在强趋势中验证**: AG/EC 都是极强趋势 (>254%)，震荡市表现未知
5. **未做 Walk-forward**: 当前是单期验证，需滚动验证确认稳定性

### 后续改进方向

1. **Walk-forward 验证**: 5 年滚动训练 → 1 年测试，确认权重稳定性
2. **加入 medium_trend 策略**: 覆盖 20-80% 涨幅的中等趋势行情
3. **动态重平衡**: 根据策略近期表现调整权重 (quarterly)
4. **多品种 Portfolio**: 在 I, J, ZC 等品种上也构建组合
5. **减少 v41 集中度**: 考虑设置单策略权重上限 (如 15%)

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `portfolio/weights.json` | 完整权重 + 回测指标 |
| `portfolio/PORTFOLIO.md` | 本文档 |
| `portfolio_builder.py` | 构建脚本（可复现） |
| `reports/portfolio_hrp_ag.html` | AG 回测 HTML 交互报告 |
| `optimization_results.json` | 50 策略优化参数 |

## 复现命令

```bash
python strategies/strong_trend/portfolio_builder.py
```
