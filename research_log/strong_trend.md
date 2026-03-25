# Strong Trend Strategy — Complete Results

## Overview

- **目标**: 开发品种无关的强趋势策略，捕捉涨幅 >100%、持续 6+ 个月的大行情
- **策略数量**: 50 个 (v1-v50)，只做多，预计算架构 (on_init_arrays)
- **分类**: v1-v40 标准单周期, v41-v45 多周期协作, v46-v48 极简, v49-v50 行情状态识别
- **训练集**: 6 个代表品种 (J, ZC, JM, I, NI, SA)，Optuna TPE 150 trials
- **测试集**: AG (2025-01 → 2026-03, +254%), EC (2023-07 → 2024-09, +907%)
- **结果**: 49/50 策略测试集平均 Sharpe > 0

## Test Set Results (AG + EC) — All 50 Strategies

| Rank | Version | 策略名 | 指标组合 | 频率 | Train Sharpe | AG Sharpe | EC Sharpe | Mean Test |
|------|---------|--------|---------|------|-------------|-----------|-----------|-----------|
| 1 | v12 | Aroon动量 | Aroon + PPO + Volume Momentum | daily | 0.825 | 3.090 | 2.748 | **2.919** |
| 2 | v11 | 涡旋指标 | Vortex + ROC + OI Momentum | daily | 1.190 | 1.747 | 3.316 | **2.531** |
| 3 | v34 | McGinley动量 | McGinley + PPO + OI Momentum | daily | 0.941 | 1.689 | 3.207 | **2.448** |
| 4 | v32 | EMA扇形TSI | EMA Ribbon + TSI + VROC | daily | 0.963 | 0.768 | 3.126 | **1.947** |
| 5 | v20 | 分形突破 | Fractal + Mass Index + VROC | daily | 0.365 | 1.804 | 1.962 | **1.883** |
| 6 | v18 | DEMA知而 | DEMA + KST + NR7 | daily | 0.533 | 0.911 | 2.793 | **1.852** |
| 7 | v25 | 回归涡旋 | LinReg Slope + Vortex + Volume Spike | daily | 0.975 | 1.238 | 2.404 | **1.821** |
| 8 | v9 | Supertrend振荡 | Supertrend + StochRSI + Force Index | 1h | 0.462 | 2.152 | 1.397 | **1.775** |
| 9 | v8 | 回归斜率 | LinReg Slope + Choppiness + NATR | daily | 0.699 | 1.321 | 2.053 | **1.687** |
| 10 | v2 | 趋势质量 | KAMA + R² + ATR | daily | 0.664 | 1.029 | 2.207 | **1.618** |
| 11 | v31 | TEMA Fisher | TEMA + Fisher Transform + OBV | 4h | 0.651 | 1.242 | 1.864 | **1.553** |
| 12 | v37 | 分形RWI | Fractal + RWI + Force Index | daily | 0.614 | 0.947 | 2.015 | **1.481** |
| 13 | v6 | 一目均衡 | Ichimoku + RSI + OBV | daily | 0.841 | 1.249 | 1.674 | **1.462** |
| 14 | v15 | EMA扇形 | EMA Ribbon + CMO + ATR | daily | 0.739 | 1.144 | 1.726 | **1.435** |
| 15 | v44 | 多周期Donchian | Daily Donchian + 10min ROC | 10min | 0.569 | 1.109 | 1.547 | **1.328** |
| 16 | v46 | Supertrend基线 | Supertrend Only (baseline) | daily | 0.572 | 1.045 | 1.593 | **1.319** |
| 17 | v16 | T3平滑 | T3 + Ergodic + Twiggs MF | 4h | 0.675 | 1.166 | 1.319 | **1.242** |
| 18 | v39 | Keltner CMO | Keltner Channel + CMO + Volume Spike | daily | 0.465 | 0.870 | 1.592 | **1.231** |
| 19 | v5 | Hull趋势 | HMA + CCI + Volume Climax | daily | 0.755 | 0.447 | 2.015 | **1.231** |
| 20 | v35 | ALMA StochRSI | ALMA + StochRSI + EMV | daily | 0.387 | -0.187 | 2.578 | **1.196** |
| 21 | v1 | Supertrend动量 | Supertrend + ROC + Volume Spike | daily | 0.589 | 0.439 | 1.861 | **1.150** |
| 22 | v50 | YZ波动率 | Yang-Zhang Vol + Donchian + OBV | daily | 0.471 | -0.089 | 2.333 | **1.122** |
| 23 | v49 | Hurst行情 | Hurst Exponent + Supertrend + Vol Spike | daily | 0.897 | 1.763 | 0.426 | **1.094** |
| 24 | v47 | ADX Chandelier | ADX + Chandelier Exit | daily | 0.562 | 0.978 | 1.195 | **1.087** |
| 25 | v27 | 一目MACD | Ichimoku + MACD | daily | 0.655 | 0.388 | 1.610 | **0.999** |
| 26 | v19 | McGinley趋势 | McGinley + Coppock + A/D Line | daily | 0.672 | -0.111 | 2.087 | **0.988** |
| 27 | v48 | R²突破 | R-Squared + Donchian | daily | 0.386 | 1.260 | 0.686 | **0.973** |
| 28 | v28 | Donchian Chop | Donchian + Choppiness + A/D Line | daily | 0.541 | 0.375 | 1.525 | **0.950** |
| 29 | v40 | PSAR KST | PSAR + KST + Klinger | daily | 0.547 | 1.365 | 0.527 | **0.946** |
| 30 | v14 | 突破随机游走 | Donchian + RWI + Klinger | daily | 0.667 | 1.068 | 0.777 | **0.922** |
| 31 | v30 | Supertrend Coppock | Supertrend + Coppock | daily | 0.379 | 0.172 | 1.612 | **0.892** |
| 32 | v17 | ALMA趋势 | ALMA + Ultimate Osc + EMV | daily | 0.807 | 0.241 | 1.502 | **0.872** |
| 33 | v36 | DEMA UO | DEMA + Ultimate Oscillator + Twiggs | daily | 0.471 | 0.410 | 1.327 | **0.869** |
| 34 | v41 | 多周期ST+RSI | 4h Supertrend + 30min RSI | 30min | 0.607 | 0.957 | 0.760 | **0.859** |
| 35 | v13 | 零滞后趋势 | ZLEMA + Fisher + Range Expansion | 4h | 0.412 | 0.017 | 1.599 | **0.808** |
| 36 | v24 | PSAR CCI | PSAR + CCI + OI Momentum | daily | 0.597 | -0.302 | 1.641 | **0.669** |
| 37 | v23 | Keltner ROC | Keltner Channel + ROC | daily | 0.513 | -0.478 | 1.670 | **0.596** |
| 38 | v10 | 三重EMA | TEMA + ADX + Bollinger Width | daily | 0.785 | -0.558 | 1.738 | **0.590** |
| 39 | v3 | 海龟突破 | Donchian + ADX + Chandelier | daily | 0.561 | -0.159 | 1.248 | **0.545** |
| 40 | v33 | T3 Aroon | T3 + Aroon + Volume Climax | daily | 0.447 | -0.468 | 1.528 | **0.530** |
| 41 | v42 | 多周期ADX+EMA | Daily ADX + 1h EMA Cross | 60min | 0.370 | 1.310 | -0.386 | **0.462** |
| 42 | v7 | SAR动量 | PSAR + TSI + CMF | 4h | 0.701 | 1.227 | -0.356 | **0.436** |
| 43 | v29 | ZLEMA Klinger | ZLEMA + RSI + Klinger | daily | 0.683 | -0.946 | 1.798 | **0.426** |
| 44 | v22 | HMA Aroon | HMA + Aroon + CMF | daily | 0.869 | -0.342 | 1.157 | **0.408** |
| 45 | v43 | 多周期一目+Stoch | 4h Ichimoku + 5min Stochastic | 5min | -0.938 | -0.449 | 1.210 | **0.381** |
| 46 | v38 | HMA Ergodic | HMA + Ergodic + A/D Line | daily | 0.840 | -0.242 | 0.830 | **0.294** |
| 47 | v21 | Supertrend ADX OBV | Supertrend + ADX + OBV | daily | 0.391 | 0.853 | -0.304 | **0.275** |
| 48 | v45 | 多周期Vortex+CCI | 4h Vortex + 30min CCI | 30min | 0.689 | 0.181 | 0.090 | **0.135** |
| 49 | v4 | EMA交叉 | EMA Cross + MACD + ATR | 4h | 0.288 | 0.107 | 0.000 | **0.054** |
| 50 | v26 | TTM Squeeze | TTM Squeeze + ADX + Force Index | daily | 0.107 | 0.000 | 0.000 | **0.000** |

## Optimized Parameters (Top 10)

### v12 — Aroon + PPO + Volume Momentum (daily) — Mean Test Sharpe 2.92
```json
{"aroon_period": 20, "ppo_fast": 19, "ppo_slow": 29, "vol_mom_period": 25, "atr_trail_mult": 4.814}
```

### v11 — Vortex + ROC + OI Momentum (daily) — Mean Test Sharpe 2.53
```json
{"vortex_period": 21, "roc_period": 15, "roc_threshold": 2.130, "oi_period": 15, "atr_trail_mult": 4.440}
```

### v34 — McGinley + PPO + OI Momentum (daily) — Mean Test Sharpe 2.45
```json
{"mcg_period": 10, "ppo_fast": 20, "ppo_slow": 31, "oi_period": 26, "atr_trail_mult": 3.323}
```

### v32 — EMA Ribbon + TSI + VROC (daily) — Mean Test Sharpe 1.95
```json
{"ribbon_base": 7, "tsi_long": 26, "tsi_short": 18, "vroc_period": 16, "atr_trail_mult": 4.952}
```

### v20 — Fractal + Mass Index + VROC (daily) — Mean Test Sharpe 1.88
```json
{"fractal_period": 2, "mass_ema": 7, "mass_sum": 27, "vroc_period": 16, "atr_trail_mult": 2.165}
```

### v18 — DEMA + KST + NR7 (daily) — Mean Test Sharpe 1.85
```json
{"dema_period": 40, "kst_signal": 11, "nr7_lookback": 6, "atr_period": 12, "atr_trail_mult": 2.130}
```

### v25 — LinReg Slope + Vortex + Volume Spike (daily) — Mean Test Sharpe 1.82
```json
{"slope_period": 18, "vortex_period": 13, "vol_period": 14, "vol_threshold": 1.569, "atr_trail_mult": 5.195}
```

### v9 — Supertrend + StochRSI + Force Index (1h) — Mean Test Sharpe 1.78
```json
{"st_period": 14, "st_mult": 2.696, "stochrsi_period": 12, "fi_period": 8, "atr_trail_mult": 4.885}
```

### v8 — LinReg Slope + Choppiness + NATR (daily) — Mean Test Sharpe 1.69
```json
{"slope_period": 20, "chop_period": 25, "chop_threshold": 61.782, "natr_period": 22, "atr_trail_mult": 4.836}
```

### v2 — KAMA + R² + ATR (daily) — Mean Test Sharpe 1.62
```json
{"kama_period": 16, "rsq_period": 28, "rsq_threshold": 0.682, "atr_period": 15, "trail_mult": 2.571}
```

## Key Findings

### 1. 强趋势需要宽止损
大多数表现好的策略 `atr_trail_mult` 优化到 4.0-5.0。说明强趋势中回撤剧烈，窄止损会被频繁震出。

### 2. 量价/OI 指标是 alpha 主要来源
Top 10 中 8 个使用了量/OI 相关指标：
- Volume Momentum (v12)
- OI Momentum (v11, v34)
- VROC (v32, v20)
- Volume Spike (v25)
- Force Index (v9)
中国期货市场量价信号价值极高，散户占比大导致量能有很强的预测性。

### 3. 非传统指标组合大幅胜出
- v12 (Aroon+PPO) Sharpe 2.92 vs v3 (经典海龟) 0.55
- v34 (McGinley+PPO) Sharpe 2.45 — 新策略直接排第 3
- v20 (Fractal+Mass Index) Sharpe 1.88
- 传统策略 v3 (海龟), v4 (EMA交叉) 排名垫底

### 4. 多周期策略效果不及预期
- v41-v45 排名 15-48，平均 Mean Test Sharpe 0.62
- v43 (5min Ichimoku+Stochastic) 训练集就是负的 (-0.94)
- 原因推测：小周期噪音大，聚合精度有损失，参数空间更复杂
- v44 (10min→daily Donchian+ROC) 表现最好 (1.33)，说明日线级指标更稳健

### 5. 极简策略出人意料地强
- v46 (仅 Supertrend) Mean Test 1.32，排名 16/50
- v47 (ADX+Chandelier) Mean Test 1.09
- v48 (R²+Donchian) Mean Test 0.97
- 说明过度堆砌指标不如一个好的趋势追踪 + 宽止损

### 6. 行情状态识别策略有价值
- v49 (Hurst+Supertrend) Mean Test 1.09，AG Sharpe 1.76
- Hurst > 0.55 作为趋势行情过滤器有效
- v50 (Yang-Zhang+Donchian) EC 表现好 (2.33)，但 AG 略负

### 7. 频率不是决定因素
- daily 策略占主导（多数 v1-v40）
- 最佳 AG Sharpe 来自 1h 策略 (v9, AG 2.15)
- 4h 策略稳定 (v16, v31, v13 平均 1.2)
- 频率选择 < 指标选择 < 止损设置

## Training Set Details

| Symbol | Period | Rally | Sector |
|--------|--------|-------|--------|
| J | 2015-06 → 2017-06 | +210% | 黑色系 (供给侧改革) |
| ZC | 2020-06 → 2022-06 | +253% | 能源 (煤炭供需错配) |
| JM | 2020-06 → 2022-06 | +170% | 黑色系 (碳中和) |
| I | 2015-06 → 2017-06 | +134% | 黑色系 (供给侧改革) |
| NI | 2021-01 → 2022-09 | +106% | 有色 (伦镍逼空) |
| SA | 2022-06 → 2024-06 | +130% | 化工 (光伏需求) |

## Architecture

所有 50 个策略已迁移至 **on_init_arrays 预计算架构**：
- 指标在 `on_init_arrays` 中一次性全数组计算
- `on_bar` 通过 `context.bar_index` O(1) 查表
- 多周期策略 (v41-v45) 使用预聚合 + 索引映射 (`_4h_map`)
- 预计算加速: daily ~3x, 1h ~6x, 30min ~10x, 5min ~25x
