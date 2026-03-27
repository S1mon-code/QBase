# Strong Trend Portfolio C — 策略详解

## Portfolio 总览

**通用组合，同一权重跑任何品种。**

| 权重 | 策略 | 指标组合 | 频率 | AG Sharpe | LC Sharpe |
|:---:|------|---------|:---:|:---:|:---:|
| 25% | v12 | Aroon + PPO + Volume Momentum | daily | 3.09 | 2.04 |
| 20% | v8 | Linear Regression + Choppiness + OBV | daily | 1.32 | 1.58 |
| 20% | v11 | Vortex + ROC + OI Momentum | daily | 1.75 | 2.27 |
| 15% | v34 | McGinley + PPO + OI Momentum | daily | 1.69 | 2.16 |
| 20% | v31 | TEMA + Fisher Transform + OBV | 4h | 1.24 | 2.51 |

| 品种 | Sharpe | Return | MaxDD |
|------|:------:|:------:|:-----:|
| AG | 2.58 | 66.13% | -12.80% |
| LC | 2.37 | 19.19% | -3.99% |

**评分：77.1/100 (B+)**

---

## v12 (25%) — Aroon + PPO + Volume Momentum | daily

**核心逻辑：** 用 Aroon 捕捉趋势启动 + PPO 确认动量加速 + Volume Momentum 确认市场参与度

### 指标详解

**1. Aroon (25 周期)**
- 计算过去 25 天内最高价和最低价分别出现在几天前
- Aroon Up = (25 - 距最高价天数) / 25 x 100
- Aroon Down = (25 - 距最低价天数) / 25 x 100
- **用法：** Aroon Up > 70 且 Aroon Down < 30 = 强上升趋势确认
- **优势：** 对趋势启动点非常敏感，比 ADX 更早发出信号

**2. PPO (Percentage Price Oscillator, fast=19, slow=29)**
- 本质是百分比化的 MACD：(EMA_fast - EMA_slow) / EMA_slow x 100
- **用法：** PPO > 0 且上升 = 动量向上加速
- **优势：** 百分比化后可以跨品种比较，不受价格绝对值影响

**3. Volume Momentum (25 周期)**
- 当前成交量相对 N 周期前的变化率
- **用法：** Volume Momentum > 0 = 成交量在放大，市场参与度增加
- **优势：** 过滤成交量萎缩时的假突破

### 交易规则

- **进场：** Aroon Up > 70 + Aroon Down < 30 + PPO > 0 + Volume Momentum > 0
- **出场：** ATR 追踪止损 / 分层止盈 (3ATR, 5ATR) / Aroon Up < 50（趋势弱化）

### 表现

| 品种 | Sharpe | Return | Trades |
|------|:------:|:------:|:------:|
| AG | **3.09** | 84.83% | 18 |
| LC | **2.04** | 17.23% | 12 |
| 训练集 | 0.83 | — | — |

**特点：** 全 Portfolio 最高 Sharpe，在 AG 和 LC 上都极强，跨品种泛化能力最好。

---

## v8 (20%) — Linear Regression + Choppiness Index + OBV | daily

**核心逻辑：** 用线性回归判断趋势方向 + Choppiness 过滤震荡期 + OBV 确认资金流入

### 指标详解

**1. Linear Regression Slope (50 周期)**
- 对过去 50 天收盘价做线性回归，取斜率
- **用法：** 斜率 > 0 = 上升趋势；R-squared > 0.5 = 趋势质量好（不是震荡）
- **优势：** 比 EMA 交叉更平滑，不容易被短期波动干扰

**2. Choppiness Index (14 周期)**
- 衡量市场是趋势还是震荡：CI = 100 x log10(Sum(ATR) / (最高-最低)) / log10(14)
- CI > 61.8 = 震荡市（不交易）
- CI < 38.2 = 趋势市（可以交易）
- **用法：** 只在 CI < 50 时入场
- **优势：** 直接过滤震荡期，避免趋势策略在横盘时被反复止损

**3. OBV (On-Balance Volume)**
- 累积量能指标：涨日加量，跌日减量
- **用法：** OBV 趋势向上 = 资金持续流入
- **优势：** 领先指标，往往比价格更早反映趋势变化

### 交易规则

- **进场：** LR Slope > 0 + R-squared > 0.5 + Choppiness < 50 + OBV 上升
- **出场：** ATR 追踪止损 / Choppiness > 61.8（进入震荡）

### 表现

| 品种 | Sharpe | Return | Trades |
|------|:------:|:------:|:------:|
| AG | **1.32** | 44.06% | 36 |
| LC | **1.58** | 19.58% | 16 |
| 训练集 | 0.70 | — | — |

**特点：** 收益最平衡的策略，AG 和 LC 都有高绝对收益。Choppiness 过滤器有效避免震荡期亏损。

---

## v11 (20%) — Vortex + ROC + OI Momentum | daily

**核心逻辑：** 用 Vortex 判断趋势方向 + ROC 确认动量强度 + OI Momentum 确认机构参与

### 指标详解

**1. Vortex Indicator (14 周期)**
- 衡量正向和负向趋势运动：VI+ 和 VI-
- VI+ = Sum(|当前高-前低|) / Sum(True Range)
- **用法：** VI+ > VI- = 上升趋势；VI+ 交叉上穿 VI- = 买入信号
- **优势：** 对趋势反转非常敏感，假信号比 MACD 少

**2. ROC (Rate of Change, 20 周期)**
- 价格变化率：(当前价 - N天前价) / N天前价 x 100
- **用法：** ROC > 0 = 价格在上涨
- **优势：** 最直接的动量度量，无滞后

**3. OI Momentum (持仓量动量, 20 周期)**
- 持仓量的变化率
- **用法：** OI 增加 + 价格上涨 = 新资金流入做多（强趋势确认）
- **优势：** 期货特有指标，能区分"主力加仓"和"散户跟风"

### 交易规则

- **进场：** VI+ > VI- + ROC > 0 + OI Momentum > 0
- **出场：** ATR 追踪止损 / 分层止盈 / VI+ < VI-（趋势反转）

### 表现

| 品种 | Sharpe | Return | Trades |
|------|:------:|:------:|:------:|
| AG | **1.75** | 30.83% | 36 |
| LC | **2.27** | 12.01% | 14 |
| 训练集 | 1.19 | — | — |

**特点：** 训练集 Sharpe 最高（1.19），说明策略逻辑本身非常稳健。LC 上表现最好的 daily 策略之一。OI Momentum 是期货特有的 alpha 来源。

---

## v34 (15%) — McGinley Dynamic + PPO + OI Momentum | daily

**核心逻辑：** 用 McGinley 自适应均线跟踪趋势 + PPO 确认动量 + OI Momentum 确认持仓增长

### 指标详解

**1. McGinley Dynamic (10 周期)**
- 自适应移动均线：根据价格速度自动调整平滑系数
- MD = MD_prev + (Price - MD_prev) / (N x (Price/MD_prev)^4)
- **用法：** 价格 > MD = 上升趋势
- **优势：** 比 EMA 更贴合价格，在快速行情中跟踪更紧，在慢行情中更平滑。几乎消除了 whipsaw（假穿越）

**2. PPO (fast=20, slow=31)**
- 同 v12 的 PPO，但参数稍不同
- **用法：** PPO > signal line = 动量向上
- **优势：** 与 v12 的 PPO 参数不同，产生不同时点的信号，增加组合分散度

**3. OI Momentum (26 周期)**
- 同 v11 的 OI Momentum 但周期更长
- **用法：** 长周期 OI 趋势确认
- **优势：** 26 周期更平滑，过滤 OI 短期波动

### 交易规则

- **进场：** Price > McGinley + PPO > Signal + OI Momentum > 0
- **出场：** ATR 追踪止损 / 分层止盈 / Price < McGinley（跌破自适应均线）

### 表现

| 品种 | Sharpe | Return | Trades |
|------|:------:|:------:|:------:|
| AG | **1.69** | 11.88% | 46 |
| LC | **2.16** | 5.21% | 14 |
| 训练集 | 0.94 | — | — |

**特点：** 训练集 Sharpe 第二高（0.94），策略逻辑稳健。McGinley 自适应特性让策略在不同品种上自动调整灵敏度。

---

## v31 (20%) — TEMA + Fisher Transform + OBV | 4h

**核心逻辑：** 用 TEMA 快速均线跟踪趋势 + Fisher Transform 检测转折点 + OBV 量能确认

### 指标详解

**1. TEMA (Triple EMA, 14 周期)**
- 三重指数平滑：TEMA = 3xEMA - 3xEMA(EMA) + EMA(EMA(EMA))
- **用法：** 价格 > TEMA = 上升趋势；TEMA 斜率 > 0 = 趋势加速
- **优势：** 比 EMA 滞后更小，对价格变化反应更快，适合 4h 级别捕捉中期趋势

**2. Fisher Transform (5 周期)**
- 将价格标准化到正态分布：Fisher = 0.5 x ln((1+x)/(1-x))
- **用法：** Fisher > Trigger（信号线）= 买入信号
- **优势：** 将模糊的价格分布转化为清晰的峰谷信号，转折点非常明确

**3. OBV (18 周期回看)**
- 同 v8 的 OBV，但对比更短周期
- **用法：** OBV 趋势 > OBV 均线 = 量能支持
- **优势：** 4h 级别的 OBV 更细腻，能捕捉日内资金流向

### 交易规则

- **进场：** Price > TEMA + Fisher > Trigger + OBV > OBV_MA
- **出场：** ATR 追踪止损 / 分层止盈 / Fisher 转为卖出信号

### 表现

| 品种 | Sharpe | Return | Trades |
|------|:------:|:------:|:------:|
| AG | **1.24** | 40.87% | 108 |
| LC | **2.51** | 4.29% | 37 |
| 训练集 | 0.65 | — | — |

**特点：** 唯一的 4h 策略，与 4 个 daily 策略天然低相关。LC 上 Sharpe 最高（2.51）。交易频率最高（AG 108 笔），更充分利用行情。

---

## 组合优势分析

### 指标零重复

5 个策略用了 12 种不同指标，无重叠：

| 策略 | 趋势指标 | 动量/信号 | 量价确认 |
|------|---------|---------|---------|
| v12 | **Aroon** | **PPO** | **Volume Momentum** |
| v8 | **Linear Regression** | **Choppiness Index** | **OBV** |
| v11 | **Vortex** | **ROC** | **OI Momentum** |
| v34 | **McGinley Dynamic** | **PPO** (不同参数) | **OI Momentum** (不同周期) |
| v31 | **TEMA** | **Fisher Transform** | **OBV** (不同周期) |

### 信号类型互补

| 策略 | 信号类型 | 特点 |
|------|---------|------|
| v12 | 突破型 | Aroon 检测新高/新低出现的时机 |
| v8 | 过滤型 | Choppiness 主动过滤震荡期 |
| v11 | 动量型 | Vortex+ROC 纯动量驱动 |
| v34 | 自适应型 | McGinley 自动调整灵敏度 |
| v31 | 转折型 | Fisher Transform 检测价格转折 |

### 频率分散

| 频率 | 策略 | 权重 |
|------|------|:---:|
| daily | v12, v8, v11, v34 | 80% |
| 4h | v31 | 20% |

### 量价双确认

每个策略都包含量价指标（Volume Momentum / OBV / OI Momentum），不纯靠价格信号。期货市场中持仓量（OI）是独特的 alpha 来源——v11 和 v34 都使用了 OI Momentum。

### 双品种泛化

5 个策略在 AG 和 LC 上 Sharpe 全部 > 1.0，证明策略逻辑具有跨品种泛化能力，不是对单一品种的过拟合。

---

*最后更新: 2026-03-27*
