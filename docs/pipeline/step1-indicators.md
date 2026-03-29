# Step 1: 指标池（Indicator Pool）

> 完整指标目录见 [`indicators/indicators.md`](../../indicators/indicators.md)。

## 指标库概览

10 大类 320 个指标，纯 numpy 函数（numpy in → numpy out）：

| 分类 | 数量 | 说明 |
|------|:---:|------|
| momentum | 35 | 动量/振荡类 |
| trend | 35 | 趋势类 |
| volatility | 35 | 波动率类 |
| volume | 39 | 成交量/持仓类 |
| microstructure | 15 | 市场微观结构 |
| ml | 65 | 机器学习/统计学习 |
| regime | 33 | 行情状态识别 |
| seasonality | 15 | 季节性/日历效应 |
| spread | 25 | 跨品种价差/比值 |
| structure | 23 | 持仓结构分析 |

## 新指标开发规则

**指标来源不限于 320 个库存指标。** 可以随时 research 并开发新指标加入库中，例如：
- 品种特有指标（如铁煤比 I/J ratio、金银比 AU/AG ratio）
- 跨品种价差/比值指标
- 基本面衍生指标（库存、持仓结构等）
- 自定义统计量
- 网上 research 发现的有效指标

开发新指标时**必须**：
1. 放入 `indicators/` 对应分类，保持纯函数风格（numpy in → numpy out）
2. **更新 `indicators/indicators.md`**，添加新指标的名称、文件、函数签名

## 分类规则（10 个现有分类）

| 分类 | 放入 | 示例 |
|------|------|------|
| 品种比值/价差 | `indicators/spread/` | 铁煤比、金银比 |
| 跨品种相关性 | `indicators/spread/` | 板块动量扩散 |
| 波动率衍生 | `indicators/volatility/` | 自定义波动率模型 |
| 量价/持仓衍生 | `indicators/volume/` | OI 结构分析 |
| 趋势衍生 | `indicators/trend/` | 自定义趋势评分 |
| 市场微观结构 | `indicators/microstructure/` | 流动性、价格冲击 |
| 机器学习特征 | `indicators/ml/` | Kalman、PCA、聚类 |
| 行情状态识别 | `indicators/regime/` | 趋势/震荡判断 |
| 季节性/日历 | `indicators/seasonality/` | 月效应、节假日 |
| 持仓结构 | `indicators/structure/` | OI 分析、资金流向 |

如果现有 10 个分类都不合适，可以新建分类文件夹。

## 各分类使用规范

**原始 4 类（momentum/trend/volatility/volume）：** 标准 OHLCV 数据即可，直接在 `on_init_arrays` 中使用。

**spread 类：** 需要多品种数据。策略中需额外加载第二品种的价格数组：
```python
# 示例：在 AG 策略中使用金银比
from indicators.spread.gold_silver_ratio import gold_silver_ratio
# 需要在 on_init_arrays 中同时获取 AU 的 close 数组
```

**seasonality 类：** 需要 datetime 数组。通过 `context.get_full_datetime_array()` 获取：
```python
from indicators.seasonality.weekday_effect import weekday_effect
datetimes = context.get_full_datetime_array()
effect = weekday_effect(closes, datetimes, lookback=252)
```

**ml 类：** 多数需要 features_matrix（多列特征矩阵）。通常先计算若干基础指标，再组合成矩阵传入：
```python
from indicators.ml.kmeans_regime import kmeans_regime
features = np.column_stack([rsi_arr, adx_arr, atr_arr])
regime = kmeans_regime(features, period=120, n_clusters=3)
```

**regime 类：** 用于判断当前行情状态（趋势/震荡/突破），适合作为策略的过滤层而非核心信号。

**structure 类：** 依赖 OI（持仓量）数据，通过 `context.get_full_oi_array()` 获取。适合期货品种，非期货品种不可用。

**microstructure 类：** 分析市场微观结构（流动性、价格冲击等），适合高频策略（5min/10min）的辅助过滤。

## Alpha 贡献（来自归因分析）

- **量价/OI 指标是 alpha 的几乎全部来源**，不只是 "重要"
- Volume Momentum 单指标贡献了 v12（最强策略）**51.7%** 的 alpha
- 传统趋势指标（Aroon）alpha 贡献仅 **0.1%**
- 动量指标（PPO）alpha 贡献仅 **2.8%**
- 中国期货散户占比高 → 量能信号比西方市场有价值

## 指标选择指南（写策略时参考）

| 场景 | 推荐指标 | 不推荐 |
|------|---------|--------|
| 强趋势策略 | Volume Momentum, OI Momentum, Force Index | EMA Cross, MACD Cross |
| 趋势过滤 | Hurst, ADX (仅做门槛) | Aroon (alpha 贡献低) |
| 止损基准 | ATR (mult > 4.0) | 固定点数 |
| 行情识别 | Yang-Zhang, Efficiency Ratio | TTM Squeeze (强趋势无效) |
| 入场精确度 | Fractal Level, NR7 Squeeze | — |

## 添加新指标的 Checklist

1. 写纯 numpy 函数，放入 `indicators/<分类>/`
2. 用测试数据验证 NaN 处理和边界情况
3. 更新 `indicators/indicators.md`
4. Commit: `[indicator] add <name> to <category>`

### 函数签名规范

```python
# 所有指标必须是纯 numpy 函数
def my_indicator(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """一句话描述。"""
    result = np.full_like(closes, np.nan)
    # ... numpy 计算 ...
    return result
```

**必须：**
- 输入：numpy array（不接受 pandas）
- 输出：numpy array（与输入等长，前 N 个为 NaN）
- 无状态、无副作用
- 返回值个数在 docstring 中说明（单值 / tuple）

**禁止：**
- 不要用 pandas（性能差 10-50x）
- 不要引入外部状态
- 不要在指标函数中做交易逻辑

### 命名和归类

- 文件名：`snake_case.py`，放在对应分类目录下
- 函数名：与文件名一致（如 `volume_momentum.py` → `volume_momentum()`）
- 新增指标立即更新 `indicators/indicators.md`

### 性能要求

- 指标在 `on_init_arrays` 中一次性全数组计算
- 10 万 bar 的计算应 < 1s
- 避免 Python for 循环，优先用 numpy 向量化
- 复杂计算可用 `@njit`（numba），但不是必须
