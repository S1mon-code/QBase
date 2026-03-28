# AlphaForge API 参考

> AlphaForge 是 QBase 的回测引擎。路径配置：[`config.yaml`](../../config.yaml)

## Context API

策略通过 `context` 对象与引擎交互。分为两个阶段：`on_init_arrays`（初始化，全数组预计算）和 `on_bar`（逐 bar 执行）。

### on_init_arrays 阶段（全数组）

```python
def on_init_arrays(self, context, bars):
    # 价格数组（前复权，numpy array, shape=(N,)）
    closes = context.get_full_close_array()      # 完整复权 close
    highs  = context.get_full_high_array()       # 完整 high
    lows   = context.get_full_low_array()        # 完整 low
    opens  = context.get_full_open_array()       # 完整 open

    # 量价数组
    volumes = context.get_full_volume_array()    # 完整 volume
    oi      = context.get_full_oi_array()        # 完整持仓量

    # 原始价格（未复权）
    closes_raw = context.get_full_close_raw_array()  # 完整原始 close（未复权）
```

### on_bar 阶段（逐 bar）

```python
def on_bar(self, context):
    i = context.bar_index               # 当前 bar 在完整数组中的索引（关键！）
    val = self._precomputed[i]          # O(1) 查表

    # 快速属性访问（跳过 Bar namedtuple 创建，推荐）
    price = context.close_raw           # 比 context.current_bar.close_raw 更快
    context.open_raw / context.high_raw / context.low_raw / context.volume

    # 传统方式（仍可用，但慢）
    context.current_bar.close_raw
    context.get_close_array(n)          # 最近 n 根 close

    # 持仓与权益
    side, lots = context.position       # side: 1=多, -1=空, 0=无
    context.equity                      # 当前权益
    context.available_cash              # 可用资金

    # 换仓检测
    if context.is_rollover:             # ✅ 正确写法
        return                          # 换仓日不交易

    # 交易指令（下一 bar 的 open 执行）
    context.buy(lots)                   # 开多
    context.sell(lots)                  # 开空
    context.close_long()                # 平多（全部）
    context.close_long(lots=5)          # 平多（指定手数）
    context.close_short()               # 平空
```

---

## V4 性能优化（必须遵守）

**当前版本：V4.3.1** — 5min 回测从 77s → 0.56s（137x 加速），QBase 实测 140-200x 加速。

### P0 必须改（不改就慢 100x）

**1. `context.is_rollover` 替代 `context.current_bar.is_rollover`**

```python
# ❌ 旧方式（每 bar 创建 17 字段 Bar namedtuple）
if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:

# ✅ 新方式（直接数组访问，零对象创建）
if context.is_rollover:
```

同样可用：`context.origin_symbol`（替代 `context.current_bar.origin_symbol`）。

**2. `ContractSpecManager()` 缓存到模块级别**

```python
# ❌ 旧方式（每次交易读 YAML — 6000 笔 = 6000 次文件 IO）
def _calc_lots(self, context, atr_val):
    from alphaforge.data.contract_specs import ContractSpecManager
    spec = ContractSpecManager().get(context.symbol)

# ✅ 新方式（模块级别单例，只读一次）
from alphaforge.data.contract_specs import ContractSpecManager
_SPEC_MANAGER = ContractSpecManager()

class MyStrategy(TimeSeriesStrategy):
    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
```

**全局替换命令：**
```bash
# 1. 替换 is_rollover
grep -rl "context.current_bar.is_rollover" strategies/ | xargs sed -i '' \
  's/hasattr(context.current_bar, '\''is_rollover'\'') and context.current_bar.is_rollover/context.is_rollover/g'

# 2. ContractSpecManager 需手动改为模块级缓存
grep -rn "ContractSpecManager()" strategies/ --include="*.py"
```

### P1 推荐改

**3. `signal_mask()` 向量化预筛选（稀疏策略额外 10-50x 加速）**

```python
from alphaforge.indicators import crossover, crossunder

class MyStrategy(TimeSeriesStrategy):
    def signal_mask(self, context, bars):
        """只在信号 bar 调用 on_bar，跳过 95%+ 无信号 bar"""
        return crossover(self._sma_fast, self._sma_slow) | crossunder(self._sma_fast, self._sma_slow)
```

规则：返回 `None` = 不启用 | `np.ndarray[bool]` = 启用 | **有持仓时始终调用 on_bar** | Mask 应保守

**4. `on_init_arrays_static()` — ML 指标只算一次**

```python
class MyStrategy(TimeSeriesStrategy):
    def on_init_arrays_static(self, context, bars):
        """Optuna 优化前只调一次，放重的 ML 指标"""
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._regime = kmeans_regime(features, period=120)  # ~3s，只算一次

    def on_init_arrays(self, context, bars):
        """每个 trial 调。self._regime 已从缓存注入。"""
        self._sma = sma(closes, self.fast_period)  # 参数相关，每次重算
```

效果：ML 指标 3s x 200 trials = 600s → 只算 1 次 = 3s。

**5. Numba 加速指标库**

```python
from alphaforge.indicators import sma, ema, rsi, atr, macd, bollinger_bands, supertrend, crossover, crossunder
```

全部 `@njit(cache=True)`，C 速度。QBase 自己的 `indicators/` 库仍可用。

**6. 零拷贝数组访问**

```python
closes = context.get_close_array_view(20)  # 只读 view，零拷贝（不可修改）
```

### QBase 实测性能

| 策略 | 类型 | Before V4 | After V4 | 加速 |
|------|------|:---------:|:--------:|:----:|
| v1 | 纯 5min | 5.58s | 0.04s | **140x** |
| v121 | 多周期 (daily+5min) | 1.34s | 0.03s | **45x** |
| v14 | 跨品种 (AG+AU) | 1.07s | 0.02s | **54x** |

结果完全一致，纯性能提升零行为变更。

### 性能对比（按频率）

| 周期 | ~Bars (5年) | 传统模式 | 预计算模式 | 加速比 |
|------|------------|---------|-----------|--------|
| 1min | ~1,500K | >10min | ~8-15s | ~50x |
| 5min | ~350K | ~98s | ~3-5s | ~25x |
| 30min | ~25K | ~7s | <1s | ~10x |
| 1h | ~12K | ~3s | <0.5s | ~6x |
| daily | ~1.2K | <0.3s | <0.1s | ~3x |

---

## 组合回测（PortfolioBacktester）

```python
from alphaforge.engine.portfolio import PortfolioConfig, StrategyAllocation, PortfolioBacktester
from alphaforge.data.contract_specs import ContractSpecManager

config = PortfolioConfig(
    total_capital=3_000_000,
    allocations=[
        StrategyAllocation(
            name="trend_v9",
            strategy_file="strategies/strong_trend/v9.py",
            symbols="AG",
            freq="1h",
            weight=0.4,
            params={},  # 可选：覆盖策略属性
        ),
        StrategyAllocation("trend_v12", "strategies/strong_trend/v12.py", "AG", "daily", weight=0.3),
        StrategyAllocation("trend_v1",  "strategies/strong_trend/v1.py",  "AG", "daily", weight=0.3),
    ],
    rebalance="none",  # "none", "monthly", "quarterly"
)

runner = PortfolioBacktester(spec_manager=ContractSpecManager(), data_dir="data/")
report = runner.run(config, start="2023-01-01", end="2025-12-31")

# 组合指标
report.combined_result.sharpe / .total_return / .max_drawdown / .equity_curve
# 逐策略
for slot in report.slot_results:
    slot.name, slot.weight, slot.result.sharpe
# 相关性矩阵
report.correlation_matrix  # pd.DataFrame
```

### Portfolio 新功能

```python
config = PortfolioConfig(
    total_capital=5_000_000,
    allocations=[...],
    n_workers=4,                        # 并行回测
    min_capital_per_strategy=500_000,   # 防小权重开不了仓
)
```

---

## BacktestConfig 新选项

```python
config = BacktestConfig(
    dynamic_margin=True,        # 交割月阶梯保证金（T-3:+3%, T-2:+5%, T-1:+8%）
    time_varying_spread=True,   # 盘口时段价差（开盘/收盘加宽）
)
```

已配置动态保证金：I、RB、CU、AG、M。

---

## Report 生成规范

### 目录结构

所有 HTML 报告按策略类型和品种组织：

```
reports/
├── strong_trend/
│   ├── ag/
│   │   ├── portfolio.html      # AG 强趋势 Portfolio 报告
│   │   ├── v7.html ~ v49.html  # 各策略单独报告
│   └── lc/
│       ├── portfolio.html      # LC 强趋势 Portfolio 报告
│       └── v5.html ~ v34.html
├── medium_trend/
│   └── <symbol>/
├── all_time/
│   └── ag/
└── ...
```

### 必须使用优化参数

**生成回测报告或 portfolio 报告时，必须传入 Optuna 优化后的参数。不能用默认参数。**

```python
# ❌ 错误：用默认参数（未经优化）
strat = StrategyV20()
result = engine.run(strat, {'AG': bars})
reporter.generate(result, 'reports/v20.html')  # 结果不准确！

# ✅ 正确：传入优化后的参数
from strategies.strong_trend.optimizer import create_strategy_with_params
import json

with open('strategies/strong_trend/optimization_results.json') as f:
    opt_results = json.load(f)
params = {r['version']: r['best_params'] for r in opt_results}

strat = create_strategy_with_params('v20', params['v20'])
result = engine.run(strat, {'AG': bars}, warmup=strat.warmup)
reporter.generate(result, 'reports/strong_trend/ag/v20.html')
```

**教训（v20 参数 bug）：** 默认参数 v20 AG Sharpe=-0.78，优化后为 1.80。差异巨大。

### 单策略报告

**必须传 `bar_data` 和 `freq` 才有 K 线图 + 进出场标记。不传就没有 K 线。**

```python
from alphaforge.report import HTMLReportGenerator

reporter = HTMLReportGenerator()
reporter.generate(
    result,
    'reports/strong_trend/ag/v20.html',
    bar_data={'AG': bars},  # 必须传这个才有 K-line
    freq='daily',           # K 线频率
)
```

### Portfolio 报告（新 API）

**必须传入 `slot_results` 和 `kline_bars`**，这样报告中会有：
- 每个策略的单独报告 + K 线图 + 进出场标记
- 策略链接和权重信息
- 组合级净值曲线和回撤

```python
from alphaforge.engine.portfolio import PortfolioConfig, StrategyAllocation, PortfolioBacktester
from alphaforge.report import HTMLReportGenerator

# 1. 加载优化参数
with open('strategies/strong_trend/optimization_results.json') as f:
    opt_results = json.load(f)
params_map = {r['version']: r.get('best_params', {}) for r in opt_results}

# 2. 构建 PortfolioConfig（必须传入 params）
config = PortfolioConfig(
    total_capital=3_000_000,
    allocations=[
        StrategyAllocation('v20', 'strategies/strong_trend/v20.py', 'AG', 'daily',
                           weight=0.20, params=params_map['v20']),
        StrategyAllocation('v12', 'strategies/strong_trend/v12.py', 'AG', 'daily',
                           weight=0.144, params=params_map['v12']),
        # ... 其他策略
    ],
    rebalance='none',
)

# 3. 运行回测
runner = PortfolioBacktester(data_dir=get_data_dir())
report = runner.run(config, start='2025-01-01', end='2026-03-01')

# 4. 构建 kline_bars（从 slot_results 提取 K 线数据）
kline_bars = {}
for s in report.slot_results:
    if s.bar_data:
        sym = s.symbol.split(',')[0].strip().upper()
        if sym in s.bar_data:
            kline_bars[sym] = s.bar_data[sym]

# 5. 生成报告（必须传 slot_results + kline_bars）
reporter = HTMLReportGenerator()
reporter.generate_portfolio_report(
    [s.result for s in report.slot_results],
    report.combined_result,
    'reports/strong_trend/ag/portfolio.html',
    slot_results=report.slot_results,     # 关键：单策略报告+链接+权重
    kline_bars=kline_bars,                # 关键：K 线图+进出场标记
    kline_freq='daily',                   # K 线频率
)
```

**不传 `slot_results` 就没有单策略报告和 K 线。这是必须的。**

### 当前 Portfolio C 组成

**Strong Trend 使用统一通用 Portfolio（Portfolio C），同一组策略和权重跑任何品种：**

| 权重 | 策略 | 指标组合 | 频率 |
|:---:|------|---------|:---:|
| 25% | v12 | Aroon + PPO + Volume Momentum | daily |
| 20% | v8 | LinReg + Choppiness + OBV | daily |
| 20% | v11 | Vortex + ROC + OI Momentum | daily |
| 15% | v34 | McGinley + PPO + OI Momentum | daily |
| 20% | v31 | TEMA + Fisher + OBV | 4h |

| 品种 | Sharpe | Return | MaxDD | 路径 |
|------|:------:|:------:|:-----:|------|
| AG | 2.58 | 66.13% | -12.80% | `reports/strong_trend/ag/portfolio.html` |
| LC | 2.37 | 19.19% | -3.99% | `reports/strong_trend/lc/portfolio.html` |

---

## 数据加载

### 支持的频率

| 频率 | AlphaForge 原生 | 备注 |
|------|:-:|------|
| 1min | ✅ | 噪音大，不推荐 |
| 5min | ✅ | 最低推荐频率 |
| 10min | ✅ | |
| 15min | ✅ | |
| 30min | ✅ | |
| 60min | ✅ | |
| daily | ✅ | |
| 1h | 用 60min | 别名 |
| 4h | 60min x 4 重采样 | `resample_bars(bars, 4)` |
| 20min | 10min x 2 重采样 | `resample_bars(bars, 2)` |

**支持任意频率：** `<N>min`（如 `1min`, `5min`, `20min`）、`<N>h`（如 `1h`, `4h`）、`daily`

### BarArray 属性

```python
bars.datetime       # 时间
bars.open           # 前复权开盘价
bars.high           # 前复权最高价
bars.low            # 前复权最低价
bars.close          # 前复权收盘价
bars.volume         # 成交量
bars.amount         # 成交额
bars.oi             # 持仓量
bars.trading_day    # 交易日
bars.open_raw       # 未复权开盘价
bars.close_raw      # 未复权收盘价
bars.origin_symbol  # 原始合约代码
bars.factor         # 复权因子
bars.is_rollover    # 是否换仓日
```

### 数据加载示例

```python
from alphaforge.data.market import MarketDataLoader

loader = MarketDataLoader(data_dir)
symbols = loader.available_symbols()  # 所有可用品种
bars = loader.load("AG", freq="daily", start="2012-05-10")
```

---

## BacktestResult

`engine.run()` 返回 `BacktestResult` 对象：

```python
result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)

# 核心指标
result.sharpe           # float: Sharpe ratio
result.max_drawdown     # float: 最大回撤（负数）
result.total_return     # float: 总收益率
result.n_trades         # int: 总交易次数（或 result.total_trades）

# 权益曲线
result.equity_curve     # pandas Series 或 numpy array
# 或
result.equity           # 备选属性名

# 交易明细（归因分析需要）
result.trades           # DataFrame: 每笔交易记录
```

---

## 回测引擎

```python
from alphaforge.data.market import MarketDataLoader
from alphaforge.data.contract_specs import ContractSpecManager
from alphaforge.engine.event_driven import EventDrivenBacktester

loader = MarketDataLoader(data_dir)
bars = loader.load(symbol, freq="daily", start="2022-01-01", end="2026-03-01")

engine = EventDrivenBacktester(
    spec_manager=ContractSpecManager(),
    initial_capital=1_000_000,
    slippage_ticks=1.0,          # 滑点（tick 数）
)
result = engine.run(strategy, {symbol: bars}, warmup=strategy.warmup)
```

---

## 运行命令

```bash
# 单品种单频率
./run.sh strategies/strong_trend/v1.py --symbols AG --freq daily --start 2022

# 多品种测试
./run.sh strategies/strong_trend/v1.py --symbols AG,J,ZC --freq daily --start 2020

# 不同频率
./run.sh strategies/strong_trend/v9.py --symbols AG --freq 60min --start 2024
```

---

## 性能优化要点汇总

| 优化 | 影响 | 方法 |
|------|------|------|
| **on_init_arrays 预计算** | ~20x 加速 | 指标一次算完，on_bar 查表 |
| **ContractSpecManager 缓存** | 避免重复初始化 | 模块级单例 |
| **numpy 向量化** | 10-50x vs pandas | 指标用纯 numpy |
| **signal_mask()** | 额外 10-50x | 稀疏策略跳过无信号 bar |
| **on_init_arrays_static()** | ML 指标只算一次 | Optuna 多 trial 不重复 |
| **Numba 指标** | C 速度 | `@njit(cache=True)` |
| **零拷贝 view** | 减少内存分配 | `get_close_array_view(n)` |
| **频率重采样** | 减少 bar 数 | 4h = 60min x 4 |
| **warmup 设合理值** | 避免浪费 bar | max(lookback) + 20 |

---

## 配置文件

```yaml
# config.yaml
alphaforge:
  path: ~/Desktop/AlphaForge
  data_dir: ~/Desktop/AlphaForge/data

defaults:
  capital: 1_000_000
  freq: daily
  slippage_ticks: 2.0
```

```python
# config.py
from config import get_data_dir
data_dir = get_data_dir()  # 读取 config.yaml 中的 data_dir
```
