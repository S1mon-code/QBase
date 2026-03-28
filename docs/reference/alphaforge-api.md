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

## V6.0 性能优化（必须遵守）

**当前版本：V6.0**（1505 tests） — 5min 回测从 77s → 0.56s（137x 加速），QBase 实测 140-200x 加速。

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

## BacktestConfig（V6.0 扩展至 ~30 参数）

### 基础参数（V4 已有）

```python
config = BacktestConfig(
    dynamic_margin=True,        # 交割月阶梯保证金（T-3:+3%, T-2:+5%, T-1:+8%）
    time_varying_spread=True,   # 盘口时段价差（开盘/收盘加宽）
)
```

已配置动态保证金：I、RB、CU、AG、M。

### V6.0 新增参数

```python
config = BacktestConfig(
    # --- V4 基础参数 ---
    dynamic_margin=True,
    time_varying_spread=True,

    # --- V6 新增参数 ---
    margin_check_mode="daily",       # 保证金检查模式: "bar" (每bar) | "daily" (每日结算)
    margin_call_grace_bars=5,        # 保证金追缴宽限期（bar数）
    rollover_window_bars=20,         # 渐进式换仓窗口（替代单bar换仓）
    detect_locked_limit=True,        # 涨跌停锁仓检测
    volume_adaptive_spread=True,     # 成交量自适应价差
    asymmetric_impact=True,          # 非对称市场冲击（大单vs小单）
    overnight_gap=True,              # 隔夜跳空风险模拟
    auction_spread=True,             # 集合竞价价差扩大
    broker=BrokerConfig(...),        # 券商配置（见下方 BrokerConfig 章节）
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `margin_check_mode` | str | `"bar"` | `"bar"`: 每bar检查保证金; `"daily"`: 每日结算时检查（更贴近真实交易所） |
| `margin_call_grace_bars` | int | `0` | 保证金追缴宽限期，0=立即强平 |
| `rollover_window_bars` | int | `1` | 换仓窗口，>1 时渐进式换仓（分多bar完成） |
| `detect_locked_limit` | bool | `False` | 启用涨跌停锁仓检测，锁仓时拒绝平仓 |
| `volume_adaptive_spread` | bool | `False` | 价差随成交量动态调整（低量时价差扩大） |
| `asymmetric_impact` | bool | `False` | 大单市场冲击更大（非线性） |
| `overnight_gap` | bool | `False` | 模拟隔夜跳空风险 |
| `auction_spread` | bool | `False` | 集合竞价时段价差扩大 |
| `broker` | BrokerConfig | `None` | 券商级别配置 |

### 两种回测模式

V6 引擎有两种使用模式，选择不当会导致优化结果不可靠。

**Basic 模式**（默认，开发用）：
```python
config = BacktestConfig(initial_capital=10_000_000)
# 固定 1-tick 滑点，总是成交，瞬时换仓，仅开盘时检查保证金
```

**Industrial 模式**（生产级验证用）：
```python
config = BacktestConfig(
    initial_capital=10_000_000,
    volume_adaptive_spread=True,     # 低量→更宽价差
    dynamic_margin=True,             # 交割月阶梯保证金
    time_varying_spread=True,        # 开盘/收盘时段更宽价差
    rollover_window_bars=20,         # 20 bar 渐进换仓
    asymmetric_impact=True,          # 顺势冲击低，逆势冲击高
    detect_locked_limit=True,        # 涨跌停 + 低量→拒绝全部订单
    margin_check_mode="daily",       # 每日结算价检查保证金
    margin_call_grace_bars=3,        # 追保宽限 3 bar
)
```

### 实测 Sharpe 衰减基准

以下为 Basic→Industrial 的真实衰减数据，作为各频率预期参考：

| 策略 | 频率 | Basic Sharpe | Industrial Sharpe | 衰减 | 成交量变化 |
|------|------|:-----------:|:-----------------:|:----:|:----------:|
| v12 | daily | 3.09 | 2.84 | -8.1% | 18→18 |
| v9 | 1h | 2.15 | 1.00 | -53.7% | 261→43 |

**各频率预期衰减范围：**

| 频率 | 预期衰减 | 说明 |
|------|:-------:|------|
| daily | 5-10% | 几乎不受影响 |
| 4h | 10-25% | 需要关注 |
| 1h | 25-55% | 必须在 Industrial 下优化 |
| 5min | > 60% 可能 | 高风险，仅适用 A 级流动性品种 |

### 使用场景

| 场景 | 推荐模式 | 理由 |
|------|:-------:|------|
| 策略开发/快速迭代 | Basic | 速度快 |
| 粗调优化（Coarse） | Basic | 方向对即可 |
| **精调优化（所有频率）** | **Industrial** | **确保参数在真实成本下有效** |
| Portfolio 验证/入选前 | **Industrial** | 必须用真实数字 |
| Portfolio 组合回测 | **Industrial** | 真实 Sharpe |

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
| 1h | ✅ | V6.0 原生支持（不再需要用 60min） |
| daily | ✅ | |
| 4h | 60min x 4 重采样 | `resample_bars(bars, 4)` 或 V6 `resample_freqs` |
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

## BrokerConfig（V6.0 新增）

券商级别配置，模拟真实交易环境中的佣金结构：

```python
from alphaforge.engine.broker import BrokerConfig

broker = BrokerConfig(
    commission_markup=1.1,          # 佣金加成（1.1 = 交易所手续费 × 1.1）
    volume_tiers=[                  # 成交量分级佣金
        (0, 0.00005),              # 0-1000手: 万分之0.5
        (1000, 0.00004),           # 1000手以上: 万分之0.4
    ],
    close_today_override={          # 平今仓手续费覆盖（品种级别）
        "AG": 0.00001,             # AG 平今万分之0.1
        "AU": 0.0,                 # AU 平今免手续费
    },
)

config = BacktestConfig(
    broker=broker,
    # ... 其他参数
)
```

---

## ContractSpecManager V6.0 新字段

V6.0 为 `ContractSpecManager` 新增了更精细的市场微观结构参数：

```python
spec = ContractSpecManager().get("AG")

# V4 已有字段
spec.multiplier          # 合约乘数
spec.tick_size           # 最小变动价位
spec.margin_rate         # 保证金率
spec.commission_open     # 开仓手续费率

# V6 新增字段
spec.impact_exponent     # 市场冲击指数（非线性冲击模型的指数）
spec.spread_elasticity   # 价差弹性（成交量对价差的影响系数）
spec.max_spread_mult     # 最大价差倍数（价差上限 = tick_size × max_spread_mult）
spec.max_open_lots_per_day  # 每日最大开仓手数限制
```

---

## Paper Trading（V6.0 新增）

AlphaForge V6.0 内置模拟盘引擎，支持策略从回测到实盘的过渡验证。

### PaperTradingEngine API

```python
from alphaforge.paper import PaperTradingEngine

engine = PaperTradingEngine(
    strategy=my_strategy,
    symbols=["AG"],
    freq="daily",
    initial_capital=1_000_000,
)
engine.start()  # 开始接收实时行情并执行策略
```

### CLI 命令

```bash
# 启动模拟盘
af paper --strategy strategies/strong_trend/v12.py --symbols AG --freq daily

# 查看模拟盘状态
af paper-status

# 停止模拟盘
af paper-stop
```

### 策略生命周期中的位置

```
回测开发 → 参数优化 → 样本外验证 → **模拟盘 (af paper)** → 小资金实盘 → 正式实盘
```

模拟盘是策略上线前的必要验证步骤。运行 1 个月以上，确认实时表现与回测一致后，方可进入小资金实盘。

---

## Iron Rules（V6.0 — 13 条铁律）

AlphaForge V6.0 定义了 13 条不可违反的回测铁律，确保回测结果的真实性：

| # | 铁律 | 说明 |
|---|------|------|
| 1 | **Signal Delay** | 信号在当前 bar 产生，下一 bar 的 open 执行。不可绕过 |
| 2 | **Tick Snap** | 成交价必须对齐到 tick_size 的整数倍 |
| 3 | **FIFO** | 平仓遵循先进先出：先平昨仓（手续费低），再平今仓（手续费高） |
| 4 | **Margin Check** | 保证金不足时拒绝开仓 |
| 5 | **Locked Limit** | 涨跌停时拒绝成交（V6 新增 `detect_locked_limit` 检测锁仓） |
| 6 | **Partial Fill** | 单笔成交量不超过该 bar 成交量的 10% |
| 7 | **Single Direction** | 同品种同时间只能持有单一方向（不可同时多空） |
| 8 | **Night Session** | 夜盘归属下一个交易日，`trading_day` 已正确处理 |
| 9 | **Forced Liquidation** | 权益低于维持保证金时强制平仓 |
| 10 | **Nonlinear Impact** | 大单市场冲击非线性增长（V6 `asymmetric_impact`） |
| 11 | **Bid-Ask Spread** | 买卖价差模拟，时段/成交量自适应（V6 `volume_adaptive_spread`） |
| 12 | **Rollover Cost** | 换仓成本已包含在模型中（V6 支持 `rollover_window_bars` 渐进换仓） |
| 13 | **Settlement Mark-to-Market** | 每日结算盯市，结算价重新计算保证金和盈亏 |

---

## Common Pitfalls（V6.0 更新）

| 陷阱 | 说明 | 正确做法 |
|------|------|---------|
| 用 `context.current_bar.is_rollover` | 每 bar 创建 namedtuple，慢 100x | 用 `context.is_rollover` |
| `ContractSpecManager()` 在方法内实例化 | 每笔交易读 YAML | 模块级单例 `_SPEC_MANAGER` |
| 手动 reshape 做多周期 | 容易出错，不处理边界 | V6: `resample_freqs` + `context.get_resampled_bars()` |
| 用 `"60min"` 代替 `"1h"` | V6 之前的遗留写法 | V6 原生支持 `"1h"` |
| 不启用 `volume_adaptive_spread` | 回测滑点不真实 | 生产级回测必须启用 |
| 单 bar 换仓 | 大仓位换仓冲击大 | V6: `rollover_window_bars=20` 渐进换仓 |
| 忽略涨跌停锁仓 | 持仓被锁无法平仓 | V6: `detect_locked_limit=True` |
| 不做模拟盘就上实盘 | 回测与实盘偏差大 | V6: `af paper` 模拟盘验证 |

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
