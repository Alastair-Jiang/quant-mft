# 因子目录 (Factor Catalog)

> 本文档列出所有 Alpha 因子的定义、计算公式和理论依据。
> 作用：因子字典 + 避免同一个因子在不同地方被重复定义（或不一致定义）。

---

## 因子分类概览

| 类别 | 数量 | 描述 |
|------|------|------|
| 收益类 (Return) | 4 | 不同时间窗口的历史收益率 |
| 均线偏离类 (MA Deviation) | 4 | 价格相对于均线的偏离程度 |
| 动量/趋势类 (Momentum) | 3 | MACD 及其衍生指标 |
| 波动率类 (Volatility) | 3 | 历史波动率、ATR、布林带宽度 |
| 成交量类 (Volume) | 3 | 成交量变化、量比、换手率变化 |
| 技术指标类 (Technical) | 1 | RSI 相对强弱 |
| **合计** | **18+** | 目标 ≥ 20 个 |

---

## 1. 收益类因子 (Return Factors)

### ret_1d — 1日收益率
```
ret_1d = close(t) / close(t-1) - 1
```
**逻辑：** 短期动量——今天涨的股票明天可能继续涨（或反转，模型自己学）

### ret_5d — 5日收益率
```
ret_5d = close(t) / close(t-5) - 1
```
**逻辑：** 周度动量，捕捉一周级别的趋势

### ret_10d — 10日收益率
```
ret_10d = close(t) / close(t-10) - 1
```
**逻辑：** 双周动量

### ret_20d — 20日收益率
```
ret_20d = close(t) / close(t-20) - 1
```
**逻辑：** 月度动量，约等于过去一个月的表现

---

## 2. 均线偏离类因子 (Moving Average Deviation)

### ma_dev_5 / ma_dev_10 / ma_dev_20 / ma_dev_60
```
ma_N = close.rolling(N).mean()
ma_dev_N = close / ma_N - 1
```
**逻辑：** 价格偏离均线的程度。
- 正值 = 价格在均线上方（短期强势）
- 负值 = 价格在均线下方（短期弱势）
- 均线回归是 A 股最常见的 alpha 来源之一

**选 N 的依据：**
- 5日 = 一周交易日的短期趋势
- 10日 = 双周
- 20日 = 一个月（约等于月线）
- 60日 = 一个季度（约等于季线，机构常用）

---

## 3. 动量/趋势类因子 (Momentum & Trend)

### macd — MACD 快慢线差值
```
ema_12 = close.ewm(span=12).mean()
ema_26 = close.ewm(span=26).mean()
macd = ema_12 - ema_26
```
**逻辑：** 短周期 EMA 与长周期 EMA 的差值。正值 = 短期趋势向上

### macd_signal — MACD 信号线
```
macd_signal = macd.ewm(span=9).mean()
```
**逻辑：** MACD 的平滑版本，用于判断 MACD 的方向

### macd_hist — MACD 柱
```
macd_hist = macd - macd_signal
```
**逻辑：** MACD 与其信号线的差值。正值扩大 = 趋势加速

---

## 4. 波动率类因子 (Volatility)

### volatility_20 — 20日年化波动率
```
volatility_20 = ret_1d.rolling(20).std() * sqrt(252)
```
**逻辑：** 衡量过去一个月的价格波动剧烈程度。
- 高波动 = 风险大但机会也可能大
- 低波动 = 横盘整理中

### atr_14 — 平均真实波幅 (14日)
```
tr = max(high-low, abs(high-close_prev), abs(low-close_prev))
atr_14 = tr.rolling(14).mean()
```
**逻辑：** 考虑了跳空缺口的真实波动范围，比简单的高低差价更准确。
用于衡量波动性 + 后续止损位计算

### bb_width_20 — 布林带宽度 (20日)
```
bb_mid = close.rolling(20).mean()
bb_upper = bb_mid + 2 * close.rolling(20).std()
bb_lower = bb_mid - 2 * close.rolling(20).std()
bb_width_20 = (bb_upper - bb_lower) / bb_mid
```
**逻辑：** 布林带宽相对于价格的比例。
- 宽度扩大 = 波动加剧，可能突破
- 宽度收窄 = 波动收缩，可能变盘

---

## 5. 成交量类因子 (Volume)

### volume_ratio_5 / volume_ratio_20
```
volume_ratio_N = volume(t) / volume.rolling(N).mean()
```
**逻辑：** 今日成交量相对于过去 N 日平均成交量的倍数。
- > 1.5 = 放量（市场关注度提升）
- < 0.5 = 缩量（市场冷淡）
- 放量上涨 vs 放量下跌的含义完全不同，模型会自己学

### turnover_change_5 — 换手率变化
```
turnover_change_5 = turnover(t) / turnover(t-5) - 1
```
**逻辑：** 换手率反映筹码交换速度。换手率突然升高 = 有大资金进出

---

## 6. 技术指标类 (Technical Indicators)

### rsi_14 — 相对强弱指标 (14日)
```
avg_gain = max(ret_1d, 0).rolling(14).mean()
avg_loss = abs(min(ret_1d, 0)).rolling(14).mean()
rs = avg_gain / avg_loss
rsi_14 = 100 - 100 / (1 + rs)
```
**逻辑：** 衡量近期涨跌力量的对比。
- RSI > 70 = 超买（可能回调）
- RSI < 30 = 超卖（可能反弹）
- 用的是 Wilder 平滑算法，不是简单移动平均

---

## 因子评估标准

每个因子在加入模型前需要过以下检查：

| 检查项 | 标准 | 不通过则 |
|--------|------|---------|
| IC 绝对值 | \|IC\| > 0.02 | 考虑剔除 |
| IC 稳定性 | IC 标准差 < IC 均值 | 考虑剔除 |
| 互相关性 | 与其他因子相关系数 < 0.8 | 二选一保留 |
| 信息熵 | > 随机噪声的信息熵 | 信息量不足则剔除 |
| 缺失率 | < 30% | 缺失太多则剔除 |

---

## 因子命名规范

- 收益类：`ret_{N}d`
- 均线偏离：`ma_dev_{N}`
- 波动率：`volatility_{N}` / `atr_{N}`
- 量比：`volume_ratio_{N}`
- 技术指标：小写缩写 `rsi_14`, `macd`

> ⚠️ 禁止在代码中重复定义因子的计算公式。所有因子统一在 `src/features/alpha_factors.py` 中实现，其他模块只调用不重写。
