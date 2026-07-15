# 数据字典 (Data Dictionary)

> 本文档定义项目中所有数据文件的字段含义、数据类型、来源和更新频率。
> 作用：让任何人都能快速理解数据内容，避免"这一列是什么"的反复沟通。

---

## 1. 原始日线数据 — `data/a_stock_daily.csv`

**来源：** akshare `stock_zh_a_hist()`（底层调东方财富 API）
**更新频率：** 每日收盘后（15:30 cron 触发）
**复权方式：** 前复权（qfq）

| 字段名 | 中文名 | 类型 | 说明 |
|--------|--------|------|------|
| `date` | 交易日期 | str (YYYYMMDD) | 交易日，非自然日 |
| `open` | 开盘价 | float | 前复权后 |
| `high` | 最高价 | float | 前复权后 |
| `low` | 最低价 | float | 前复权后 |
| `close` | 收盘价 | float | 前复权后 |
| `volume` | 成交量 | int | 单位：股（不是手！1手=100股） |
| `amount` | 成交额 | float | 单位：元 |
| `code` | 股票代码 | str | 6位数字，如 "000001" |
| `name` | 股票名称 | str | 如 "平安银行" |

**注意事项：**
- ⚠️ akshare 返回的原始列名是中文（日期/开盘/收盘…），fetcher.py 已自动重命名为英文
- ⚠️ 停牌日没有记录（不是 NaN，是直接没有这一行）
- 包含沪深京三地交易所全部股票

---

## 2. 清洗后数据 — `data/a_stock_daily_clean.parquet`

**来源：** `src/data/cleaner.py` 处理原始 CSV 后输出
**格式：** Parquet（压缩率高、读写快、保留列类型）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| （继承原始数据全部字段） | — | — |
| `is_suspended` | bool | 是否为停牌日（通过连续交易日历推断） |
| `is_st` | bool | 是否为 ST 股票 |
| `turnover_rate` | float | 换手率（从 akshare 原始数据提取，单位 %） |
| `pct_change` | float | 涨跌幅（从 akshare 原始数据提取，单位 %） |

**清洗规则：**
1. 极值处理：MAD (Median Absolute Deviation) 方法，阈值 5 倍 MAD
2. 缺失值：停牌日标记 `is_suspended=True`（不做前向填充）
3. 复权检查：验证 close 与 open/high/low 的逻辑一致性
4. ST 标记：代码含 "ST" 或 "\\*ST" 的标记 `is_st=True`

---

## 3. 特征数据 — `data/features.parquet`

**来源：** `src/features/alpha_factors.py` 从清洗后数据衍生
**更新频率：** 跟随每日管线

| 字段名 | 中文名 | 类型 | 公式/说明 |
|--------|--------|------|-----------|
| `ret_1d` | 1日收益率 | float | close / close.shift(1) - 1 |
| `ret_5d` | 5日收益率 | float | close / close.shift(5) - 1 |
| `ret_10d` | 10日收益率 | float | close / close.shift(10) - 1 |
| `ret_20d` | 20日收益率 | float | close / close.shift(20) - 1 |
| `ma_5` | 5日均线 | float | close.rolling(5).mean() |
| `ma_10` | 10日均线 | float | close.rolling(10).mean() |
| `ma_20` | 20日均线 | float | close.rolling(20).mean() |
| `ma_60` | 60日均线 | float | close.rolling(60).mean() |
| `ma_dev_5` | 5日均线偏离 | float | close / ma_5 - 1 |
| `ma_dev_10` | 10日均线偏离 | float | close / ma_10 - 1 |
| `ma_dev_20` | 20日均线偏离 | float | close / ma_20 - 1 |
| `ma_dev_60` | 60日均线偏离 | float | close / ma_60 - 1 |
| `rsi_14` | RSI(14) | float | 相对强弱指标，Wilder 算法 |
| `volatility_20` | 20日波动率 | float | ret_1d.rolling(20).std() * sqrt(252) |
| `atr_14` | ATR(14) | float | 平均真实波幅 |
| `bb_width_20` | 布林带宽度 | float | (布林上轨 - 布林下轨) / ma_20 |
| `volume_ratio_5` | 5日量比 | float | volume / volume.rolling(5).mean() |
| `volume_ratio_20` | 20日量比 | float | volume / volume.rolling(20).mean() |
| `macd` | MACD | float | EMA(12) - EMA(26) |
| `macd_signal` | MACD 信号线 | float | MACD 的 EMA(9) |
| `macd_hist` | MACD 柱 | float | MACD - MACD_signal |
| `turnover_change_5` | 换手率变化 | float | turnover / turnover.shift(5) - 1 |

> 完整因子列表和详细计算公式见 [因子目录](factor_catalog.md)

---

## 4. 预测信号 — `data/signals.csv`

**来源：** `src/strategy/signal_generator.py`
**更新频率：** 每日

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `date` | str | 信号生成日期 |
| `code` | str | 股票代码 |
| `name` | str | 股票名称 |
| `pred_prob` | float | LightGBM 预测次日上涨的概率 (0~1) |
| `pred_label` | int | 预测标签（0=跌，1=涨） |
| `signal` | int | 交易信号（-1=卖出，0=持有，1=买入） |
| `confidence` | str | 置信度等级（high/medium/low） |

---

## 5. 回测交易明细 — `data/backtest_trades.csv`

**来源：** `src/backtest/engine.py`
**更新频率：** 每次回测运行

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `trade_id` | int | 交易编号 |
| `code` | str | 股票代码 |
| `name` | str | 股票名称 |
| `buy_date` | str | 买入日期 |
| `buy_price` | float | 买入价（含滑点） |
| `sell_date` | str | 卖出日期 |
| `sell_price` | float | 卖出价（含滑点） |
| `shares` | int | 交易股数 |
| `cost` | float | 总成本（佣金+印花税+滑点损失） |
| `return` | float | 单笔收益率 |
| `holding_days` | int | 持有天数 |
| `exit_reason` | str | 出场原因（signal/take_profit/stop_loss） |

---

## 6. 实验记录 — `data/experiments.csv`

**来源：** `src/models/experiment.py`
**更新频率：** 每次模型训练

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `exp_id` | str | 实验编号（时间戳） |
| `train_start` | str | 训练集起始日期 |
| `train_end` | str | 训练集结束日期 |
| `test_start` | str | 测试集起始日期 |
| `test_end` | str | 测试集结束日期 |
| `features` | str | 使用的特征列表（逗号分隔） |
| `n_features` | int | 特征数量 |
| `model_params` | str | 模型超参（JSON 格式） |
| `accuracy` | float | 测试集准确率 |
| `auc` | float | 测试集 AUC |
| `sharpe` | float | 回测夏普比率 |
| `max_drawdown` | float | 回测最大回撤 |
| `notes` | str | 备注 |
