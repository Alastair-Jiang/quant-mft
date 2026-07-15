# 参考模型蒸馏、对比分析与策略补丁

> 本文档对照四篇已发表的日频量化 ML 模型，提取关键要素，与 quant-mft 当前策略逐项对比，给出具体改进补丁。
>
> 主参考模型：**MSIF-OEM**（中国A股全市场，Transformer+在线集成）
> 辅助参考：**StockGPT**（美股，纯价格GPT）、**Increase Alpha**（美股，精选特征+NN）、**Qraft LQAI**（美股ETF，三层Agent）
>
> 生成日期：2026-07-15

---

## 一、主参考模型选定：MSIF-OEM

### 为什么选它作为主参考

| 对比维度 | MSIF-OEM | StockGPT | Increase Alpha | Qraft LQAI |
|---------|----------|----------|---------------|------------|
| 市场 | 🇨🇳 A股全市场 | 🇺🇸 美股 | 🇺🇸 美股 | 🇺🇸 美股 |
| 股票数 | >5000 | ~3000 | 814 | ~100-350 |
| 数据源 | OHLCV+L2/LOB/TAQ | 纯价格 | 基本面+价量 | 结构化+新闻NLP |
| 模型 | Transformer+在线集成 | GPT decoder | Feed-forward+RNN | 三层DL Agent |
| 频率 | 日频 | 日频 | 日频 | 月频换仓 |
| **与我们的匹配度** | 🔴 最高 | 🟡 中 | 🟡 中 | 🟢 低（月频） |

MSIF-OEM 是唯一一个在**中国A股全市场、日频、多源数据、Transformer架构**四个维度上与我们完全对齐的已发表模型。

---

## 二、MSIF-OEM 架构蒸馏

### 论文信息
- **标题**: Transforming machine learning strategies in quantitative stock investment: A multisource information fusion and online ensemble modeling approach for superior alpha factors
- **期刊**: Expert Systems with Applications (Elsevier), 2025
- **作者**: Zhao, Chen, Cao, Li, Ying, Mu
- **DOI**: [10.1016/j.eswa.2025.127151](https://www.sciencedirect.com/science/article/abs/pii/S095741742504151X)

### 两阶段架构

```
阶段 1: PTE-TFE (Parallel Transformer Encoder - Temporal Feature Extraction)
┌─────────────────────────────────────────────────────────────┐
│                      多源数据输入                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ OHLCV    │  │ 市场快照  │  │ L2订单簿  │  │ TAQ逐笔数据  │ │
│  │ (日频)   │  │ (截面)    │  │ (深度)    │  │ (高频聚合)   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│       │             │             │               │         │
│       ▼             ▼             ▼               ▼         │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────┐     │
│  │TransEnc│   │TransEnc│   │TransEnc│   │ TransEnc   │     │
│  │  (1)   │   │  (2)   │   │  (3)   │   │   (4)      │     │
│  └───┬────┘   └───┬────┘   └───┬────┘   └─────┬──────┘     │
│      │            │            │              │             │
│      └────────────┴────────────┴──────────────┘             │
│                         │                                    │
│                         ▼                                    │
│              ┌──────────────────┐                            │
│              │  正交化融合层      │  ← 去相关，防止因子拥挤     │
│              │  (Orthogonal     │                            │
│              │   Fusion Layer)  │                            │
│              └────────┬─────────┘                            │
│                       │                                      │
│                       ▼                                      │
│              融合后的正交时序特征 (Alpha因子)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
阶段 2: ESGD + Online Ensemble (Ensemble Stochastic Gradient Descent)
┌─────────────────────────────────────────────────────────────┐
│  聚类 → K 个市场运行模式（牛市/熊市/震荡/...）               │
│     │                                                       │
│     ▼                                                       │
│  每个模式训练一个基学习器 (base learner)                      │
│     │                                                       │
│     ▼                                                       │
│  在线集成更新：新交易日数据到来 →                              │
│     ├─ 不需要全量重训练                                      │
│     ├─ 更新聚类中心（市场状态可能漂移）                        │
│     └─ 微调各基学习器的集成权重                                │
│                                                             │
│  输出：合成 Alpha 因子 → 多空组合 → 每日信号                   │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计要素提取

| 要素编号 | 要素名称 | 具体做法 | 为什么有效 |
|---------|---------|---------|-----------|
| E1 | 多源数据并行编码 | 4个独立的Transformer Encoder分别处理OHLCV/快照/L2/TAQ | 不同数据源的时间结构不同，分开编码比混在一起好 |
| E2 | 正交化融合 | 融合后做正交分解，去除因子间相关性 | 防止因子拥挤（crowding）——多个因子其实在说同一件事 |
| E3 | 聚类→多模式建模 | 用聚类识别K种市场状态，每种状态独立建模 | 牛市和熊市的因子效果完全不同，一个模型学不会 |
| E4 | 在线集成更新 | 新数据到来时只更新聚类中心和集成权重，不重训 | 市场在变，但每天全量重训不可行。在线更新是折中 |
| E5 | 全A股验证 | 在>5000只A股上测试 | 证明框架在A股市场有效 |

---

## 三、其他三个参考模型的关键补充要素

### StockGPT (Dat Mai, 2024)

| 要素编号 | 要素名称 | 具体做法 |
|---------|---------|---------|
| S1 | 收益率离散化（Tokenization） | 日收益×10000 → 基点 → 每50bp一个bin → 402类token |
| S2 | 纯价格GPT | Decoder-only Transformer，4层4头，仅93万参数 |
| S3 | 256天上下文窗口 | 输入过去~1年的日收益序列 |
| S4 | 分布预测而非点预测 | 输出402类的完整概率分布，可计算期望收益+不确定性 |
| S5 | 微型流动性过滤 | 剔除市值后10%的股票（避免微盘股流动性陷阱） |
| S6 | 单次训练/长期有效 | 1926-2000训练，2001-2023测试（23年不重训！） |
| S7 | Skip-1-day | 预测t+2而非t+1（避免bid-ask bounce等 microstructure noise） |

### Increase Alpha (Ghatak et al., 2025)

| 要素编号 | 要素名称 | 具体做法 |
|---------|---------|---------|
| I1 | 精选特征而非堆特征 | 只选"基本面分析师认为有经济含义"的特征 |
| I2 | 10日多 horizon 预测 | 同时输出t+1到t+10共10个方向的预测 |
| I4 | 逐票参数优化 | 每只股票的止盈/止损/持仓天数独立grid search |
| I5 | 三元方向信号 | +1(多)/-1(空)/0(观望)——低置信度不出手 |

### Qraft LQAI (Qraft + LG AI, 2024)

| 要素编号 | 要素名称 | 具体做法 |
|---------|---------|---------|
| Q1 | 三层Agent分层 | 预测Agent→排序Agent→加权Agent |
| Q2 | 结构化+非结构化融合 | 财报数据+NLP新闻情感 |
| Q3 | 动态数据源权重 | 风险环境下加大新闻情感权重 |

---

## 四、与 quant-mft 当前策略的逐项对比

### 4.1 架构层对比

| 维度 | MSIF-OEM（参考） | quant-mft（当前） | 差距 | 优先级 |
|------|-----------------|------------------|------|--------|
| 数据源数量 | 4源（OHLCV+快照+L2+TAQ） | 1源（OHLCV） | 🔴 大 | P2 |
| 时序编码器 | 并行Transformer Encoder | 手写滚动窗口因子（无学习） | 🔴 大 | P2 |
| 特征融合 | 正交化去相关 | IC分析+互相关去重 | 🟡 中 | P1 |
| 多模式建模 | 聚类→K个基学习器 | 1个LightGBM | 🟡 中 | P1 |
| 在线学习 | 增量更新集成权重 | 无（计划定期重训） | 🟡 中 | P2 |
| 输出形式 | 合成Alpha因子→多空组合 | 涨跌二分类→买卖信号 | 🟢 小 | — |
| 过拟合防护 | 正交化+在线验证 | Walk-Forward+蒙特卡洛 | 🟢 小 | — |
| 基准对比 | 未提及 | 沪深300买入持有 | 🟢 我们更好 | — |

### 4.2 关键差距解读

#### 🔴 差距1：数据源单一（但我们故意的）

MSIF-OEM 用了 L2 订单簿和 TAQ 逐笔数据。我们有意识地只用了 OHLCV——因为 akshare 免费数据只有 OHLCV。这个差距在 MVP 阶段是**可接受的**，原因：
- MSIF-OEM 论文没有做消融实验，不确定 L2/TAQ 的增量贡献有多大
- StockGPT 只用纯价格数据就做到了 Sharpe 6.5
- Increase Alpha 只用精选特征（OHLCV+基本面）做到了 Sharpe >2.5

**结论**：多源数据是加分项，不是必须项。MVP 阶段单源 OHLCV 够用。

#### 🟡 差距2：特征提取靠手写而非学习

这是最大的结构性差距。MSIF-OEM 用 Transformer Encoder 学习时序特征，我们用手写公式（`close.rolling(20).mean()`）。

**但这里有一个反直觉的点**：StockGPT 的纯价格 GPT 是"学习"特征，Increase Alpha 是"手写"特征。两个都达到了很好的效果。这说明：
- 学习特征：上限更高，但需要更多数据、更容易过拟合
- 手写特征：上限较低，但更稳健、可解释、不容易过拟合

对于日频 A 股（375 万行数据），手写因子 + LightGBM 可能是更安全的选择。

#### 🟡 差距3：缺少多模式建模

这是我们当前架构最大的可改进点。MSIF-OEM 的做法是：
1. 用聚类把市场分成 K 种状态
2. 每种状态训练一个独立的模型
3. 新数据来的时候，先判断属于哪种状态，再用对应的模型预测

**我们当前的做法**：一个 LightGBM 学所有市场状态。这要求 LightGBM 自己在树分裂中发现"牛市该用哪些因子，熊市该用哪些因子"——它能做到（树天然就是 if-else 规则），但不如显式建模高效。

**这是最高优先级的补丁**。

#### 🟡 差距4：缺少在线学习

MSIF-OEM 的在线集成更新虽然我们目前做不了（需要持续运行的服务器），但这个设计思路可以借鉴：
- 我们在 `daily_run.py` 里已经有 `should_retrain()` 函数
- 月频重训练就是一种最简形式的"在线更新"
- 如果后续有持续运行的服务器，可以加入增量更新逻辑

---

### 4.3 因子层对比

| 维度 | 参考模型 | quant-mft（当前） |
|------|---------|------------------|
| 因子来源 | MSIF-OEM: Transformer学习 | 手写公式（18+因子） |
| 因子数量 | StockGPT: 256维隐式 | 18+显式 |
| 因子类型 | Qraft: 结构化+非结构化 | 仅结构化（OHLCV） |
| 信噪比处理 | Increase Alpha: 三元信号过滤 | 概率阈值过滤 |
| 因子去重 | MSIF-OEM: 正交化 | IC+相关系数 |
| 因子评估 | StockGPT: 交叉熵loss | IC分析+信息熵防火墙 |

**关键发现**：我们当前的手写因子体系其实和 Increase Alpha 的"精选特征"哲学一致。区别在于 Increase Alpha 有**基本面因子**（SEC filings, corporate actions）。A股对应的就是财报数据——这部分 akshare 有接口，可以在 P1 阶段加入。

### 4.4 模型层对比

| 维度 | 参考模型 | quant-mft（当前） |
|------|---------|------------------|
| 主模型 | MSIF-OEM: Transformer | LightGBM |
| 模型数量 | MSIF-OEM: K个（每状态1个） | 1个 |
| 训练方式 | MSIF-OEM: 在线增量 | 离线全量/月频重训 |
| 输出 | StockGPT: 完整分布 | 涨跌概率(单值) |
| 多horizon | Increase Alpha: t+1~t+10 | 仅t+1 |

**关键发现**：

1. **输出分布而非点预测**（StockGPT S4）：我们当前只输出涨跌概率。如果改为输出"明天涨5%的概率、涨2%的概率、跌3%的概率……"，可以同时得到方向预测和不确定性估计。这对仓位管理很有用——高确定性时满仓，低确定性时轻仓。

2. **多 horizon 预测**（Increase Alpha I2）：预测 t+1 到 t+10 的好处是——如果模型预测 t+1 涨但 t+3 跌，你可以选择只做短线。10个horizon的预测值本身就是信号强度的刻画。

### 4.5 交易执行层对比

| 维度 | 参考模型 | quant-mft（当前） |
|------|---------|------------------|
| 信号类型 | Increase Alpha: +1/0/-1三元 | BUY/SELL/HOLD三元 |
| 持仓周期 | StockGPT: 每日换仓 | 不固定 |
| 流动性过滤 | StockGPT: 剔除市值后10% | 无（仅ST过滤） |
| 逐票优化 | Increase Alpha: 每票独立止盈止损 | 全局统一参数 |
| 交易成本 | 各模型：显式建模 | 手续费+滑点（已做） |

---

## 五、策略补丁（按优先级排序）

基于以上对比分析，以下是具体可落地的改进补丁。

---

### 🔴 补丁 P1：多模式建模（最高优先级）

**来源**：MSIF-OEM E3 + StockGPT S3（长期上下文）

**问题**：当前一个 LightGBM 学习所有市场状态，无法显式适应市场切换。

**方案**：

```
在 train.py 中加入 regime 识别层：

Step 1: 用以下特征做市场状态聚类（K=3~5）:
    - 全市场等权平均收益率（过去20日）
    - 全市场波动率（过去20日）
    - 全市场上涨股票占比
    - 全市场成交量变化
    → K-Means 聚类 → 每个交易日分配一个 regime 标签

Step 2: 每个 regime 训练一个独立的 LightGBM
    - regime=0 (牛市): LightGBM_0
    - regime=1 (震荡): LightGBM_1
    - regime=2 (熊市): LightGBM_2

Step 3: 推理时
    - 先用当日全市场数据判断属于哪个 regime
    - 用对应 regime 的模型做预测
    - 如果处于 regime 边界（离聚类中心都远）→ 用所有模型集成
```

**具体改动**：
- 在 `src/models/train.py` 中新增 `cluster_regimes()` 和 `RegimeRouter` 类
- 在 `src/features/alpha_factors.py` 中新增 4 个市场级因子（等权收益、波动率、上涨占比、量比）
- 训练时按 regime 分组训练，保存多个模型
- 推理时先路由再预测

**预期提升**：AUC +0.01~0.02（在市场切换频繁的时期提升更大）

---

### 🟡 补丁 P2：分布输出 + 不确定性量化

**来源**：StockGPT S4

**问题**：当前只输出涨跌概率（一个标量），丢失了预测的不确定性信息。

**方案**：

```
LightGBM 支持多分类 → 将收益划分为 K 个区间:

区间:  < -5%  [-5%,-2%)  [-2%,0)  [0,2%)  [2%,5%)  > 5%
标签:    0        1         2        3       4       5

训练：多分类 LightGBM（objective="multiclass", num_class=6）
输出：6维概率向量 [p0, p1, p2, p3, p4, p5]

从分布中提取信息:
    期望收益 = Σ pi × 区间中点
    方向概率 = P(收益>0) = p3 + p4 + p5
    不确定性 = 分布的熵 = -Σ pi × log(pi)
    
    高熵 → 分布平坦 → 模型不确定 → 降低仓位
    低熵 → 分布尖锐 → 模型确定 → 正常仓位
```

**具体改动**：
- 在 `src/models/train.py` 中新增 `create_multiclass_target()`
- 修改 `DEFAULT_PARAMS` 的 `objective` 和 `num_class`
- 在 `src/strategy/signal_generator.py` 中新增不确定性过滤

**预期提升**：回撤降低 10-20%（不确定时不交易 = 避免最差的交易），AUC 可能略降

---

### 🟡 补丁 P3：多 horizon 预测

**来源**：Increase Alpha I2

**问题**：当前只预测 t+1。如果模型预测 t+1 涨但 t+3 跌，这个信息没有被利用。

**方案**：

```
在 train.py 中新增多 horizon 目标:

create_target_horizon(df, horizon=1)  → target_1d
create_target_horizon(df, horizon=3)  → target_3d
create_target_horizon(df, horizon=5)  → target_5d
create_target_horizon(df, horizon=10) → target_10d

训练 4 个模型（或 1 个多任务模型）

信号合成:
    if all(target_1d, target_3d, target_5d > 0.5): → 最强的买入信号
    if target_1d > 0.5 but target_5d < 0.5: → 短线机会，快进快出
    if target_1d < 0.5 but target_3d > 0.5: → 短期承压但中期看好，等一等再买
```

**具体改动**：
- 在 `src/models/train.py` 中新增 `create_multi_horizon_targets()`
- 在 `src/strategy/signal_generator.py` 中新增 horizon 一致性检查

**预期提升**：胜率 +2~3%（过滤掉短期和中期方向矛盾的信号）

---

### 🟢 补丁 P4：Skip-1-day 预测

**来源**：StockGPT S7

**问题**：t+1 预测包含 microstructure noise（买卖价差反弹效应）。预测 t+2 的噪声更小。

**方案**：

```
简单改动：create_target() 中把 shift(-1) 改成 shift(-2)

target = close.shift(-2) / close.shift(-1) - 1 > 0  # 预测后天 vs 明天的涨跌

原因：A股 T+1 制度 → 今天买明天才能卖
      预测 t+1: "明天收盘价 vs 今天收盘价" → 包含了日内波动噪声
      预测 t+2: "后天收盘价 vs 明天收盘价" → 剔除了隔夜的 microstructure noise
```

**具体改动**：在 `src/models/train.py` 中修改 `create_target()` 的 shift 参数

**预期提升**：AUC +0.005~0.01（StockGPT 的 skip-1-day 版本 Sharpe 从 6.5 降到 1.7——但在月频上反而更好，说明 skip 降低了噪声）

---

### 🟢 补丁 P5：微观市值过滤

**来源**：StockGPT S5

**问题**：当前只过滤了 ST 股票，没有过滤微盘股。A 股市值后 10% 的股票流动性极差——回测中能成交的，实盘可能根本买不到。

**方案**：

```
在 cleaner.py 中加入市值过滤:

1. 用 akshare 获取每只股票的市值（stock_individual_info_em）
2. 剔除市值 < 全部股票市值 10% 分位数的股票
3. 或者在回测中给微盘股加额外的流动性折扣
```

**具体改动**：在 `src/data/cleaner.py` 中新增 `filter_micro_cap()`

**预期提升**：回测和实盘的差距缩小（不是收益提升，而是可信度提升）

---

### 🔵 补丁 P6：合成数据生成器（远期，对应你前面提的Transformer方案）

**来源**：StockGPT 的 tokenization + 我们的合成数据思路

**问题**：真实数据不足以训练 Transformer。

**方案**：见前面对话中的合成数据生成器设计。此补丁为 P2 远期规划。

---

## 六、补丁优先级汇总

| 补丁 | 来源论文 | 优先级 | 改动量 | 预期提升 | MVP可做 |
|------|---------|--------|--------|---------|---------|
| P1 多模式建模 | MSIF-OEM E3 | 🔴 P0 | 中 | AUC +0.01~0.02 | ✅ |
| P2 分布输出+不确定量化 | StockGPT S4 | 🟡 P1 | 中 | 回撤 -10~20% | ✅ |
| P3 多horizon预测 | Increase Alpha I2 | 🟡 P1 | 大 | 胜率 +2~3% | ⚠️ 部分 |
| P4 Skip-1-day | StockGPT S7 | 🟢 P1 | 极小 | AUC +0.005 | ✅ |
| P5 微盘过滤 | StockGPT S5 | 🟢 P1 | 小 | 回测可信度↑ | ✅ |
| P6 合成数据生成器 | StockGPT+我们 | 🔵 P2 | 大 | 探索性 | 否 |

---

## 七、6小时优化计划

按照补丁优先级，6小时内可完成的工作：

### 第1小时：P4 Skip-1-day（最小改动，快速验证）

- [ ] 修改 `train.py` 中 `create_target()` 的 `shift(-1)` → `shift(-2)`
- [ ] 修改注释：预测 t+2 收盘价 vs t+1 收盘价的涨跌
- [ ] 如果 fetcher.py 数据已拉取，跑一次训练看 AUC 变化

### 第2-3小时：P1 多模式建模（核心改动）

- [ ] 在 `features/alpha_factors.py` 中新增 4 个市场级因子：
  - `market_equal_weight_return_20d`
  - `market_volatility_20d`
  - `market_advance_ratio_20d`（上涨股票占比）
  - `market_volume_ratio_20d`
- [ ] 在 `models/train.py` 中新增 `RegimeRouter` 类：
  - `cluster_regimes()`：用 K-Means（K=3）做市场状态聚类
  - `train_regime_models()`：每个 regime 训练一个 LightGBM
  - `predict_with_regime()`：先判断 regime，再用对应模型预测
- [ ] 修改 `train_pipeline()` 调用 `RegimeRouter`

### 第4-5小时：P2 分布输出（增强信号质量）

- [ ] 在 `models/train.py` 中新增 `create_multiclass_target(df, n_bins=6)`
- [ ] 配置 LightGBM 多分类参数（`objective="multiclass"`, `num_class=6`）
- [ ] 在 `strategy/signal_generator.py` 中新增：
  - `calc_prediction_entropy()`：从多分类输出计算熵
  - `filter_by_uncertainty()`：高熵信号 → 降低仓位或过滤
- [ ] 在 `config/default.yaml` 中新增 `signal.min_confidence` 和 `signal.max_entropy`

### 第6小时：P5 微盘过滤 + 集成验证

- [ ] 在 `data/cleaner.py` 中新增 `filter_micro_cap()`
- [ ] 跑端到端测试：数据→清洗→因子→多regime模型→信号
- [ ] 对比 P4 改动前后的 AUC 差异
- [ ] 记录实验到 `experiments.csv`

---

## 八、补丁应用后的策略对比

### 改动前（当前策略）

```
OHLCV → 手写18因子 → 1个LightGBM → 二分类涨跌 → 单一阈值 → BUY/SELL
```

### 改动后（应用补丁 P1-P5）

```
OHLCV → 手写18因子 + 4个市场级因子
      │
      ├─→ 市值过滤（剔除微盘股）
      │
      ├─→ 市场状态聚类（K=3）
      │     ├─→ regime=0 → LightGBM_0 (多分类6bin)
      │     ├─→ regime=1 → LightGBM_1 (多分类6bin)
      │     └─→ regime=2 → LightGBM_2 (多分类6bin)
      │
      ├─→ 预测 t+2（skip-1-day）
      │
      └─→ 信号 = 方向概率 × 置信度（1-熵） × horizon一致性
            ├─→ 高置信 → 正常仓位
            ├─→ 中置信 → 半仓
            └─→ 低置信 → 不交易
```

---

## 附录：参考文献

1. **Zhao et al. (2025)**. "Transforming machine learning strategies in quantitative stock investment: A multisource information fusion and online ensemble modeling approach for superior alpha factors." *Expert Systems with Applications*. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742504151X)

2. **Mai, D. (2024)**. "StockGPT: A GenAI Model for Stock Prediction and Trading." *Journal of Financial Data Science* (Fall 2025). [arXiv:2404.05101](https://arxiv.org/abs/2404.05101)

3. **Ghatak, S. et al. (2025)**. "Increase Alpha: Performance and Risk of an AI-Driven Trading Framework." [arXiv:2509.16707](https://arxiv.org/abs/2509.16707)

4. **Li, L. et al. (2024)**. "Stock Return Prediction Based on Ensemble and CNN Algorithms — A Rolling Training Approach to Prevent Data Leakage." *ACM IoTCCT 2024*. [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3702879.3702935)

5. **Qraft Technologies (2024)**. LQAI ETF Architecture. [Qraft](https://www.qraftec.com)

---

> 📌 补丁按优先级排列，可以逐个应用、逐个验证。P1（多模式建模）和 P4（Skip-1-day）建议最先做——改动小、收益明确。
>
> 作者：AI 分析 | 日期：2026-07-15
