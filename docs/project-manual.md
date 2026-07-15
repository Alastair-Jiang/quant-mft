# quant-mft 项目说明书

> 本文档是 quant-mft 量化交易系统的完整说明书。
> 面向读者：项目开发者（蒋东旭）、协作者（李桂聿）、以及任何想理解这个项目的人。
>
> **阅读顺序建议**：先读本文档（了解全貌）→ 再读 `architecture.md`（理解架构）→ 然后逐个模块深入。

---

## 目录

- [一、项目是什么](#一项目是什么)
- [二、项目文件全景图](#二项目文件全景图)
- [三、根目录文件详解](#三根目录文件详解)
- [四、配置文件详解](#四配置文件详解)
- [五、文档层详解](#五文档层详解)
- [六、源码层详解](#六源码层详解)
- [七、测试层详解](#七测试层详解)
- [八、数据流向](#八数据流向)
- [九、开发顺序建议](#九开发顺序建议)

---

## 一、项目是什么

**quant-mft** 是一个中频（日线级别）机器学习量化交易系统。

**一句话描述**：每天收盘后，自动从东方财富拉取 A 股全市场数据 → 用 LightGBM 模型预测每只股票明天是涨还是跌 → 回测评估策略表现 → 通过 Telegram 推送交易信号。

**目标用户**：东北财经大学金融工程大一学生蒋东旭。这是他从零到一完整做出一个量化交易系统的第一个项目。

**核心理念**：
- 每个概念都要理解底层逻辑（回测引擎手写而不是用 backtrader）
- 工程落地能力 > 想法数量（"打破想法多、落地为零的模式"）
- 坚持比结果重要（日线涨跌预测 51% 准确率在量化领域是正常的）

---

## 二、项目文件全景图

```
quant-mft/                           # 项目根目录
│
├── README.md                        # 项目首页（GitHub 展示用）
├── CLAUDE.md                        # AI 编程助手指令
├── requirements.txt                 # Python 依赖包清单
├── .gitignore                       # Git 忽略规则
│
├── config/                          # ⚙️ 配置层
│   └── default.yaml                 # 全局默认配置（所有参数集中管理）
│
├── docs/                            # 📄 文档层
│   ├── project-manual.md            # 【本文档】项目说明书
│   ├── architecture.md              # 系统架构文档
│   ├── data_dictionary.md           # 数据字典
│   ├── factor_catalog.md            # 因子目录
│   ├── mindmap-mapping.md           # 思维导图→架构的映射表
│   └── gemini-mindmap-prompt.md     # Gemini 生成思维导图的提示词
│
├── data/                            # 📊 数据层（运行时生成）
│   ├── a_stock_daily.csv            # 原始 OHLCV 数据
│   ├── a_stock_daily_clean.parquet  # 清洗后数据
│   ├── features.parquet             # 特征数据
│   ├── signals.csv                  # 交易信号
│   ├── backtest_trades.csv          # 回测交易明细
│   └── experiments.csv              # 实验记录
│
├── notebooks/                       # 📓 Jupyter 探索笔记
│   ├── 01_data_explore.ipynb        # 数据探索
│   ├── 02_feature_engineering.ipynb # 特征工程探索
│   ├── 03_model_training.ipynb      # 模型训练分析
│   └── 04_backtest_analysis.ipynb   # 回测结果分析
│
├── src/                             # 💻 源码层
│   ├── data/                        # 模块 1+2：数据接入与清洗
│   │   ├── fetcher.py               # 从 akshare 拉取 A 股日线数据
│   │   └── cleaner.py               # 数据清洗（极值/停牌/复权）
│   │
│   ├── features/                    # 模块 3：特征工程
│   │   ├── alpha_factors.py         # Alpha 因子计算（18+因子）
│   │   ├── information_metrics.py   # 信息熵防火墙
│   │   └── selector.py              # 因子筛选与降维
│   │
│   ├── models/                      # 模块 3：机器学习
│   │   ├── train.py                 # LightGBM 训练管线
│   │   ├── evaluate.py              # 多维度模型评估
│   │   └── experiment.py            # 实验追踪记录
│   │
│   ├── backtest/                    # 模块 4+5：回测与诊断
│   │   ├── engine.py                # 手写逐日回测引擎
│   │   ├── risk.py                  # 仓位管理 + 止损 + 熔断
│   │   ├── benchmark.py             # 基准对比（沪深300）
│   │   └── diagnostics.py           # 过拟合诊断（蒙特卡洛）
│   │
│   ├── strategy/                    # 策略层
│   │   └── signal_generator.py      # 信号生成（预测→买卖指令）
│   │
│   └── pipeline/                    # 模块 6：自动化
│       ├── daily_run.py             # 每日一键运行管线
│       └── monitor.py               # 绩效跟踪与漂移检测（v2.0）
│
└── tests/                           # 🧪 测试层
    ├── test_fetcher.py              # 数据获取 + 清洗测试
    ├── test_features.py             # 特征工程测试
    └── test_backtest.py             # 回测 + 模型测试
```

---

## 三、根目录文件详解

### 3.1 `README.md`

**是什么**：项目的"名片"。GitHub 仓库首页展示的内容。

**写给谁看**：任何第一次来到这个仓库的人（未来的你、面试官、协作者）。

**包含什么**：
- 项目一句话介绍 + 技术栈
- 环境配置步骤（让新人 5 分钟内跑起来）
- 完整的目录结构（每个文件后面标注了优先级 [P0]/[P1]/[P2] 和实现状态 ✅/🟡）
- 快速开始命令
- 6 周路线图（对应 6 个 Milestone）
- 架构设计理念摘要
- 相关文档链接

**设计思路**：README 是"摘要"，不是"全貌"。细节放在 docs/ 下的专题文档中，README 只给链接。

---

### 3.2 `CLAUDE.md`

**是什么**：给 Claude Code（AI 编程助手）的指令文件。

**写给谁看**：AI 助手（不是人类）。

**包含什么**：
- 项目背景（蒋东旭是谁、项目目标是什么）
- 技术栈选型和理由
- 目录结构约定
- 6 周路线图的里程碑划分
- **教学原则**（每行代码写中文注释、每个概念用类比讲、每 1-2 小时让他自己动手）
- **蒋东旭的技术基线**（会什么、不会什么）
- **已知风险**（akshare 可能被墙、5060Ti 外接显卡但不需要 GPU）

**为什么它很重要**：每次 AI 助手被调用时都会读取这个文件，确保 AI 给出的代码风格、注释习惯、教学节奏都符合项目规范。

---

### 3.3 `requirements.txt`

**是什么**：Python 依赖包清单。

**写给谁看**：任何需要配置开发环境的人（包括未来的你换电脑时）。

**包含什么**：
- 核心依赖：pandas, numpy, akshare, lightgbm, scikit-learn
- 可视化：matplotlib, jupyter
- 配置管理：pyyaml, python-dotenv
- 通知：python-telegram-bot
- 开发依赖（注释状态）：pytest, black, isort, pre-commit

**安装方式**：`pip install -r requirements.txt`

**设计思路**：
- 主版本号固定（`pandas>=2.0`），小版本允许浮动
- 核心依赖和开发依赖分开标注
- 不锁定到精确版本（`pandas==2.1.3`），因为不同平台可能需要不同的小版本

---

### 3.4 `.gitignore`

**是什么**：Git 忽略规则——告诉 Git 哪些文件不要提交到仓库。

**写给谁看**：Git。

**包含什么**：
- `venv/`：虚拟环境（几万个文件，绝对不能提交）
- `data/*.csv`, `data/*.parquet`, `data/*.pkl`：数据文件不提交（太大且包含东方财富的版权数据）
- `__pycache__/`：Python 编译缓存
- `.env`, `.env.local`, `config.local.*`：含密钥的配置文件不提交
- `.vscode/`, `.idea/`：IDE 个人配置
- `.ipynb_checkpoints/`：Jupyter 自动保存的检查点

**一个重要的坑（已预防）**：
```
data/*.csv     ← 忽略所有 CSV 数据文件
!data/.gitkeep ← 但保留 .gitkeep（空目录占位文件）
```
这样 `data/` 目录会在 git clone 时就存在，不会因为空目录被忽略而导致脚本报错"目录不存在"。

---

## 四、配置文件详解

### 4.1 `config/default.yaml`

**是什么**：项目的"控制面板"——所有可调参数集中在一个文件里。

**为什么需要它**：量化项目最常见的低级错误 = 参数散落在各个 `.py` 文件中。今天改了 `learning_rate=0.01`，下周忘了改过，回测结果完全不可复现。

**使用方式**：
```bash
cp config/default.yaml config/local.yaml   # 复制一份
# 编辑 local.yaml 覆盖你想改的参数
# local.yaml 在 .gitignore 中，不会提交到 GitHub
```

**包含什么**（按模块分组）：

| 配置段 | 控制的模块 | 典型参数 |
|--------|-----------|---------|
| `data` | 数据获取 | start_date, adjust, request_delay, 数据存储路径 |
| `cleaning` | 数据清洗 | Winsorization 方法和阈值, ST股票处理, 停牌过滤 |
| `features` | 特征工程 | 均线窗口, RSI周期, MACD参数, 因子筛选阈值 |
| `model` | 模型训练 | 时间序列切分比例, LightGBM 全部超参 |
| `backtest` | 回测评估 | 初始资金, 交易成本, 风控参数, 过拟合检测阈值 |
| `pipeline` | 自动化 | 定时执行时间, 信号推送阈值, Telegram 配置 |
| `logging` | 日志 | 日志级别, 格式, 文件路径 |

**设计思路**：
- 每个参数都有注释说明其含义和推荐值
- 按模块分组，对应 src/ 下的目录结构
- 代码中用 `yaml.safe_load()` 读取，`local.yaml` 的值覆盖 `default.yaml`

---

## 五、文档层详解

### 5.1 `docs/project-manual.md`（本文档）

**是什么**：项目的"使用说明书"——解释每一个文件的存在理由和设计思路。

**写本文档的目的**：代码注释解释了"这一行在做什么"，本文档解释了"这个文件为什么存在"、"它和其他文件是什么关系"、"你应该在什么时候打开它"。这两个层次的信息缺一不可。

---

### 5.2 `docs/architecture.md`

**是什么**：系统的"建筑蓝图"——用一棵树形结构图展示整个量化系统的 6 大模块和所有子节点。

**写给谁看**：需要理解系统全貌的人。

**包含什么**：
- **系统全景图**：一棵完整的树形图（约 80 个节点）
- **数据流向图**：从 akshare API 到 Telegram 推送的完整链路
- **优先级说明**：每个节点标注 [P0]/[P1]/[P2] 和实现状态
- **6 大模块的详细展开**：每个模块的基本设计逻辑

**和思维导图的关系**：这份文档就是把你的思维导图展开成了可读的树形文本。

---

### 5.3 `docs/data_dictionary.md`

**是什么**：项目所有数据的"说明书"——每一列是什么含义、什么类型、从哪里来。

**为什么需要它**：
- 场景 1：三个月后你打开 `backtest_trades.csv`，看到一列叫 `exit_reason`，里面写着 `trailing_stop`——这是什么意思？是主动止盈还是被动止损？
- 场景 2：协作者李桂聿要对接你的数据，他需要知道 `volume` 的单位是"股"还是"手"。
- 数据字典就是用来回答这些问题的。

**包含 6 张表的完整字段定义**：

| 表名 | 文件 | 包含什么 |
|------|------|---------|
| 原始日线数据 | `a_stock_daily.csv` | OHLCV + 代码 + 名称，9 个字段 |
| 清洗后数据 | `a_stock_daily_clean.parquet` | 原始字段 + is_suspended, is_st, turnover_rate 等 |
| 特征数据 | `features.parquet` | 18+ Alpha 因子，含公式简述 |
| 预测信号 | `signals.csv` | 预测概率 + 交易信号 + 置信度 |
| 回测交易明细 | `backtest_trades.csv` | 每笔交易的完整生命周期（买入→卖出） |
| 实验记录 | `experiments.csv` | 每次模型训练的元数据和结果 |

---

### 5.4 `docs/factor_catalog.md`

**是什么**：所有 Alpha 因子的"字典"——每个因子的定义、公式、理论依据。

**为什么需要它**：
- 同一个因子不能被定义两次（`alpha_factors.py` 里写了一遍公式，`selector.py` 里又写了一遍不一样的 → 结果不可复现）
- 因子目录 = 唯一的真相来源（Single Source of Truth），所有代码引用这个文档中的定义

**包含什么**：
- **因子分类概览**：收益类(4) + 均线偏离类(4) + 动量类(3) + 波动率类(3) + 成交量类(3) + 技术指标类(1) = 18+
- **每个因子的详细说明**：公式（用伪代码写）、参数选择理由、金融含义
- **因子评估标准**：IC 绝对值、IC 稳定性、互相关性、信息熵、缺失率
- **因子命名规范**：`ret_{N}d`, `ma_dev_{N}`, `volatility_{N}` 等

**关键规则（写在了文档末尾）**：
> ⚠️ 禁止在代码中重复定义因子的计算公式。所有因子统一在 `src/features/alpha_factors.py` 中实现，其他模块只调用不重写。

---

### 5.5 `docs/mindmap-mapping.md`

**是什么**：你原始思维导图的每个节点，对应到改进后架构的哪个位置。

**写给谁看**：你自己——当你需要确认"我的原始思路都被保留了吗？"的时候。

**包含什么**：
- **逐节点映射表**：原始 18 个节点 → 保留 6 个 / 增强 6 个 / 新增 15 个 / 调整 4 处
- **每一处变化的原因**：为什么这么改、改完后放在哪里
- **4 个全新横切模块的说明**：配置管理、风险管理、日志系统、代码质量
- **总体统计**：直观展示"我保留了什么、补充了什么"

**核心结论（写在文档末尾）**：
> 你的原始思维导图 = 量化策略的理论框架（What to do）
> 我的改进版本 = 量化策略的工程落地框架（How to build it on GitHub）

---

### 5.6 `docs/gemini-mindmap-prompt.md`

**是什么**：一个可以直接复制粘贴到 Google Gemini 的提示词，让 AI 帮你画思维导图。

**什么时候用**：当你想把改进后的架构可视化为一棵漂亮的思维导图时。

**怎么用**：打开 Gemini → 复制提示词 → 粘贴 → 发送 → 把输出保存为 `architecture.md` 的补充。

**注意**：这个文件是"工具"而非"文档"。生成完之后它的使命就完成了，可以不保留。

---

## 六、源码层详解

源码层按照数据流向分为 6 个子模块。每个 `.py` 文件的"骨架版本"已经创建好了——有完整的函数签名、文档字符串、设计思路注释，但函数体只有 `pass`。

### 6.1 `src/data/` — 数据接入与清洗

这两个文件对应思维导图的第 1 节（数据接入）和第 2 节（数据清洗）。

#### `src/data/fetcher.py` ✅ 已实现

**职责**：从 akshare 拉取 A 股全市场日线 OHLCV 数据。

**核心函数**：
- `fetch_all_stock_codes()` → 获取沪深京所有股票代码和名称
- `fetch_one_stock_daily()` → 拉取单只股票的历史日线（前复权）
- `fetch_all_daily()` → 遍历所有股票，合并保存为 CSV

**关键设计决策**：
- 前复权（`adjust="qfq"`）：保证历史价格和当前价格可比较
- 逐只拉取而非批量：东方财富免费接口没有批量接口，逐只拉 + 加延迟防止封 IP
- 异常隔离：单只股票拉失败不影响其他股票

**依赖**：akshare, pandas

**被谁依赖**：`cleaner.py`（下游，读取它生成的 CSV）

---

#### `src/data/cleaner.py` 🟡 骨架已建

**职责**：把 `fetcher.py` 的原始 CSV 处理成可直接用于特征工程的干净数据。

**核心函数**：
- `winsorize_mad()` → 用 MAD 方法处理极值（为什么不用 3σ？见文件内注释）
- `detect_suspension()` → 检测并标记停牌日（禁止前向填充！）
- `check_adjust_consistency()` → 验证复权价格的逻辑一致性
- `mark_st_stocks()` → 标记 ST 股票（涨跌停规则不同）
- `clean_pipeline()` → 一键执行全部清洗流程

**关键设计决策**：
- MAD 而非 3σ：金融数据有 fat tail，正态分布假设不成立
- 停牌日单独标记而非填充：前向填充会制造假信号
- ST 股票可配置过滤：因为它们涨跌停限制是 5% 而非 10%

**依赖**：pandas, numpy, `fetcher.py`（上游）

**被谁依赖**：`alpha_factors.py`（下游，读取清洗后的 Parquet）

---

### 6.2 `src/features/` — 特征工程

这三个文件对应思维导图的第 2 节后半（信息量度）和第 3 节前半（要素提取）。

#### `src/features/alpha_factors.py` 🟡 骨架已建

**职责**：从 OHLCV 数据衍生出 18+ 个 Alpha 因子。

**核心函数**：
- `calc_ret()` / `calc_all_return_factors()` → 收益率因子（4个）
- `calc_ma_dev()` / `calc_all_ma_factors()` → 均线偏离因子（4个）
- `calc_macd()` → MACD 三件套（3个）
- `calc_volatility()` → 年化波动率
- `calc_atr()` → 平均真实波幅（考虑跳空缺口）
- `calc_bollinger_bands()` → 布林带宽度
- `calc_volume_ratio()` → 量比（2个）
- `calc_turnover_change()` → 换手率变化
- `calc_rsi()` → RSI（Wilder 平滑，非简单移动平均）
- `compute_all_factors()` → 一键计算全部因子

**关键设计决策**：
- 每个因子是独立的函数 → 方便单独测试、按需组合
- 向量化计算（pandas rolling/ewm）→ 不写 for 循环
- 按 code 分组计算 → 防止跨股票混算（平安银行和茅台混在一起算均值）
- Wilder 平滑用于 RSI → 比简单移动平均更稳定

**依赖**：pandas, numpy, `cleaner.py`（上游）

**被谁依赖**：`selector.py`（下游）、`train.py`（下游）

---

#### `src/features/information_metrics.py` 🟡 骨架已建

**职责**：实现"信息熵防火墙"——在特征进入模型之前，先评估每个特征包含多少有效信息。低信息量特征直接过滤。

**核心函数**：
- `calc_entropy()` → 计算信息熵（特征本身的信息量）
- `calc_mutual_info()` → 计算互信息量（特征与目标的关联度，能捕捉非线性关系）
- `evaluate_factor_quality()` → 综合评估：熵 + 互信息 + 推荐建议
- `firewall_filter()` → 防火墙过滤主函数

**为什么叫"防火墙"**：类比网络安全——防火墙在数据进入系统之前拦截威胁。这里的"威胁"= 低质量特征会增加过拟合风险。在特征进入模型之前就过滤掉。

**关键设计决策**：
- 互信息量而非相关系数：金融数据很多关系是非线性的（如 RSI 与涨跌的 U 型关系），相关系数会漏掉这些
- 双指标过滤：熵低 + 互信息低 → 直接过滤；一项低 → 标记 review

**依赖**：scikit-learn, scipy, `alpha_factors.py`（上游）

**被谁依赖**：`selector.py`（下游，在其筛选管线中被调用）

---

#### `src/features/selector.py` 🟡 骨架已建

**职责**：从 20+ 候选因子中选出最优子集——IC 分析 + 去相关 + 可选 PCA 降维。

**核心函数**：
- `calc_ic()` → 计算单个因子的 IC（Spearman 秩相关）
- `calc_ic_summary()` → 所有因子的 IC 统计（均值/标准差/IR）
- `filter_by_ic()` → 剔除 |IC| < 0.02 的弱因子
- `calc_factor_correlation()` → 因子互相关矩阵
- `remove_highly_correlated()` → 冗余因子去重（保留 IC 更高的）
- `apply_pca()` → PCA 降维（备选）
- `select_factors()` → 完整筛选管线

**关键设计决策**：
- Spearman 而非 Pearson：只关心排序关系，不关心线性还是非线性
- 去重规则：两个因子相关系数 > 0.8 → 保留 IC 更高的那个
- PCA 是备选：主成分不可解释（你不知道 PC1 是什么金融含义），不适合需要因子归因的场景

**依赖**：pandas, scikit-learn, `alpha_factors.py`（上游）, `information_metrics.py`（上游）

**被谁依赖**：`train.py`（下游，使用筛选后的因子列表）

---

### 6.3 `src/models/` — 机器学习

这三个文件对应思维导图的第 3 节后半（模型训练、指标评估）。

#### `src/models/train.py` 🟡 骨架已建

**职责**：LightGBM 二分类模型训练——预测次日涨(1)还是跌(0)。

**核心函数**：
- `create_target()` → 构造目标变量（次日涨跌标签）
- `time_series_split()` → 按时间顺序切分 train/val/test（禁止随机切！）
- `prepare_data()` → 整理 LightGBM 训练格式
- `train_model()` → 训练 + Early Stopping
- `train_with_cv()` → 时间序列交叉验证训练（备选）
- `save_model()` / `load_model()` → 模型持久化
- `train_pipeline()` → 完整训练管线

**关键设计决策**：
- 二分类而非回归：涨跌方向比涨跌幅更稳定
- 按时间切分（非随机）：时间序列有因果律——只能用过去预测未来
- Early Stopping：验证集 loss 不降即停，防止过拟合训练集
- L1/L2 正则化在训练时就加，不是过拟合之后才补救

**依赖**：lightgbm, scikit-learn, `alpha_factors.py`（上游）, `selector.py`（上游）

**被谁依赖**：`signal_generator.py`（下游）, `evaluate.py`（同级）, `experiment.py`（同级）

---

#### `src/models/evaluate.py` 🟡 骨架已建

**职责**：多维度评估模型表现——不止看准确率。

**核心函数**：
- `calc_classification_metrics()` → 准确率、精确率、召回率、F1、AUC
- `plot_confusion_matrix()` → 混淆矩阵可视化
- `calc_aic_bic()` → 模型复杂度惩罚（防止无意义地堆参数）
- `analyze_feature_importance()` → 特征重要性排名（gain 和 split 两种）
- `evaluate_by_time_period()` → 按时间段分组评估（检测过拟合）
- `generate_evaluation_report()` → 完整评估报告

**关键设计决策**：
- AUC 比准确率更重要：AUC 衡量模型的排序能力——好股票排在坏股票前面
- AIC/BIC 用于特征选择：新加一个特征，如果增量不如惩罚项大 → 不加
- 特征重要性用 gain 而非 split：gain = 特征带来的信息增益，更有意义

**心理建设（写在文件注释中）**：
> 日线涨跌预测的准确率 50-53% 是正常的。比扔硬币（50%）好一点就足够了——在交易中用仓位管理放大优势。

**依赖**：scikit-learn, matplotlib, `train.py`（上游）

**被谁依赖**：`daily_run.py`（下游，生成每日评估报告）

---

#### `src/models/experiment.py` 🟡 骨架已建

**职责**：实验追踪——记录每次训练的全套参数和结果，方便日后对比。

**核心函数**：
- `log_experiment()` → 记录一次实验（特征列表 + 超参 + 日期范围 + 指标）
- `load_experiments()` → 加载所有历史实验
- `get_best_experiment()` → 查 AUC/夏普 排名前 N 的实验
- `compare_experiments()` → 对比两次实验的差异
- `generate_experiment_summary()` → 所有实验的总结报告

**为什么用 CSV 而不是 MLflow**：
- MVP 阶段数据量小（几十次实验），CSV 完全够用
- GitHub 友好：CSV 可以直接在网页上预览
- 不需要额外部署 MLflow 服务器

**依赖**：pandas, `train.py`（上游）

---

### 6.4 `src/backtest/` — 回测与诊断

这四个文件对应思维导图的第 4 节（验证回测）和第 5 节（绩效诊断）。

#### `src/backtest/engine.py` 🟡 骨架已建

**职责**：手写逐日回测引擎——模拟"如果按照模型的信号来交易，历史表现会怎样"。

**核心数据结构**：
- `Trade` → 单笔交易记录（买入日/价、卖出日/价、收益率、持有天数、出场原因）
- `Position` → 当前持仓快照（成本价、当前价、浮动盈亏）

**核心函数**：
- `calc_buy_price()` / `calc_sell_price()` → 撮合价格（含滑点）
- `calc_commission()` → 交易费用（佣金 + 印花税）
- `calc_shares_to_buy()` → 仓位计算（整数手约束 + 资金约束）
- `run_backtest()` → 主循环：逐日遍历信号 → 执行买卖 → 更新持仓
- `calc_backtest_metrics()` → 计算夏普、回撤、胜率、盈亏比等
- `plot_equity_curve()` → 资金曲线可视化

**为什么手写而不用 backtrader**：
- 理解底层逻辑：撮合、成本、持仓管理——每个环节都知道在做什么
- 完全控制：不受框架限制，可以自由实现任何策略逻辑
- 教育意义：这是蒋东旭第一次做量化，理解底层比用现成框架重要 100 倍

**依赖**：pandas, numpy, `signal_generator.py`（上游）, `risk.py`（同级）, `benchmark.py`（同级）

**被谁依赖**：`diagnostics.py`（下游）, `daily_run.py`（下游）

---

#### `src/backtest/risk.py` 🟡 骨架已建

**职责**：三层风控防护——仓位管理（事前）+ 止损（事中）+ 熔断（事后）。

**三层防护结构**：
```
第一层：仓位管理（开仓前）
  ├─ 单票最大 20%
  ├─ 最多同时持有 5 只
  └─ 单一行业不超过 30%

第二层：止损逻辑（持仓中）
  ├─ 固定比例止损：亏 5% 强制卖出
  ├─ 移动止损：从最高盈利点回落 3% 卖出
  └─ 时间止损：持仓 60 天未盈利 → 卖出

第三层：熔断机制（交易后）
  ├─ 单日亏损超 5% → 暂停新开仓
  └─ 连续 5 天亏损 → 人工复核
```

**核心函数**：
- `check_position_limit()` → 仓位数量限制
- `check_stop_loss()` → 固定比例止损
- `check_trailing_stop()` → 移动止损（锁利润）
- `check_time_stop()` → 时间止损（机会成本）
- `check_daily_loss_limit()` → 熔断
- `risk_check()` → 综合风控决策

**为什么风控模块如此重要**：
> 没有风控的回测报告是废纸。单票 100% 仓位翻倍 = 运气，不是策略好。不止损的策略在实盘里活不过一个月的熊市。

**依赖**：pandas, numpy

**被谁依赖**：`engine.py`（回测循环中调用风控检查）

---

#### `src/backtest/benchmark.py` 🟡 骨架已建

**职责**：生成基准策略（沪深 300 买入持有），用于回答"你的策略真的跑赢大盘了吗？"

**核心函数**：
- `fetch_benchmark_data()` → 获取沪深 300 指数日线数据
- `buy_and_hold_returns()` → 计算买入持有收益率
- `compare_to_benchmark()` → 对比分析（Alpha, Beta, 信息比率, 跟踪误差）
- `rolling_excess_return()` → 滚动超额收益（检测 Alpha 的稳定性）

**关键指标解读**：
- **Alpha**：策略收益中不能用大盘涨跌解释的部分 → 这才是你真正的能力
- **Beta**：策略对大盘的敏感度 → Beta=1 就是跟着大盘走，没超额能力
- **信息比率**：超额收益 / 跟踪误差 → > 0.5 算良好，> 1.0 算优秀

**核心理念（写在文件注释中）**：
> 没有基准的回测毫无意义——牛市里所有策略都赚钱，你不知道是市场好还是你厉害。

**依赖**：akshare, pandas, numpy

**被谁依赖**：`engine.py`（回测结果对比基准）

---

#### `src/backtest/diagnostics.py` 🟡 骨架已建

**职责**：过拟合诊断——用蒙特卡洛参数扰动测试判断策略是否对参数过度敏感。

**核心函数**：
- `check_overfitting_signals()` → 静态预警（夏普 > 3 可疑、夏普 > 4 确定过拟合）
- `perturb_params()` → 对关键参数做 ±20% 随机扰动
- `run_monte_carlo_test()` → 跑 N=1000 次扰动后回测
- `diagnose_overfitting()` → 判断过拟合程度
- `generate_fix_suggestions()` → 生成修复方案
- `run_diagnostic_pipeline()` → 完整诊断管线

**诊断逻辑**：
```
基准回测 → 参数扰动 ±20% → 跑 1000 次回测 → 扰动后夏普下降 > 50% → 过拟合
```

**修复策略（负反馈闭环）**：
```
过拟合判定 → 特征侧：降维，去弱因子
           → 模型侧：增强正则化，剪枝
           → 再回测 → 重新诊断 → 直到通过
```

**量化阈值（写在文件注释中）**：
- 夏普 > 3.0 → 🟡 可疑（日线策略极难达到）
- 夏普 > 4.0 → 🔴 几乎确定过拟合
- 年化 > 50% 且回撤 < 5% → 🔴 不可能三角

**依赖**：numpy, `engine.py`（用于跑扰动后的回测）

---

### 6.5 `src/strategy/` — 策略层

#### `src/strategy/signal_generator.py` 🟡 骨架已建

**职责**：把模型的预测概率转化为具体的买卖指令。

**信号三种状态**：
- **BUY (+1)**：预测概率 > 0.55，排名在 Top N → 买入
- **SELL (-1)**：持仓中但不再推荐，或触发止损 → 卖出
- **HOLD (0)**：不做任何操作

**核心函数**：
- `filter_by_prob()` → 按预测概率筛选（默认 > 0.55）
- `rank_by_confidence()` → 按置信度排序，取 Top N
- `generate_buy_signals()` → 生成买入信号
- `generate_sell_signals()` → 生成卖出信号
- `merge_signals()` → 合并买卖信号
- `generate_signals()` → 主函数

**为什么需要这个模块（而不是直接用模型输出）**：
- 模型输出概率（连续值），交易需要决策（离散：买/卖/持有）
- 需要过滤低置信度预测（50.1% 和 80% 不应同等对待）
- 需要控制每日信号数量（5000+ 只股票不可能全买）

**依赖**：pandas

**被谁依赖**：`engine.py`（下游）, `daily_run.py`（下游）

---

### 6.6 `src/pipeline/` — 自动化管线

#### `src/pipeline/daily_run.py` 🟡 骨架已建

**职责**：项目的"主程序"——把 8 个步骤串联起来，一键执行完整流程。

**8 步流程**：
```
步骤 1: 拉取数据    → fetcher.py
步骤 2: 数据清洗    → cleaner.py
步骤 3: 计算因子    → alpha_factors.py
步骤 4: 因子筛选    → selector.py
步骤 5: 模型预测    → train.py（加载已训练的模型）
步骤 6: 信号生成    → signal_generator.py
步骤 7: 回测更新    → engine.py
步骤 8: Telegram推送 → python-telegram-bot
```

**核心函数**：
- `is_trading_day()` → 检查今天是否为 A 股交易日
- `step_fetch_data()` ~ `step_send_notification()` → 各步骤独立执行
- `run_daily_pipeline()` → 主流程
- `should_retrain()` → 判断是否需要模型重训练
- `setup_logging()` → 配置结构化日志

**设计理念**：
- 每步独立执行 → 一步失败不影响后续（容错）
- 返回每步状态 → 方便排查问题
- 关键步骤失败 → Telegram 告警
- 用 logging 而非 print → 有时间戳、级别、可写文件

**定时触发**（配置层面，不在代码内）：
```cron
30 15 * * 1-5 cd /mnt/d/Code/Quant && venv/bin/python src/pipeline/daily_run.py
```
WSL2 的 cron，每个交易日（周一到周五）下午 3:30 执行。

**依赖**：本项目所有模块 + python-telegram-bot + python-dotenv

---

#### `src/pipeline/monitor.py` 🟡 骨架已建 [P2]

**职责**：实盘后的绩效跟踪与概念漂移检测（v2.0 远期规划）。

**核心功能（接口已定义，待实现）**：
- `track_live_performance()` → 实盘 vs 回测对比
- `calc_rolling_auc()` → 滚动 AUC（检测模型衰减）
- `detect_concept_drift()` → KL 散度检测特征分布变化
- `check_data_quality()` → 上游数据质量监控

**状态**：P2 远期规划（开学后），当前仅定义接口。

**依赖**：pandas, numpy, scipy

---

## 七、测试层详解

测试文件位于 `tests/` 目录，使用 pytest 框架。

### 7.1 `tests/test_fetcher.py`

**覆盖范围**：`fetcher.py` + `cleaner.py`

**测试用例**：
- 股票代码列表的格式验证
- 单只股票数据拉取（以平安银行 000001 为例）
- 数据列的完整性检查
- 日期范围正确性
- 未来数据检查（确保没有 look-ahead）
- 停牌股/新股的空数据处理
- 网络异常的容错

---

### 7.2 `tests/test_features.py`

**覆盖范围**：`alpha_factors.py` + `information_metrics.py` + `selector.py`

**测试用例**：
- 收益率因子数值正确性
- 均线偏离边界条件（价格=均线 → 偏离=0）
- MACD 符号逻辑（涨→正，跌→负）
- RSI 值域验证（0-100）和 Wilder 平滑验证
- ATR 跳空缺口处理
- 信息熵常数序列→熵=0、均匀分布→熵最大
- 互信息独立变量→MI≈0
- IC 计算正确性
- 冗余因子去重逻辑

---

### 7.3 `tests/test_backtest.py`

**覆盖范围**：`engine.py` + `risk.py` + `diagnostics.py` + `benchmark.py` + `train.py`

**测试用例**：
- 买入/卖出价滑点方向
- 佣金最低 5 元、印花税仅卖出
- 整数手约束、资金约束
- 资金曲线初始值
- 止损触发/不触发边界
- 移动止损上移逻辑
- 夏普 > 3 预警、不可能三角
- Alpha/Beta 计算
- 时间序列切分无泄露
- 模型输出概率范围

---

## 八、数据流向

整个系统的数据是怎么流动的：

```
[东方财富API]
     │ akshare 调用
     ▼
[fetcher.py] ──────────────────→ data/a_stock_daily.csv
     │                                    │
     │                                    ▼
     │                           [cleaner.py]
     │                                    │
     │                                    ▼
     │                           data/a_stock_daily_clean.parquet
     │                                    │
     │                                    ▼
     │                           [alpha_factors.py]
     │                                    │
     │                                    ▼
     │                           data/features.parquet
     │                                    │
     │                    ┌───────────────┼───────────────┐
     │                    ▼               ▼               ▼
     │           [information_metrics]  [selector.py]  [train.py]
     │           (信息熵防火墙)     (IC+去相关+PCA)   (LightGBM训练)
     │                    │               │               │
     │                    └───────┬───────┘               │
     │                            ▼                       ▼
     │                    筛选后的因子列表 ───────→ models/model_xxx.txt
     │                                                    │
     │                                                    ▼
     │                                           [signal_generator.py]
     │                                                    │
     │                                                    ▼
     │                                           data/signals.csv
     │                                                    │
     │                                    ┌───────────────┼───────────────┐
     │                                    ▼               ▼               ▼
     │                            [engine.py]      [risk.py]      [benchmark.py]
     │                            (逐日回测)      (风控检查)      (基准对比)
     │                                    │               │               │
     │                                    └───────┬───────┘               │
     │                                            ▼                       ▼
     │                                    [diagnostics.py]        沪深300对比
     │                                    (过拟合诊断)
     │                                            │
     │                                            ▼
     │                                    回测报告 + 资金曲线
     │                                            │
     │                                            ▼
     └────────────────────────────── [daily_run.py] ──→ [Telegram Bot]
                                      (每日自动)         (信号推送)
```

---

## 九、开发顺序建议

按照数据流的依赖关系，推荐的开发顺序：

```
第 1 步：src/data/fetcher.py       ✅ 已完成
         └─ 验证：能拉取数据 → data/a_stock_daily.csv 存在

第 2 步：src/data/cleaner.py       🟡 下一优先
         └─ 验证：清洗后 → data/a_stock_daily_clean.parquet 存在

第 3 步：src/features/alpha_factors.py  🟡
         └─ 验证：因子计算 → data/features.parquet 存在

第 4 步：src/models/train.py       🟡
         └─ 验证：模型训练完成 → 输出 AUC > 0.5

第 5 步：src/models/evaluate.py    🟡
         └─ 验证：评估报告生成 → 特征重要性排名有意义

第 6 步：src/strategy/signal_generator.py  🟡
         └─ 验证：信号表生成 → 每天有合理数量的买卖信号

第 7 步：src/backtest/risk.py      🟡
         └─ 验证：风控规则正确触发/不触发

第 8 步：src/backtest/engine.py    🟡
         └─ 验证：回测跑通 → 资金曲线 + 指标输出

第 9 步：src/backtest/benchmark.py 🟡
         └─ 验证：策略 vs 沪深300 对比图

第 10 步：src/backtest/diagnostics.py  🟡
          └─ 验证：过拟合诊断报告（蒙特卡洛）

第 11 步：src/models/experiment.py 🟡
          └─ 验证：实验记录 CSV → get_best_experiment() 有结果

第 12 步：src/features/information_metrics.py  🟡 [P1]
          └─ 验证：防火墙正确过滤低信息量特征

第 13 步：src/features/selector.py  🟡 [P1]
          └─ 验证：因子筛选后模型 AUC 不降反升或持平

第 14 步：src/pipeline/daily_run.py  🟡
          └─ 验证：一键执行 → Telegram 收到推送

第 15 步：cron + Telegram Bot 配置  🟡
          └─ 验证：每天 15:30 自动触发 → 手机收到消息

第 16 步：tests/ 测试补全           🟡 [P1]
          └─ 验证：pytest 全部通过

第 17 步：src/pipeline/monitor.py   🔴 [P2]
          └─ 验证：开学后实现
```

**依赖关系规则**：只有上游模块完成后，下游模块才能验证。所以严格按照序号顺序开发，不要跳步。

---

## 附录：快速查阅表

| 我想... | 打开这个文件 |
|---------|------------|
| 了解项目全貌 | `README.md` |
| 理解系统架构 | `docs/architecture.md` |
| 查某个字段的含义 | `docs/data_dictionary.md` |
| 查某个因子的公式 | `docs/factor_catalog.md` |
| 看我的思维导图对应到哪了 | `docs/mindmap-mapping.md` |
| 理解每个文件为什么存在 | `docs/project-manual.md`（本文档） |
| 修改模型超参 | `config/default.yaml` → `model` 段 |
| 修改回测参数 | `config/default.yaml` → `backtest` 段 |
| 看数据是怎么拉的 | `src/data/fetcher.py` |
| 看数据是怎么洗的 | `src/data/cleaner.py` |
| 看因子是怎么算的 | `src/features/alpha_factors.py` |
| 看模型是怎么训练的 | `src/models/train.py` |
| 看模型效果怎么样 | `src/models/evaluate.py` |
| 看过拟合怎么检测 | `src/backtest/diagnostics.py` |
| 看止损怎么触发 | `src/backtest/risk.py` |
| 看每天自动跑什么 | `src/pipeline/daily_run.py` |
| 安装依赖 | `requirements.txt` |
| 生成思维导图 | `docs/gemini-mindmap-prompt.md`（复制到 Gemini） |

---

> 📌 本文档会随着项目进展持续更新。每当新增或修改一个文件时，同步更新本文档的对应章节。
>
> 作者：蒋东旭 | 最后更新：2026-07-15
