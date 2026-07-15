# quant-mft 🚀

**中频 ML 量化交易系统** — A 股 + 加密货币，LightGBM 预测 + 回测 + Telegram 信号推送。

> 📌 完整架构文档：[docs/architecture.md](docs/architecture.md)

## 项目目标

每天收盘后自动拉数据 → LightGBM 模型预测次日涨跌 → 回测评估 → Telegram 推送交易信号。

## 技术栈

| 层 | 选型 |
|----|------|
| 语言 | Python 3.13 |
| 数据源 | akshare（免费 A 股 OHLCV） |
| 数据处理 | pandas, numpy |
| 可视化 | matplotlib, jupyter |
| ML 模型 | LightGBM (scikit-learn) |
| 回测 | 手写引擎 |
| 风控 | 仓位管理 + 止损 + 熔断 |
| 自动化 | cron (WSL2)，每个交易日 15:30 |
| 通知 | Telegram Bot |
| 实验追踪 | CSV 日志 |

## 环境配置

```bash
# 1. 克隆项目
git clone git@github.com:Alastair-Jiang/quant-mft.git
cd quant-mft

# 2. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置文件
cp config/default.yaml config/local.yaml
# 编辑 config/local.yaml 填入你的参数

# 5. 验证
python -c "import pandas; import akshare; print('OK')"
```

## 目录结构

```
quant-mft/
├── README.md                   # 项目说明
├── CLAUDE.md                   # AI 助手指令
├── requirements.txt            # Python 依赖清单
├── .gitignore
├── config/
│   └── default.yaml            # 配置模板（所有可调参数集中管理）
├── data/                       # 数据目录（CSV/Parquet 不提交 git）
│   └── README.md               # 数据字典快速参考
├── notebooks/                  # Jupyter 探索分析
├── src/
│   ├── data/                   # 📂 数据接入 + 清洗
│   │   ├── fetcher.py          #   [P0] A股日线数据获取 ✅
│   │   └── cleaner.py          #   [P0] 数据清洗（Winsorization/停牌/复权）
│   ├── features/               # 📂 特征工程
│   │   ├── alpha_factors.py    #   [P0] Alpha 因子计算（18+因子）
│   │   ├── information_metrics.py # [P1] 信息熵防火墙
│   │   └── selector.py         #   [P1] 因子筛选与降维
│   ├── models/                 # 📂 ML 模型
│   │   ├── train.py            #   [P0] LightGBM 训练管线
│   │   ├── evaluate.py         #   [P0] 多维度模型评估
│   │   └── experiment.py       #   [P1] 实验追踪记录
│   ├── backtest/               # 📂 回测与风控
│   │   ├── engine.py           #   [P0] 手写逐日回测引擎
│   │   ├── risk.py             #   [P0] 仓位管理 + 止损 + 熔断
│   │   ├── benchmark.py        #   [P0] 基准对比（沪深300）
│   │   └── diagnostics.py      #   [P1] 过拟合诊断（蒙特卡洛）
│   ├── strategy/               # 📂 策略
│   │   └── signal_generator.py #   [P0] 信号生成（预测→买卖指令）
│   └── pipeline/               # 📂 自动化管线
│       ├── daily_run.py        #   [P0] 每日一键运行
│       └── monitor.py          #   [P2] 绩效跟踪与漂移检测
├── tests/                      # 单元测试 (pytest)
│   ├── test_fetcher.py
│   ├── test_features.py
│   └── test_backtest.py
└── docs/                       # 文档
    ├── architecture.md         # 完整架构文档
    ├── data_dictionary.md      # 数据字典
    ├── factor_catalog.md       # 因子目录
    ├── mindmap-mapping.md      # 思维导图→架构 映射表
    └── gemini-mindmap-prompt.md # Gemini 思维导图生成提示词
```

## 快速开始

```bash
# 1. 拉取全市场 A 股日线数据
python src/data/fetcher.py

# 2. 数据清洗
python src/data/cleaner.py

# 3. 计算因子
python src/features/alpha_factors.py

# 4. 训练模型
python src/models/train.py

# 5. 运行完整每日管线（一键执行 1-4 + 信号推送）
python src/pipeline/daily_run.py
```

## 路线图（6 周）

| 阶段 | 内容 | 天数 | 关键交付 |
|------|------|------|---------|
| M1 数据管线 | 数据获取 + 清洗 | 4天 | `fetcher.py`, `cleaner.py` |
| M2 特征工厂 | Alpha因子 + 信息熵防火墙 | 7天 | `alpha_factors.py`, `information_metrics.py` |
| M3 ML预测 | LightGBM训练 + 评估 + 实验追踪 | 7天 | `train.py`, `evaluate.py`, `experiment.py` |
| M4 回测诊断 | 回测引擎 + 基准 + 风控 + 过拟合诊断 | 6天 | `engine.py`, `benchmark.py`, `risk.py`, `diagnostics.py` |
| M5 自动化 | 每日管线 + cron + Telegram | 5天 | `daily_run.py`, Telegram Bot |
| M6 虚拟盘 | xtquant/easytrader + 加密货币 | 开学后 | v2.0 |

### 优先级说明
- **[P0]** = MVP 必须（6周内完成）
- **[P1]** = 加分项（有时间就做）
- **[P2]** = v2.0 远期规划（开学后）

## 架构设计理念

```
🧠 核心理念：你的思维导图定义了"完整的量化系统长什么样"，
             本项目补充了"如何在 GitHub 上把它搭成一个能跑的项目"。

🔑 四个横切关注点（贯穿所有模块）：
   ├─ 配置管理：所有参数集中在 config/default.yaml → 回测可复现
   ├─ 风险管理：仓位 + 止损 + 熔断 → 没有风控的回测是废纸
   ├─ 日志系统：logging 模块 → print() 在生产环境排查不了问题
   └─ 基准对比：沪深300买入持有 → 没有基准的回测毫无意义
```

## 相关文档

| 文档 | 内容 |
|------|------|
| [架构文档](docs/architecture.md) | 系统全景图 + 数据流向 + 模块职责 |
| [数据字典](docs/data_dictionary.md) | 每个字段的含义、类型、来源 |
| [因子目录](docs/factor_catalog.md) | 所有 Alpha 因子的定义和公式 |
| [映射表](docs/mindmap-mapping.md) | 原始思维导图 → 改进架构的逐节点对照 |

## 作者

蒋东旭 — 东北财经大学金融工程

---
*"第一次把想法从零到一完整做出来"*
