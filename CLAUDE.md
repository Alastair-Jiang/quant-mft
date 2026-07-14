# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quant-mft** — 蒋东旭（东北财经大学金融工程大一）的中频 ML 量化交易系统。支持 A 股 + 加密货币，最终形态：每日收盘后自动拉数据 → LightGBM 预测 → 回测 → Telegram 推送信号。

- **工期：** 2026-07-14 → 2026-08-25，每天 8 小时
- **GitHub：** `https://github.com/Alastair-Jiang/quant-mft`（SSH 已配好）
- **核心理念：** 这是蒋东旭第一次把想法从零到一完整做出来——打破"想法多、落地为零"的模式，提升计算机/数学/金融/项目推进四方面能力。

## 技术栈

| 层 | 选型 |
|----|------|
| 语言 | Python 3.13.5 (Windows) |
| 环境隔离 | venv (`venv/`) |
| 数据源 | akshare（免费 A 股 OHLCV，底层调东方财富/新浪接口） |
| 数据处理 | pandas, numpy |
| 可视化 | matplotlib, jupyter |
| ML | LightGBM (scikit-learn) |
| 回测 | 手写引擎（不用 backtrader，让蒋东旭理解底层逻辑） |
| 自动化 | cron (WSL2)，每个交易日 15:30 |
| 通知 | Telegram Bot (python-telegram-bot) |
| 加密（后期） | ccxt |

## 环境配置

- **工作目录：** `D:\Code\Quant`
- **平台：** Windows 11，Git Bash 终端
- **Python：** 3.13.5（Windows 原生）
- **虚拟环境：** `venv\Scripts\activate`
- **包安装：** `pip install pandas numpy matplotlib jupyter akshare`（已安装完成）
- **akshare 网络：** 清华大学 PyPI 镜像（`pypi.tuna.tsinghua.edu.cn`），已配置好

## 项目目录结构

```
quant-mft/
├── README.md
├── .gitignore
├── data/                  # 原始数据 CSV
├── notebooks/             # Jupyter 探索
├── src/
│   ├── data/              # 数据获取脚本 (fetcher.py)
│   ├── features/          # 特征工程
│   ├── models/            # ML 模型 (LightGBM)
│   ├── backtest/          # 回测引擎（手写）
│   ├── strategy/          # 策略逻辑
│   └── pipeline/          # 每日自动运行 (daily_run.py)
├── tests/
└── docs/
```

## 6 周路线图

### 阶段 1：基础设施 + 数据管线（4 天）
- GitHub 仓库 + 项目骨架
- `src/data/fetcher.py`：akshare 拉 A 股全市场日线 OHLCV → CSV
- `notebooks/01_data_explore.ipynb`：K 线图 + 成交量分布

### 阶段 2：ML 预测（7 天）
- 从 OHLCV 构造 20+ 特征（收益率、均线偏离、RSI、波动率、成交量变化、MACD 等）
- LightGBM 二分类预测次日涨跌
- 时间序列按时间切分 train/test（不能随机切）
- 评估：准确率、AUC、混淆矩阵、特征重要性
- **预期准确率 50-53%，不要让他觉得失败——日线涨跌预测本身就是极难问题**

### 阶段 3：回测（6 天）
- 手写引擎：逐日遍历、持仓模拟、买入/卖出信号
- 手续费万三、印花税千一（卖出）、滑点 0.1%
- 资金曲线图、夏普比率、最大回撤、胜率、盈亏比
- 策略归因分析

### 阶段 4：自动化 + 通知（5 天）
- `src/pipeline/daily_run.py` 整合所有脚本
- cron 定时每个交易日 15:30
- Telegram Bot 推送信号

### 阶段 5（开学后）：虚拟盘 + 加密货币
- A 股虚拟盘：xtquant 或 easytrader 接华泰/东财
- 加密：ccxt + Binance 测试网
- 跑 1-2 个月对比回测 vs 实盘

## 教学原则

- **每行代码写中文注释，解释为什么这么写**
- **每个概念用类比讲**（"DataFrame = Excel 表格"、"git commit = 游戏存档"）
- **别堆术语**——蒋东旭听不懂会走神然后换话题
- **每 1-2 小时让他自己动手做一遍**，别连续讲太久
- **每完成一个任务 git commit + push**，同时教 Git 工作流
- **心理建设：** 他有"遇到困难→放弃→换想法"的历史模式，需要持续提醒 51% 准确率在量化领域是正常的，坚持比结果重要

## 蒋东旭技术基线

- Python：能写基础脚本，pandas/numpy 没用过
- 数学：较扎实（概率、统计、线代，备赛 CMC）
- Git/GitHub：完全不会（但 Git 已装，SSH 已配）
- 量化交易：零经验
- 理解快但工程落地弱，配合李桂聿（南航 AI 同学）做底层基础设施

## 已知风险

1. **akshare 可能被墙：** 底层调东方财富/新浪接口，需验证 WSL2 网络连通性（之前 GitHub HTTPS 被墙过，走 SSH 解决了）
2. **5060Ti 外接显卡：** 日线 LightGBM 完全用不到 GPU，CPU 就够了
3. **PEP 668：** WSL2 必须用 venv，不能直接 `pip install`
