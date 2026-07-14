# quant-mft 🚀

**中频 ML 量化交易系统** — A 股 + 加密货币，LightGBM 预测 + 回测 + Telegram 信号推送。

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
| 自动化 | cron (WSL2)，每个交易日 15:30 |
| 通知 | Telegram Bot |

## 环境配置

```bash
# 1. 克隆项目
git clone git@github.com:Alastair-Jiang/quant-mft.git
cd quant-mft

# 2. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install pandas numpy matplotlib jupyter akshare

# 4. 验证
python -c "import pandas; import akshare; print('OK')"
```

## 目录结构

```
quant-mft/
├── README.md
├── data/                  # 原始数据（CSV，不提交 git）
├── notebooks/             # Jupyter 探索分析
├── src/
│   ├── data/              # 数据获取脚本
│   ├── features/          # 特征工程
│   ├── models/            # ML 模型训练与预测
│   ├── backtest/          # 回测引擎
│   ├── strategy/          # 策略逻辑
│   └── pipeline/          # 每日自动运行管线
├── tests/
└── docs/
```

## 快速开始

```bash
# 拉取全市场 A 股日线数据
python src/data/fetcher.py
```

## 路线图（6 周）

1. **基础设施 + 数据管线**（4 天）
2. **ML 预测模型**（7 天）
3. **回测引擎**（6 天）
4. **自动化 + Telegram 通知**（5 天）
5. **虚拟盘 + 加密货币**（开学后长期）

## 作者

蒋东旭 — 东北财经大学金融工程

---
*"第一次把想法从零到一完整做出来"*
