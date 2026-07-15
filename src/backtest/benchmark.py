"""
基准对比模块 — 策略 vs 基准指数

作用：
    生成基准策略的资金曲线，用于与 ML 策略对比。
    没有基准的回测毫无意义——牛市里所有策略都赚钱。

基准选择：
    1. 沪深 300 (000300) — 代表大盘蓝筹，最常用的 A 股基准
    2. 中证 500 (000905) — 代表中盘股
    3. 买入持有策略 — 最简单的被动策略

核心问题（用基准回答）：
    - 你的策略赚了 20%，同期沪深 300 涨了 30% → 策略实际上是跑输大盘的
    - 你的策略亏了 5%，同期沪深 300 亏了 15% → 策略有超额收益（alpha）
    - 你的策略年化 30%，基准年化 8%，但最大回撤 40%（基准 15%）
      → 超额收益来自承担更高风险，而不是真正的 alpha

评估维度：
    - 超额收益 (Excess Return) = 策略收益 - 基准收益
    - 跟踪误差 (Tracking Error) = 策略与基准收益差的标准差
    - 信息比率 (Information Ratio) = 超额收益 / 跟踪误差
    - Beta = 策略收益对基准收益的敏感度
    - Alpha = 策略收益中不能用 Beta 解释的部分（这才是真正的能力！）

依赖：
    - akshare（拉指数日线数据）
    - pandas, numpy
    - src/backtest/engine.py（回测引擎，用于对比）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


# ============================================================
# 配置
# ============================================================

DEFAULT_BENCHMARK = "000300"    # 默认基准：沪深300
RISK_FREE_RATE = 0.03           # 无风险利率（约等于一年期定存）


# ============================================================
# 1. 基准数据获取
# ============================================================

def fetch_benchmark_data(index_code: str = DEFAULT_BENCHMARK,
                         start_date: str = "20230101",
                         end_date: str = None) -> pd.DataFrame:
    """
    获取基准指数的日线数据

    用 akshare 拉取沪深 300 / 中证 500 等指数的 OHLCV 数据

    参数:
        index_code: 指数代码（"000300"=沪深300, "000905"=中证500, "000852"=中证1000）
        start_date: 起始日期
        end_date: 结束日期

    返回:
        DataFrame，列: date, close, ...
    """
    pass  # TODO: 实现基准数据获取


# ============================================================
# 2. 买入持有策略
# ============================================================

def buy_and_hold_returns(price_series: pd.Series) -> pd.Series:
    """
    计算买入持有策略的每日收益率

    逻辑：
        - 第一天买入，一直持有到最后一天
        - 每日收益率 = close / close.shift(1) - 1
        - 累计净值 = (1 + daily_return).cumprod()

    参数:
        price_series: 收盘价序列（按日期排序）

    返回:
        每日收益率序列
    """
    pass  # TODO: 实现买入持有收益计算


def calc_benchmark_equity_curve(benchmark_df: pd.DataFrame,
                                initial_capital: float) -> pd.DataFrame:
    """
    计算基准策略的资金曲线

    参数:
        benchmark_df: 基准数据（含 daily_return 列）
        initial_capital: 初始资金

    返回:
        DataFrame，列: date, equity（资金/净值）
    """
    pass  # TODO: 实现基准资金曲线


# ============================================================
# 3. 对比分析
# ============================================================

def compare_to_benchmark(strategy_equity: pd.DataFrame,
                         benchmark_equity: pd.DataFrame) -> Dict[str, float]:
    """
    对比策略与基准的表现

    核心指标：
        - excess_return: 策略累计收益 - 基准累计收益
        - alpha: Jensen's Alpha（策略收益中不能用 Beta 解释的部分）
        - beta: 策略对基准的敏感度
               Beta > 1 = 比大盘波动大（进攻型）
               Beta < 1 = 比大盘波动小（防守型）
        - information_ratio: 信息比率 = 年化超额收益 / 跟踪误差
               IR > 0.5 算良好，IR > 1.0 算优秀
        - tracking_error: 跟踪误差 = 超额收益的标准差

    通俗解读：
        - Alpha 是真正的能力 → 你应该为 Alpha 骄傲
        - Beta 是"跟风" → 大盘涨你也涨，不是你的本事
        - 如果 Alpha ≈ 0 且 Beta ≈ 1 → 你的策略 ≈ 买了个指数基金

    参数:
        strategy_equity: 策略资金曲线（含 'date', 'equity' 列）
        benchmark_equity: 基准资金曲线（含 'date', 'equity' 列）

    返回:
        {'excess_return': ..., 'alpha': ..., 'beta': ..., ...}
    """
    pass  # TODO: 实现基准对比分析


# ============================================================
# 4. 滚动对比
# ============================================================

def rolling_excess_return(strategy_equity: pd.DataFrame,
                          benchmark_equity: pd.DataFrame,
                          window: int = 60) -> pd.Series:
    """
    计算滚动超额收益（用于看策略 alpha 的稳定性）

    逻辑：每 60 个交易日（约三个月）算一次超额收益
         → 观察 alpha 是否在时间上稳定
         → 如果 alpha 集中在某一段时间，可能是运气

    参数:
        strategy_equity: 策略资金曲线
        benchmark_equity: 基准资金曲线
        window: 滚动窗口（交易日数）

    返回:
        滚动超额收益序列
    """
    pass  # TODO: 实现滚动超额收益


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("基准对比模块")
    print("核心观念：没有基准的回测毫无意义")
    print(f"默认基准：沪深300 ({DEFAULT_BENCHMARK}) 买入持有")
    print("关键指标：Alpha（真正的能力）vs Beta（跟风大盘）")
