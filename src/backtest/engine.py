"""
回测引擎模块 — 手写逐日回测模拟器

作用：
    基于 ML 模型生成的交易信号，模拟真实的交易过程，计算策略的历史表现。

为什么手写而不是用 backtrader/zipline？
    - 理解底层逻辑：知道回测的每一步在做什么（撮合、成本、持仓管理）
    - 灵活性：完全控制所有细节，不会受框架限制
    - 教育意义：这是蒋东旭第一次做量化，理解底层比用现成框架重要 100 倍

核心概念（逐个理解）：
    1. 逐日遍历 = 按时间顺序，每天检查是否有买卖信号
    2. 撮合逻辑 = 以次日开盘价成交（实战中T日信号 → T+1日执行）
    3. 持仓模拟 = 记录每笔交易的入场价、出场价、持仓天数
    4. 资金管理 = 初始资金 → 买入扣资金 → 卖出加资金 → 资金曲线

回测输出：
    - 资金曲线（净值随时间变化）
    - 每笔交易明细（买入日/价、卖出日/价、收益率）
    - 回测汇总指标（总收益、夏普、最大回撤、胜率、盈亏比）

⚠️ 回测的三大陷阱：
    1. 未来函数 (Look-ahead bias): 用 t+1 已知的信息做 t 时刻的决策
    2. 幸存者偏差 (Survivorship bias): 只回测现在还在的股票，无视已退市的
    3. 过拟合 (Overfitting): 参数调得太好 → 回测曲线完美 → 实盘跑飞

依赖：
    - pandas, numpy
    - src/strategy/signal_generator.py（上游，提供交易信号）
    - src/backtest/risk.py（风控模块，提供仓位管理和止损）
    - src/backtest/benchmark.py（基准对比）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class Trade:
    """单笔交易记录"""
    trade_id: int               # 交易编号
    code: str                   # 股票代码
    name: str                   # 股票名称
    buy_date: str               # 买入日期
    buy_price: float            # 买入价（含滑点）
    sell_date: str              # 卖出日期（None=还在持仓中）
    sell_price: float           # 卖出价（含滑点）
    shares: int                 # 股数
    buy_cost: float             # 买入成本（手续费）
    sell_cost: float            # 卖出成本（手续费+印花税）
    return_pct: float           # 收益率
    holding_days: int           # 持有天数
    exit_reason: str            # 出场原因: signal / stop_loss / take_profit


@dataclass
class Position:
    """当前持仓"""
    code: str                   # 股票代码
    name: str                   # 股票名称
    shares: int                 # 持仓股数
    avg_cost: float             # 平均成本价
    current_price: float        # 当前价
    market_value: float         # 市值
    unrealized_pnl: float       # 浮动盈亏
    unrealized_pnl_pct: float   # 浮动盈亏比例
    holding_days: int           # 已持有天数


# ============================================================
# 配置
# ============================================================

INITIAL_CAPITAL = 100_000      # 初始资金（元）

# 交易成本（A股标准）
COMMISSION_RATE = 0.0003       # 佣金费率（万三，买卖双向收取）
STAMP_TAX_RATE = 0.001         # 印花税（千一，仅卖出时收取）
SLIPPAGE = 0.001               # 滑点（0.1%，买入向上滑、卖出向下滑）

# A股交易规则
MIN_SHARES = 100               # 最少买 1 手 = 100 股
SHARES_UNIT = 100              # 必须是 100 股的整数倍


# ============================================================
# 1. 撮合引擎
# ============================================================

def calc_buy_price(close_price: float, slippage: float = SLIPPAGE) -> float:
    """
    计算实际买入价

    逻辑：买入时以次日开盘价（简化用收盘价代理），加上滑点
         actual_buy = close * (1 + slippage)
         例如：close=10.00, slippage=0.1% → actual_buy=10.01

    参数:
        close_price: 信号日的收盘价
        slippage: 滑点比例

    返回:
        实际买入价
    """
    pass  # TODO: 实现买入价计算


def calc_sell_price(close_price: float, slippage: float = SLIPPAGE) -> float:
    """
    计算实际卖出价

    逻辑：卖出时以次日开盘价（简化用收盘价代理），减去滑点
         actual_sell = close * (1 - slippage)
         例如：close=10.00, slippage=0.1% → actual_sell=9.99

    参数:
        close_price: 信号日的收盘价
        slippage: 滑点比例

    返回:
        实际卖出价
    """
    pass  # TODO: 实现卖出价计算


def calc_commission(price: float, shares: int, is_sell: bool = False) -> float:
    """
    计算交易费用

    A股费用结构：
        - 佣金：成交金额 × 万三（双向收取，最低 5 元）
        - 印花税：成交金额 × 千一（仅卖出时收取）

    参数:
        price: 成交价
        shares: 成交股数
        is_sell: 是否为卖出

    返回:
        总费用（元）
    """
    pass  # TODO: 实现交易费用计算


# ============================================================
# 2. 仓位管理
# ============================================================

def calc_shares_to_buy(available_cash: float, price: float,
                       max_position_pct: float = 0.2) -> int:
    """
    计算可买入的股数

    约束：
        1. 不能超过可用资金
        2. 单票不超过总资金的 max_position_pct（如 20%）
        3. 必须是 100 股（1手）的整数倍

    参数:
        available_cash: 可用资金
        price: 买入价
        max_position_pct: 单票最大仓位比例

    返回:
        可买入股数
    """
    pass  # TODO: 实现买入股数计算


# ============================================================
# 3. 核心回测逻辑
# ============================================================

def run_backtest(signals_df: pd.DataFrame,
                 price_df: pd.DataFrame,
                 initial_capital: float = INITIAL_CAPITAL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    【主函数】运行回测

    逐日遍历逻辑（伪代码）：
        for each 交易日:
            1. 检查是否有今日的信号（买入/卖出）
            2. 处理卖出信号：以次日收盘价卖出持仓
            3. 处理买入信号：按可用资金和仓位规则买入
            4. 更新持仓市值（mark-to-market）
            5. 检查止损条件
            6. 记录当日资产快照

    撮合规则：
        - T 日信号 → T+1 日以收盘价成交
          （A股 T+1 制度：今天买入的股票明天才能卖出）
        - 涨跌停板无法交易 → 需要检查（简化版可以跳过）

    参数:
        signals_df: 信号数据（含 date, code, signal 列）
        price_df: 价格数据（含 date, code, close 列）
        initial_capital: 初始资金

    返回:
        (trades_df: 每笔交易的明细,
         daily_snapshot_df: 每日资金快照)
    """
    pass  # TODO: 实现逐日回测主循环


# ============================================================
# 4. 回测结果分析
# ============================================================

def calc_backtest_metrics(trades_df: pd.DataFrame,
                          daily_snapshot_df: pd.DataFrame,
                          benchmark_df: pd.DataFrame = None) -> Dict[str, float]:
    """
    计算回测的核心评估指标

    指标清单：
        收益类：
            - total_return: 累计收益率
            - annual_return: 年化收益率
            - excess_return: 超额收益（vs 基准）
        风险类：
            - max_drawdown: 最大回撤
            - annual_volatility: 年化波动率
            - downside_volatility: 下行波动率（只计算下跌部分）
        综合类：
            - sharpe_ratio: 夏普比率 = (年化收益 - 无风险利率) / 年化波动率
            - calmar_ratio: 卡玛比率 = 年化收益 / 最大回撤
            - sortino_ratio: 索提诺比率 = (年化收益 - 无风险利率) / 下行波动率
        交易类：
            - win_rate: 胜率 = 盈利交易数 / 总交易数
            - profit_loss_ratio: 盈亏比 = 平均盈利 / 平均亏损
            - avg_holding_days: 平均持仓天数
            - total_trades: 总交易次数

    参数:
        trades_df: 交易明细
        daily_snapshot_df: 每日资金快照（含 'equity' 列 = 总资产）
        benchmark_df: 基准资金曲线（可选）

    返回:
        Dict[str, float]，指标名 → 值
    """
    pass  # TODO: 实现回测指标计算


# ============================================================
# 5. 可视化
# ============================================================

def plot_equity_curve(daily_snapshot_df: pd.DataFrame,
                      benchmark_df: pd.DataFrame = None,
                      trades_df: pd.DataFrame = None,
                      save_path: str = None):
    """
    绘制资金曲线图

    包含：
        - 策略资金曲线（蓝色）
        - 基准资金曲线（灰色虚线，如沪深300买入持有）
        - 买入/卖出标记点
        - 最大回撤区间（红色阴影）
        - 回撤曲线（副图）

    参数:
        daily_snapshot_df: 每日快照
        benchmark_df: 基准数据
        trades_df: 交易明细（用于标注买卖点）
        save_path: 保存路径
    """
    pass  # TODO: 实现资金曲线图


def plot_trade_distribution(trades_df: pd.DataFrame, save_path: str = None):
    """
    绘制交易分布图

    包含：
        - 每笔交易收益率直方图
        - 持仓天数分布
        - 月度收益热力图
    """
    pass  # TODO: 实现交易分布图


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("回测引擎模块")
    print("用法：run_backtest(signals_df, price_df, initial_capital=100000)")
    print("设计理念：手写引擎 → 理解撮合逻辑、成本计算、资金管理的每一步")
