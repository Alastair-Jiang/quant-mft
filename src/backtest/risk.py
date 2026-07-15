"""
风控模块 — 仓位管理与止损逻辑

作用：
    在回测和实盘中执行风险控制规则。没有风控的回测报告是废纸。

为什么风控对量化交易至关重要？
    - 回测好 ≠ 实盘好，中间差的就是风险控制
    - 单票 100% 仓位翻倍了 = 运气，不是策略好
    - 不止损的策略在实盘里活不过一个月的熊市

风控规则（三层防护）：
    第一层：仓位管理（事前控制）
        - 单票最大仓位限制（如 20%）
        - 最大同时持仓数（如 5 只）
        - 行业集中度限制（如单一行业不超过 30%）

    第二层：止损逻辑（事中控制）
        - 固定比例止损：亏损超过 5% → 无条件卖出
        - 移动止损 (Trailing Stop)：从最高盈利点回落 3% → 卖出
        - 时间止损：持仓超过 30 天还没盈利 → 考虑卖出

    第三层：熔断机制（事后控制）
        - 单日亏损超过总资金 5% → 暂停交易
        - 连续 N 天跑输基准 → 人工复核

依赖：
    - pandas, numpy
    - src/backtest/engine.py（回测引擎调用风控规则）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================
# 配置
# ============================================================

# 仓位管理
MAX_POSITION_PCT = 0.2          # 单票最大仓位（20%）
MAX_POSITIONS = 5               # 最大同时持仓数
MAX_INDUSTRY_PCT = 0.3          # 单一行业最大仓位（30%）

# 止损
STOP_LOSS_PCT = 0.05            # 固定止损：亏损 5% 强制卖出
TRAILING_STOP_PCT = 0.03       # 移动止损：从最高点回落 3%
MAX_HOLDING_DAYS = 60           # 时间止损：持仓 60 天未盈利 → 卖出

# 熔断
MAX_DAILY_LOSS_PCT = 0.05      # 单日最大亏损 5%
MAX_CONSECUTIVE_LOSS_DAYS = 5   # 连续亏损天数
BENCHMARK_UNDERPERFORM_DAYS = 20  # 连续跑输基准天数


# ============================================================
# 1. 仓位管理（事前控制）
# ============================================================

def check_position_limit(current_positions: int, max_positions: int = MAX_POSITIONS) -> bool:
    """
    检查是否达到最大持仓数

    逻辑：
        - 如果已经持有 5 只股票 → 新信号不执行
        - 需要先卖出一只，才能买新的

    参数:
        current_positions: 当前持仓数
        max_positions: 最大持仓限制

    返回:
        True = 可以开新仓, False = 已达上限
    """
    pass  # TODO: 实现仓位数量检查


def calc_max_position_size(total_capital: float,
                           max_pct: float = MAX_POSITION_PCT) -> float:
    """
    计算单只股票的最大仓位金额

    参数:
        total_capital: 当前总资金
        max_pct: 单票最大仓位比例

    返回:
        最大可投入金额（元）
    """
    pass  # TODO: 实现最大仓位计算


def check_industry_exposure(industry_exposure: Dict[str, float],
                            new_industry: str,
                            max_pct: float = MAX_INDUSTRY_PCT) -> bool:
    """
    检查行业集中度是否超标

    逻辑：
        - 如果已有 25% 仓位在"银行"行业
        - 新信号又是银行股 → 即使没到 5 只上限，也拒绝

    参数:
        industry_exposure: 当前各行业仓位占比 dict
        new_industry: 新信号的股票所属行业
        max_pct: 行业最大占比

    返回:
        True = 可以买入, False = 行业超限
    """
    pass  # TODO: 实现行业集中度检查


# ============================================================
# 2. 止损逻辑（事中控制）
# ============================================================

def check_stop_loss(avg_cost: float, current_price: float,
                    stop_loss_pct: float = STOP_LOSS_PCT) -> Tuple[bool, str]:
    """
    检查固定比例止损

    逻辑：
        - 当前亏损 = (current_price / avg_cost) - 1
        - 如果亏损幅度 >= STOP_LOSS_PCT → 触发止损

    为什么 5% 止损？
        - 亏 5% 需要赚 5.26% 才能回本
        - 亏 10% 需要赚 11.1% 才能回本
        - 亏 20% 需要赚 25% 才能回本
        → 控制亏损比追求盈利更重要

    参数:
        avg_cost: 平均成本价
        current_price: 当前价
        stop_loss_pct: 止损阈值（小数，如 0.05 = 5%）

    返回:
        (是否触发止损, 止损原因描述)
    """
    pass  # TODO: 实现固定止损检查


def check_trailing_stop(avg_cost: float, current_price: float,
                        highest_price: float,
                        trailing_pct: float = TRAILING_STOP_PCT) -> Tuple[bool, str]:
    """
    检查移动止损 (Trailing Stop)

    逻辑：
        - 记录持仓期间达到的最高价
        - 当前价从最高价回落 trailing_pct% → 触发止损
        - 最高价不断上移 → 止损线也不断上移

    举例：
        - 10 元买入，最高涨到 12 元（盈利 20%）
        - 现在跌到 11.64 元（从 12 回落 3%）
        - → 触发移动止损，锁定 16.4% 利润

    移动止损 vs 固定止损：
        - 固定止损：始终设亏损 5% 线 → 保护本金
        - 移动止损：跟随价格上移 → 保护利润
        - 两者配合使用

    参数:
        avg_cost: 成本价
        current_price: 当前价
        highest_price: 持仓期间最高价
        trailing_pct: 回撤比例

    返回:
        (是否触发移动止损, 原因描述)
    """
    pass  # TODO: 实现移动止损检查


def check_time_stop(buy_date: str, current_date: str,
                    max_days: int = MAX_HOLDING_DAYS,
                    current_return: float = 0) -> Tuple[bool, str]:
    """
    检查时间止损

    逻辑：
        - 持仓时间超过 max_days → 如果还没盈利，考虑卖出
        - "时间就是金钱"：资金被套在横盘股里 → 失去买入其他牛股的机会
        - 这是机会成本 (Opportunity Cost)

    参数:
        buy_date: 买入日期
        current_date: 当前日期
        max_days: 最大持有天数
        current_return: 当前收益率

    返回:
        (是否触发时间止损, 原因)
    """
    pass  # TODO: 实现时间止损检查


# ============================================================
# 3. 熔断机制（事后控制）
# ============================================================

def check_daily_loss_limit(daily_pnl: float, total_capital: float,
                           max_loss_pct: float = MAX_DAILY_LOSS_PCT) -> Tuple[bool, str]:
    """
    检查单日亏损是否触发熔断

    逻辑：
        - 单日亏损超过总资金 5% → 触发熔断
        - 熔断后：暂停新开仓，只允许平仓
        - 需要人工检查才能恢复

    为什么需要熔断？
        - 如果一天亏 5%，可能是：
          a. 模型失效（市场风格突变）
          b. 数据错误（取到了脏数据）
          c. 程序 bug
        - 无论哪种原因，先停下来比继续赌要好

    参数:
        daily_pnl: 当日盈亏（负值 = 亏损）
        total_capital: 总资金
        max_loss_pct: 熔断阈值

    返回:
        (是否触发熔断, 原因)
    """
    pass  # TODO: 实现熔断检查


def check_consecutive_losses(pnl_history: list) -> Tuple[bool, str]:
    """
    检查是否连续亏损

    参数:
        pnl_history: 最近 N 天的盈亏列表

    返回:
        (是否触发预警, 原因)
    """
    pass  # TODO: 实现连续亏损检查


# ============================================================
# 4. 综合风控决策
# ============================================================

def risk_check(avg_cost: float, current_price: float, highest_price: float,
               buy_date: str, current_date: str,
               current_positions: int, total_capital: float,
               daily_pnl: float,
               industry_exposure: Dict[str, float] = None,
               new_industry: str = None) -> Dict[str, Tuple[bool, str]]:
    """
    【主函数】综合风控检查

    在每次需要考虑买卖时调用，返回所有风控规则的结果。

    返回值示例：
        {
            "position_limit": (True, "当前3/5仓位，可以开仓"),
            "stop_loss": (False, "当前亏损2.3%，未触发止损"),
            "trailing_stop": (False, "未触发移动止损"),
            "time_stop": (False, ""),
            "daily_loss_limit": (False, ""),
            "circuit_breaker": (False, ""),
        }

    参数:
        （见各子函数）

    返回:
        Dict[规则名 → (是否触发, 原因描述)]
    """
    pass  # TODO: 实现综合风控检查


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("风控模块 — 三层防护")
    print("  第一层：仓位管理（单票≤20%, 总数≤5, 行业≤30%）")
    print("  第二层：止损逻辑（固定5% + 移动3% + 时间60天）")
    print("  第三层：熔断机制（单日亏5%暂停, 连续5天亏损预警）")
    print("设计理念：没有风控的回测报告是废纸")
