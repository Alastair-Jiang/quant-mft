"""
过拟合诊断模块 — 蒙特卡洛参数扰动测试

作用：
    检测策略是否对参数过度敏感 → 判断过拟合程度。
    回测曲线完美 ≠ 策略好，可能是过拟合。

核心理念（对应思维导图第5节）：
    - 如果策略是真正有效的 → 微调参数后表现应该类似
    - 如果微调参数后策略崩溃 → 原始参数被过度优化了（过拟合）

诊断流程：
    1. 用当前最优参数跑一次回测 → 得到基准夏普比率
    2. 对每个关键参数做 ±20% 随机扰动 → 跑 N 次回测
    3. 如果扰动后夏普下降 > 50% → 判定为过拟合
    4. 生成诊断报告 → 给出修复建议

过拟合量化阈值：
    - 夏普 > 3.0 → 🟡 可疑（日线策略极难达到）
    - 夏普 > 4.0 → 🔴 几乎确定过拟合
    - 年化 > 50% 且最大回撤 < 5% → 🔴 不可能三角
    - 参数 ±20% 扰动 → 夏普下降 > 50% → 过拟合

依赖：
    - numpy (随机采样)
    - src/backtest/engine.py（回测引擎，用于跑扰动后的回测）

作者: 蒋东旭
日期: 2026-07-15
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable


# ============================================================
# 配置
# ============================================================

# 过拟合检测阈值
SHARPE_SUSPICIOUS = 3.0        # 夏普超过此值 → 可疑
SHARPE_DEFINITE = 4.0          # 夏普超过此值 → 几乎确定过拟合
ANNUAL_RETURN_MAX = 0.5        # 年化收益上限
MAX_DRAWDOWN_MIN = 0.05        # 最大回撤下限（回撤小于此值+收益超高=不可能）

# 蒙特卡洛参数
MONTE_CARLO_N = 1000            # 模拟次数
PERTURBATION_RANGE = 0.2        # 参数扰动范围 (±20%)
DEGRADATION_THRESHOLD = 0.5     # 夏普下降超过 50% → 判定过拟合


# ============================================================
# 1. 过拟合预警检查
# ============================================================

def check_overfitting_signals(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    检查回测指标是否触发过拟合预警

    预警条件：
        1. 夏普 > 3.0 → 🟡 WARNING
        2. 夏普 > 4.0 → 🔴 CRITICAL
        3. 年化 > 50% 且回撤 < 5% → 🔴 不可能三角
        4. 年化 > 30% 且回撤 < 10% → 🟡 可疑

    参数:
        metrics: calc_backtest_metrics() 的输出

    返回:
        {"sharpe_warning": "OK"|"SUSPICIOUS"|"CRITICAL",
         "impossible_triangle": "OK"|"WARNING", ...}
    """
    pass  # TODO: 实现过拟合信号检查


# ============================================================
# 2. 蒙特卡洛参数扰动
# ============================================================

def perturb_params(base_params: Dict[str, float],
                   perturbation_range: float = PERTURBATION_RANGE) -> Dict[str, float]:
    """
    对参数做随机扰动

    扰动方式：
        perturbed = base * (1 + uniform(-range, +range))
        例如：learning_rate=0.01, range=0.2
             → perturbed ∈ [0.008, 0.012]

    哪些参数需要扰动？
        - 模型超参：learning_rate, num_leaves, min_child_samples
        - 特征窗口：ma_windows, rsi_window
        - 交易参数：stop_loss_pct, max_position_pct, commission_rate

    参数:
        base_params: 基准参数 dict
        perturbation_range: 扰动范围（±20%）

    返回:
        扰动后的参数 dict
    """
    pass  # TODO: 实现参数随机扰动


def run_monte_carlo_test(backtest_fn: Callable,
                         base_params: Dict[str, float],
                         param_names: List[str],
                         n_simulations: int = MONTE_CARLO_N) -> pd.DataFrame:
    """
    执行蒙特卡洛参数扰动测试

    流程：
        1. 用基准参数跑一次 → 得到基准夏普
        2. 重复 N 次：
           a. 随机扰动参数
           b. 用扰动后参数跑一次回测
           c. 记录扰动后夏普
        3. 统计：扰动后夏普下降的比例、分布

    参数:
        backtest_fn: 回测函数（签名为 fn(params) -> metrics_dict）
        base_params: 基准参数
        param_names: 要扰动的参数名列表
        n_simulations: 模拟次数

    返回:
        DataFrame, 每次模拟的扰动参数和对应的夏普
    """
    pass  # TODO: 实现蒙特卡洛测试


def diagnose_overfitting(monte_carlo_results: pd.DataFrame,
                         base_sharpe: float,
                         degradation_threshold: float = DEGRADATION_THRESHOLD) -> Dict:
    """
    根据蒙特卡洛结果诊断过拟合程度

    判定规则：
        1. 扰动后夏普下降 > degradation_threshold (50%) 的模拟次数 > 30%
           → 过拟合（策略对参数高度敏感）

        2. 扰动后夏普的均值与基准夏普接近（±10%）
           → 稳健（策略对参数不敏感）

        3. 扰动后夏普分布很宽（标准差 > 基准夏普的 50%）
           → 不稳定（策略在部分参数下会失效）

    参数:
        monte_carlo_results: 蒙特卡洛结果
        base_sharpe: 基准夏普
        degradation_threshold: 夏普下降判定阈值

    返回:
        诊断结果 dict
    """
    pass  # TODO: 实现过拟合诊断


# ============================================================
# 3. 负反馈闭环
# ============================================================

def generate_fix_suggestions(diagnosis: Dict) -> List[str]:
    """
    根据诊断结果生成修复建议

    修复策略（按优先级）：
        1. 特征侧：减少特征数量（移除 IC 最低的 30% 因子）
        2. 模型侧：增强正则化（L1/L2 系数 ×2）
        3. 模型侧：降低树复杂度（num_leaves 减半）
        4. 数据侧：增加训练数据长度（如果可用）
        5. 交易侧：增加交易成本估算（更保守）

    参数:
        diagnosis: diagnose_overfitting() 的输出

    返回:
        修复建议列表（按优先级排序）
    """
    pass  # TODO: 实现修复建议生成


# ============================================================
# 4. 综合诊断报告
# ============================================================

def run_diagnostic_pipeline(metrics: Dict[str, float],
                            backtest_fn: Callable,
                            base_params: Dict[str, float],
                            param_names: List[str]) -> str:
    """
    【主函数】完整的过拟合诊断管线

    流程：
        1. check_overfitting_signals() → 检查预警信号
        2. 如果触发预警 → run_monte_carlo_test() → 深入诊断
        3. diagnose_overfitting() → 判定过拟合程度
        4. generate_fix_suggestions() → 提供修复方案

    参数:
        metrics: 基准回测指标
        backtest_fn: 回测函数
        base_params: 基准参数
        param_names: 要扰动的参数名

    返回:
        Markdown 格式的诊断报告
    """
    pass  # TODO: 实现完整诊断管线


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("过拟合诊断模块")
    print("核心逻辑：参数微调 → 策略是否崩溃？")
    print("阈值：夏普>3 可疑, 夏普>4 几乎确定过拟合")
    print("参考：docs/architecture.md 第5节")
