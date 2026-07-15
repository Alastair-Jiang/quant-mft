"""
因子筛选与降维模块

作用：
    从 20+ 个候选因子中选出最优子集，减少冗余、降低维度。

为什么需要因子筛选？
    - 不是因子越多越好——冗余因子增加噪音、加剧过拟合
    - 两个高度相关的因子（如 ret_5d 和 ma_dev_5）→ 只需要保留一个
    - 高维特征空间 → 模型训练变慢、需要更多数据（维度灾难）

筛选方法（按使用顺序）：
    1. IC 分析 (Information Coefficient)
       → 因子与未来收益的相关性
       → |IC| < 0.02 → 因子几乎没有预测能力 → 剔除

    2. 因子互相关性矩阵
       → 两个因子相关系数 > 0.8 → 二选一保留（保留 IC 更高的）

    3. PCA 降维（可选，P1）
       → 保留 90% 方差的最小维度
       → 如果 10 个因子可以解释 90% 的方差，就只用 10 个主成分

依赖：
    - pandas, numpy, scikit-learn
    - src/features/alpha_factors.py（上游，提供因子数据）
    - src/features/information_metrics.py（上游，提供信息量评估）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# 配置
# ============================================================

MIN_ABS_IC = 0.02               # 最小 IC 绝对值
MAX_CORRELATION = 0.8           # 因子间最大相关系数
PCA_VARIANCE_THRESHOLD = 0.90   # PCA 保留方差比例


# ============================================================
# 1. IC 分析 (Information Coefficient)
# ============================================================

def calc_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """
    计算单个因子的 IC (Information Coefficient)

    什么是 IC？
        - IC = 因子值 与 未来收益率 的 Spearman 秩相关系数
        - Spearman 而非 Pearson：只关心排序关系，不关心具体数值
        - IC ∈ [-1, 1]

    解读：
        - |IC| > 0.05 → 强预测因子（很少见）
        - |IC| ∈ [0.02, 0.05] → 中等预测因子（量化常用范围）
        - |IC| < 0.02 → 弱预测因子（考虑剔除）
        - IC ≈ 0 → 随机噪声，无预测能力

    为什么用 Spearman 而不是 Pearson？
        - Pearson 只能检测线性关系
        - Spearman 检测的是"单调关系"（A 大 → B 大，但不一定是直线）
        - 量化因子通常不要求精确的线性关系，只关心排序

    参数:
        factor_values: 因子值序列（某时间截面上的因子值）
        forward_returns: 对应的未来收益率序列（如次日收益率）

    返回:
        float，IC 值
    """
    pass  # TODO: 实现 IC 计算（Spearman 秩相关）


def calc_ic_summary(df: pd.DataFrame, factor_columns: List[str],
                    forward_return_column: str) -> pd.DataFrame:
    """
    计算所有因子的 IC 统计量

    输出每个因子的：
        - ic_mean: IC 均值（预测能力的平均水平）
        - ic_std: IC 标准差（预测能力的稳定性）
        - ic_ir: IC 的信息比率 = ic_mean / ic_std（越高越好，> 0.5 为佳）
        - abs_ic_mean: |IC| 均值

    参数:
        df: 含因子列和未来收益列的 DataFrame
        factor_columns: 因子列名列表
        forward_return_column: 未来收益列名

    返回:
        DataFrame，每个因子一行，含 IC 统计量
    """
    pass  # TODO: 实现 IC 汇总统计


def filter_by_ic(df: pd.DataFrame, factor_columns: List[str],
                 forward_return_column: str,
                 min_abs_ic: float = MIN_ABS_IC) -> Tuple[List[str], List[str]]:
    """
    基于 IC 筛选因子

    规则：|IC_mean| < min_abs_ic → 剔除

    参数:
        df: 数据
        factor_columns: 候选因子列名
        forward_return_column: 未来收益列名
        min_abs_ic: IC 阈值

    返回:
        (保留的因子列名列表, 被剔除的因子列名列表)
    """
    pass  # TODO: 实现 IC 筛选


# ============================================================
# 2. 因子互相关性分析
# ============================================================

def calc_factor_correlation(df: pd.DataFrame, factor_columns: List[str]) -> pd.DataFrame:
    """
    计算因子间的 Spearman 秩相关矩阵

    为什么是互相关性而不是自相关性？
        - 自相关 = 因子自己跟自己过去的关系（用于时序检查）
        - 互相关 = 两个不同因子之间的关系（用于去重）
        - 这里需要的是互相关

    参数:
        df: 含因子列的 DataFrame
        factor_columns: 因子列名列表

    返回:
        DataFrame，N×N 的相关系数矩阵
    """
    pass  # TODO: 实现因子互相关矩阵


def remove_highly_correlated(df: pd.DataFrame, factor_columns: List[str],
                             ic_summary: pd.DataFrame,
                             max_corr: float = MAX_CORRELATION) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    剔除高度相关的冗余因子

    规则：
        1. 找到相关系数 > max_corr 的因子对
        2. 对于每对冗余因子，保留 |IC| 更高的那个
        3. 剔除 |IC| 较低的那个

    举例：
        ret_5d 和 ma_dev_5 相关系数 = 0.92 → 高度冗余
        ret_5d 的 IC = 0.03, ma_dev_5 的 IC = 0.02
        → 保留 ret_5d，剔除 ma_dev_5

    参数:
        df: 含因子列的 DataFrame
        factor_columns: 候选因子列名
        ic_summary: calc_ic_summary() 的输出
        max_corr: 相关系数阈值

    返回:
        (保留的因子列名列表, 剔除详情 [(因子A, 因子B, 相关系数), ...])
    """
    pass  # TODO: 实现冗余因子剔除


# ============================================================
# 3. PCA 降维（备选，P1）
# ============================================================

def apply_pca(df: pd.DataFrame, factor_columns: List[str],
              variance_threshold: float = PCA_VARIANCE_THRESHOLD) -> Tuple[pd.DataFrame, int]:
    """
    用 PCA 对因子进行降维

    原理：
        - PCA 找到因子空间中方差最大的方向（主成分）
        - 每个主成分是原始因子的线性组合
        - 只保留能解释 variance_threshold（如 90%）方差的前 K 个主成分

    优点：
        - 自动去相关（主成分之间完全正交）
        - 保留最多信息的同时降低维度

    缺点：
        - 主成分不可解释（你不知道 PC1 是什么金融含义）
        - 不适合需要因子归因的场景

    参数:
        df: 含因子列的 DataFrame
        factor_columns: 因子列名列表
        variance_threshold: 保留方差比例

    返回:
        (降维后的 DataFrame（PCA 主成分）, 保留的主成分数量)
    """
    pass  # TODO: 实现 PCA 降维


# ============================================================
# 4. 综合因子筛选管线
# ============================================================

def select_factors(df: pd.DataFrame, factor_columns: List[str],
                   forward_return_column: str,
                   use_pca: bool = False) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    【主函数】完整的因子筛选管线

    流程：
        1. 信息熵防火墙 → 过滤低信息量特征
           (调用 information_metrics.firewall_filter())
        2. IC 分析 → 过滤低 IC 因子
           (调用 filter_by_ic())
        3. 互相关性去重 → 过滤冗余因子
           (调用 remove_highly_correlated())
        4. [可选] PCA 降维
           (调用 apply_pca())

    参数:
        df: 含因子和目标变量的 DataFrame
        factor_columns: 所有候选因子列名
        forward_return_column: 未来收益列名
        use_pca: 是否使用 PCA 降维

    返回:
        (筛选后的 DataFrame, 最终保留的列名, 筛选报告 dict)
    """
    pass  # TODO: 实现完整因子筛选管线


def generate_selection_report(initial_columns: List[str],
                              final_columns: List[str],
                              removed_breakdown: dict) -> str:
    """
    生成因子筛选报告（Markdown 格式）

    内容包括：
        - 初始因子数 → 最终因子数
        - 每个阶段过滤了多少因子、哪些因子被过滤
        - 过滤原因

    参数:
        initial_columns: 初始因子列名
        final_columns: 最终保留的因子列名
        removed_breakdown: 过滤详情 dict

    返回:
        Markdown 格式的报告字符串
    """
    pass  # TODO: 实现筛选报告生成


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("因子筛选与降维模块")
    print("用法：select_factors(df, factor_columns, 'next_day_return')")
    print("参考：docs/factor_catalog.md 中的因子评估标准")
