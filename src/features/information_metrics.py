"""
信息量度模块 — 信息熵防火墙

作用：
    在特征进入模型之前，先评估每个特征包含多少"有效信息"。
    低信息量的特征 → 直接过滤掉，从源头控制维度灾难。

核心理念（对应思维导图第 2 节的"防火墙逻辑"）：
    - 不是所有特征都有用，加无用特征 = 增加噪音
    - 用信息熵和互信息量来量化"这个特征到底有没有信息"
    - 信息量低于阈值的特征 → 不进入模型

两个核心指标：
    1. 信息熵 (Information Entropy)
       → 衡量特征本身的"不确定性"或"信息量"
       → 如果一个特征几乎不变（如退市股票的价格恒为0），熵极低 → 无信息量

    2. 互信息量 (Mutual Information)
       → 衡量特征与目标变量（次日涨跌）之间的关联度
       → 如果特征和目标完全独立，互信息 ≈ 0 → 该特征对预测无用

为什么需要这个模块：
    - 你构造了 20+ 个因子，但不是每个都对预测有帮助
    - 无用的因子不仅浪费计算时间，还会增加模型过拟合风险
    - 信息熵 + 互信息量 = 客观的衡量标准，不是拍脑袋决定

依赖：
    - scikit-learn (mutual_info_classif, mutual_info_regression)
    - scipy (entropy)

参考：docs/factor_catalog.md（因子定义）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from scipy import stats


# ============================================================
# 配置
# ============================================================

# 信息量阈值
MIN_ENTROPY_PERCENTILE = 5       # 熵低于 5% 分位数的特征 → 标记为低信息量
MIN_MUTUAL_INFO = 0.001          # 互信息量低于此值 → 标记为低关联度


# ============================================================
# 1. 信息熵计算
# ============================================================

def calc_entropy(series: pd.Series, bins: int = 50) -> float:
    """
    计算一个特征的信息熵

    公式:
        H(X) = -Σ p(x_i) * log2(p(x_i))

    其中 p(x_i) 是值落在第 i 个区间（bin）的概率。

    为什么要分箱（bins）？
        - 连续变量的每个值几乎都不同 → 每个 p ≈ 1/n → 熵 ≈ log(n)，没意义
        - 分箱后把相近的值归为一类 → 熵反映的是"分布的均匀程度"

    解读：
        - 高熵 → 特征值分布很均匀 → 信息量大
        - 低熵 → 特征值集中在少数 bin 里 → 信息量小（比如 90% 的值都一样）

    参数:
        series: 特征序列（如某因子的所有值）
        bins: 分箱数量

    返回:
        float，信息熵值（单位：bit）
    """
    pass  # TODO: 实现信息熵计算


def calc_all_entropies(df: pd.DataFrame, factor_columns: List[str]) -> pd.Series:
    """
    计算所有因子的信息熵

    参数:
        df: 含全部因子列的 DataFrame
        factor_columns: 因子列名列表

    返回:
        Series，index=因子名，value=信息熵
    """
    pass  # TODO: 实现批量信息熵计算


# ============================================================
# 2. 互信息量计算
# ============================================================

def calc_mutual_info(df: pd.DataFrame, factor_columns: List[str],
                     target_column: str) -> pd.Series:
    """
    计算每个因子与目标变量的互信息量

    什么是互信息量？
        - MI(X; Y) 衡量"知道了 X，你对 Y 的不确定性减少多少"
        - MI = 0 → X 和 Y 完全独立（X 对预测 Y 毫无帮助）
        - MI 越大 → X 对 Y 的预测能力越强
        - 比相关系数更强大：能捕捉非线性关系（相关系数只能捕捉线性关系）

    公式:
        MI(X; Y) = Σ p(x,y) * log(p(x,y) / (p(x)*p(y)))

    为什么用互信息量而不是相关系数？
        - 金融数据中很多关系是非线性的
        - 例如：RSI 在 30 和 70 附近与涨跌的关系强，但在 50 附近弱
          → 这形成一个 U 型关系，相关系数 ≈ 0，但互信息量能检测到

    参数:
        df: 含因子列和目标列的 DataFrame
        factor_columns: 因子列名列表
        target_column: 目标变量列名（如 'next_day_up'，0/1 二分类）

    返回:
        Series，index=因子名，value=互信息量
    """
    pass  # TODO: 实现互信息量批量计算


# ============================================================
# 3. 信息量综合评估
# ============================================================

def evaluate_factor_quality(df: pd.DataFrame, factor_columns: List[str],
                            target_column: str) -> pd.DataFrame:
    """
    综合评估因子质量：信息熵 + 互信息量

    输出一个 DataFrame，包含每个因子的：
        - entropy: 信息熵
        - entropy_percentile: 熵在所有因子中的分位数
        - mutual_info: 互信息量
        - is_low_entropy: 是否为低信息熵因子
        - is_low_mi: 是否为低互信息量因子
        - recommendation: 建议（keep/filter/review）

    参数:
        df: 含因子和目标变量的 DataFrame
        factor_columns: 因子列名列表
        target_column: 目标变量列名

    返回:
        DataFrame，每个因子一行，含评估指标和建议
    """
    pass  # TODO: 实现因子质量综合评估


# ============================================================
# 4. 防火墙逻辑
# ============================================================

def firewall_filter(df: pd.DataFrame, factor_columns: List[str],
                    target_column: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    【主函数】信息熵防火墙

    规则：
        1. 信息熵低于 5% 分位数 → 标记为低信息量
        2. 互信息量低于 MIN_MUTUAL_INFO → 标记为低关联度
        3. 两项都低 → 直接过滤（不进入模型）
        4. 一项低 → 标记为 "review"（人工判断是否保留）

    为什么叫"防火墙"？
        - 类比网络安全：防火墙在数据进入系统之前就拦截威胁
        - 这里的"威胁" = 低质量特征会增加过拟合风险
        - 在特征进入模型之前就过滤掉，而不是等训练完才发现

    参数:
        df: 含因子和目标变量的 DataFrame
        factor_columns: 待评估的因子列名列表
        target_column: 目标变量列名

    返回:
        (过滤后的 DataFrame, 保留的因子列名, 被过滤的因子列名)
    """
    pass  # TODO: 实现防火墙过滤逻辑


# ============================================================
# 5. 可视化辅助
# ============================================================

def plot_factor_quality_report(quality_df: pd.DataFrame, output_path: str = None):
    """
    生成因子质量可视化报告

    包含：
        - 信息熵分布直方图
        - 互信息量排名条形图
        - 熵 vs 互信息量散点图（标注过滤线）

    参数:
        quality_df: evaluate_factor_quality() 的输出
        output_path: 图表保存路径（None = 显示）
    """
    pass  # TODO: 实现因子质量可视化


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("信息熵防火墙模块")
    print("用法：从特征工程管线中调用 firewall_filter()")
    print("参考：docs/factor_catalog.md 中的因子评估标准")
