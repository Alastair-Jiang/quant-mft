"""
模型评估模块 — 多维度评估预测模型表现

作用：
    全面评估训练好的 LightGBM 模型的预测能力和泛化能力。
    不止看准确率——在量化场景下，AUC、混淆矩阵、特征重要性比准确率更重要。

评估维度：
    1. 分类指标 → 准确率、精确率、召回率、F1、AUC
    2. 混淆矩阵 → 看预测涨/跌的分布（是否存在严重偏向）
    3. 复杂度惩罚 → AIC/BIC（防止无意义地堆参数）
    4. 特征重要性 → 哪些因子贡献最大（指导因子筛选）
    5. 分组评估 → 按时间段/行业/市值分组看表现（检测过拟合）

⚠️ 心理建设：
    日线涨跌预测的准确率 50-53% 是正常的。
    比扔硬币（50%）好一点就足够了——在交易中用仓位管理放大优势。

依赖：
    - scikit-learn (metrics, confusion_matrix, roc_auc_score 等)
    - lightgbm (feature_importances_)
    - matplotlib (可视化)
    - src/models/train.py（上游，提供训练好的模型）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from sklearn import metrics


# ============================================================
# 1. 分类指标计算
# ============================================================

def calc_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: np.ndarray) -> Dict[str, float]:
    """
    计算分类任务的全部核心指标

    指标说明：
        - Accuracy (准确率): 预测对的样本 / 总样本
          → 最直观但不一定最有用的指标
        - Precision (精确率): 预测为"涨"且实际涨 / 预测为"涨"
          → "你推荐买的股票里，真的涨了的比例"
        - Recall (召回率): 预测为"涨"且实际涨 / 实际涨
          → "实际涨的股票里，你抓住了多少"
        - F1 Score: Precision 和 Recall 的调和平均
        - AUC (ROC曲线下面积): 模型对正负样本的排序能力
          → 量化交易中最重要的分类指标！
          → AUC > 0.5 = 比随机好，AUC = 1 = 完美

    为什么 AUC 最重要？
        - AUC 衡量的是模型的"排序能力"（好股票排在坏股票前面）
        - 交易不是买所有预测涨的股票，而是买预测概率最高的前 N 只
        - AUC 高 = 概率排在前面的股票确实表现更好

    参数:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签 (0/1)
        y_prob: 预测概率 (0~1 float)

    返回:
        Dict，如 {"accuracy": 0.52, "auc": 0.55, ...}
    """
    pass  # TODO: 实现分类指标计算


def calc_multiclass_metrics(y_true_multiclass: np.ndarray,
                            y_prob_matrix: np.ndarray,
                            y_true_binary: np.ndarray = None) -> Dict[str, float]:
    """
    评估多分类分布输出模型 (P2补丁: 分布输出评估)

    除了标准分类指标外，额外计算:
        - direction_accuracy: 方向预测准确率（涨 vs 跌，忽略幅度）
        - mean_entropy: 所有预测的平均熵（越低表示模型越确定）
        - calibration_error: 预测概率与实际频率的偏差
          （例如: 模型说"80%概率涨"的那些样本中，实际涨的比例是多少？）

    参数:
        y_true_multiclass: 真实类别标签 (0~5)
        y_prob_matrix: (n_samples, n_bins) 预测概率矩阵
        y_true_binary: 二分类标签 (0/1)，如果为None则从y_true_multiclass转换

    返回:
        Dict，含 multi_logloss, direction_accuracy, mean_entropy, calibration_error
    """
    pass  # TODO: 实现多分类评估指标



# ============================================================
# 2. 混淆矩阵
# ============================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str = None) -> np.ndarray:
    """
    绘制混淆矩阵

    混淆矩阵结构：             预测跌  预测涨
                   实际跌  [  TN      FP  ]
                   实际涨  [  FN      TP  ]

    - TN (True Negative): 预测跌，实际跌 → 正确不买
    - FP (False Positive): 预测涨，实际跌 → 错误买入（亏钱！）
    - FN (False Negative): 预测跌，实际涨 → 错过机会
    - TP (True Positive): 预测涨，实际涨 → 正确买入（赚钱！）

    我们最关心的：
        - FP 率要低（买错了 = 亏真金白银）
        - TP 率要高（买对了 = 赚钱）

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 图表保存路径

    返回:
        混淆矩阵 (2×2 numpy array)
    """
    pass  # TODO: 实现混淆矩阵可视化


# ============================================================
# 3. 模型复杂度惩罚 (AIC / BIC)
# ============================================================

def calc_aic_bic(n_samples: int, n_features: int, log_likelihood: float) -> Dict[str, float]:
    """
    计算 AIC 和 BIC 模型复杂度惩罚指标

    为什么需要 AIC/BIC？
        - 你加一个特征，准确率可能从 51% → 51.1%，看起来"提升"了
        - 但这可能是随机波动，而不是真正的改进
        - AIC/BIC 会在"拟合优度"和"模型复杂度"之间做权衡
        - 如果新特征带来的增量不如惩罚项大，AIC/BIC 会变差

    公式:
        AIC = 2*k - 2*ln(L)
        BIC = k*ln(n) - 2*ln(L)
        其中 k = 模型参数数量, n = 样本数, L = 似然函数值

    解读：
        - AIC/BIC 越低越好（越低 = 用更少的参数达到更好的拟合）
        - BIC 对复杂度的惩罚比 AIC 更重（在 n>7 时）

    参数:
        n_samples: 样本数量
        n_features: 特征数量（近似模型参数数量）
        log_likelihood: 对数似然函数值

    返回:
        {"aic": float, "bic": float}
    """
    pass  # TODO: 实现 AIC/BIC 计算


# ============================================================
# 4. 特征重要性分析
# ============================================================

def analyze_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    分析特征重要性

    LightGBM 提供两种重要性：
        - split: 该特征被用作分裂节点的次数（默认）
        - gain: 该特征带来的平均信息增益（更有意义！）
          → gain 高的特征 = 对预测贡献大

    为什么要分析特征重要性？
        - 看到哪个因子真正在驱动预测
        - 如果重要性排第一的因子是 ret_1d（昨日涨跌）→ 模型只是"追涨"
        - 如果 top 因子有合理的金融解释 → 模型学到的是真实信号

    参数:
        model: 训练好的 LightGBM Booster
        feature_names: 特征名列表

    返回:
        DataFrame，列: feature, gain_importance, split_importance
    """
    pass  # TODO: 实现特征重要性分析


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20,
                            save_path: str = None):
    """
    绘制特征重要性条形图（Top N）
    """
    pass  # TODO: 实现特征重要性可视化


# ============================================================
# 5. 分组评估（检测过拟合）
# ============================================================

def evaluate_by_time_period(y_true: np.ndarray, y_prob: np.ndarray,
                            dates: pd.Series, freq: str = "M") -> pd.DataFrame:
    """
    按时间分组评估（检测模型在特定时间段是否失效）

    例如：
        - 训练集（2024年）：AUC = 0.56
        - 测试集（2025年）：AUC = 0.53 → 轻微衰减，正常
        - 测试集（2026年Q1）：AUC = 0.49 → 可能概念漂移

    参数:
        y_true, y_prob: 标签和预测概率
        dates: 每个样本对应的日期
        freq: 分组频率 ("M"=月, "Q"=季度, "Y"=年)

    返回:
        DataFrame，每段时间的 AUC
    """
    pass  # TODO: 实现按时间分组评估


# ============================================================
# 6. 综合评估报告
# ============================================================

def generate_evaluation_report(model, X_test: pd.DataFrame, y_test: np.ndarray,
                               feature_names: List[str],
                               test_dates: pd.Series = None) -> str:
    """
    【主函数】生成完整的模型评估报告 (Markdown)

    报告内容：
        1. 分类指标汇总表
        2. 混淆矩阵
        3. AIC/BIC 复杂度评估
        4. Top 20 特征重要性排名
        5. 按时间段 AUC 变化趋势
        6. 与基准（随机猜测 50%）的对比

    参数:
        model: 训练好的 LightGBM
        X_test: 测试集特征
        y_test: 测试集标签
        feature_names: 特征名列表
        test_dates: 测试集日期（可选，用于分组评估）

    返回:
        Markdown 格式的评估报告
    """
    pass  # TODO: 实现综合评估报告


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("模型评估模块")
    print("用法：generate_evaluation_report(model, X_test, y_test, feature_names)")
    print("⚠️ 预期准确率 50-53%，不要因为'只比随机好一点'就觉得失败")
