"""
实验追踪模块 — 记录和管理模型训练实验

作用：
    每次训练模型时自动记录：用了什么特征、什么超参、训练集范围、结果如何。
    → 一周后还能知道哪组参数跑出最好的结果。

为什么需要这个模块？
    场景：你跑完 10 组实验，3 天后想对比时：
        ❌ 没有实验追踪 → "我记得有一组 AUC 挺高的…是 ret_20d + rsi 那组吧？不对，好像是加了 MACD…"
        ✅ 有实验追踪 → 打开 experiments.csv，排序 AUC，一目了然

设计方案（MVP 阶段用 CSV，不需要 MLflow）：
    - 每次实验一行记录
    - 用 CSV 存储（轻量、人类可读、GitHub 友好）
    - 保留所有关键参数和结果指标

字段说明见 docs/data_dictionary.md → 实验记录表

依赖：
    - pandas
    - src/models/train.py（上游，提供训练好的模型和元数据）

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


# ============================================================
# 配置
# ============================================================

EXPERIMENT_LOG_PATH = "data/experiments.csv"


# ============================================================
# 1. 实验记录 CRUD
# ============================================================

def generate_exp_id() -> str:
    """
    生成唯一的实验编号

    格式: YYYYMMDD_HHMMSS
    例如: 20260715_153001
    """
    pass  # TODO: 实现实验ID生成


def log_experiment(features: List[str],
                   model_params: Dict,
                   train_dates: tuple,
                   test_dates: tuple,
                   metrics: Dict,
                   notes: str = "") -> str:
    """
    记录一次训练实验

    参数:
        features: 使用的特征列名列表
        model_params: 使用的超参 dict
        train_dates: (train_start, train_end) 训练集日期范围
        test_dates: (test_start, test_end) 测试集日期范围
        metrics: 评估指标 dict（accuracy, auc, sharpe, max_drawdown 等）
        notes: 备注（如"第一次尝试"、"加入了MACD因子"）

    返回:
        实验 ID
    """
    pass  # TODO: 实现实验记录


def load_experiments() -> pd.DataFrame:
    """
    加载所有历史实验记录
    """
    pass  # TODO: 实现实验加载


def get_best_experiment(metric: str = "auc", top_n: int = 5) -> pd.DataFrame:
    """
    获取指定指标排名前 N 的实验

    参数:
        metric: 排序指标（accuracy/auc/sharpe/max_drawdown）
        top_n: 返回前几名

    返回:
        DataFrame，按指标降序（或升序，如 max_drawdown 越低越好）
    """
    pass  # TODO: 实现最佳实验查询


# ============================================================
# 2. 实验对比
# ============================================================

def compare_experiments(exp_id_1: str, exp_id_2: str) -> str:
    """
    对比两次实验的差异

    输出：
        - 特征差异（多了什么、少了什么）
        - 超参差异
        - 指标差异（涨了还是跌了）

    参数:
        exp_id_1, exp_id_2: 两次实验的 ID

    返回:
        Markdown 格式的对比报告
    """
    pass  # TODO: 实现实验对比


# ============================================================
# 3. 实验总结报告
# ============================================================

def generate_experiment_summary() -> str:
    """
    生成所有实验的总结报告

    内容：
        - 总共跑了多少次实验
        - 指标趋势（AUC/夏普 在变好还是变差）
        - Top 3 最佳实验详情
        - 最常用的 Top 10 特征（出现次数最多的）
    """
    pass  # TODO: 实现实验总结


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("实验追踪模块")
    print("用法：每次训练后调用 log_experiment() 自动记录")
    print(f"实验记录保存在: {EXPERIMENT_LOG_PATH}")
