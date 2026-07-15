"""
模型训练模块 — LightGBM 多模式预测 + 分布输出 + Skip-1-day

作用：
    用筛选后的因子训练 LightGBM 模型。

    补丁已应用:
    - P4 (StockGPT S7): Skip-1-day — 预测 t+2 而非 t+1，消除 microstructure noise
    - P1 (MSIF-OEM E3): 多模式建模 — K-Means 聚类市场状态, 每状态独立模型
    - P2 (StockGPT S4): 分布输出 — 多分类预测收益区间, 输出完整分布+不确定性

核心设计决策：
    1. 目标变量: t+2 收盘 vs t+1 收盘 (Skip-1-day)
       → 避免 t+1 的 bid-ask bounce 和 microstructure noise
       → A股 T+1 制度: T日买入 → T+1日才能卖出 → 预测 T+2 更合理
    2. 多模式: K-Means 聚类 → K 个独立 LightGBM
       → 牛市和熊市的因子效果完全不同, 一个模型学不会
    3. 分布输出: 6分类 (大涨/小涨/微涨/微跌/小跌/大跌)
       → 输出完整概率分布 → 可计算期望收益 + 不确定性(熵)

⚠️ 关键规则：
    - 训练集/验证集/测试集必须按时间顺序切分
    - 禁止随机打乱 (shuffle=False)
    - 禁止使用未来数据
    - 按股票分组，确保同一只股票的数据不在 train 和 test 之间泄露
    - Skip-1-day: target用 close(t+2)/close(t+1)-1, 特征用 t 及之前的全部已知信息

参考：
    - docs/reference-model-analysis.md (补丁来源)
    - docs/architecture.md (第3节)
    - StockGPT (Mai 2024): Skip-1-day, 分布输出
    - MSIF-OEM (Zhao et al. 2025): 多模式建模

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 配置
# ============================================================

# 时间序列切分比例
TRAIN_RATIO = 0.7        # 前 70% 时间 → 训练集
VAL_RATIO = 0.15         # 中间 15% 时间 → 验证集（用于 Early Stopping）
TEST_RATIO = 0.15        # 最后 15% 时间 → 测试集（仅最终评估用）

# 预测 horizon
PREDICT_HORIZON = 2      # P4补丁: Skip-1-day → 预测 t+2 (1=传统t+1)

# 多模式建模
N_REGIMES = 3            # P1补丁: 市场状态聚类数 (牛市/震荡/熊市)

# 分布输出
N_RETURN_BINS = 6        # P2补丁: 收益区间数
RETURN_BIN_EDGES = [-np.inf, -0.05, -0.02, 0, 0.02, 0.05, np.inf]
                          # 大跌/小跌/微跌/微涨/小涨/大涨
RETURN_BIN_LABELS = ["大跌(<-5%)", "小跌(-5%~-2%)", "微跌(-2%~0)",
                     "微涨(0~2%)", "小涨(2%~5%)", "大涨(>5%)"]

# LightGBM 默认超参（二分类）
DEFAULT_PARAMS_BINARY = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.01,
    "n_estimators": 1000,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbose": -1,
}

# LightGBM 默认超参（多分类 → 分布输出）
DEFAULT_PARAMS_MULTICLASS = {
    "objective": "multiclass",
    "num_class": N_RETURN_BINS,
    "metric": "multi_logloss",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.01,
    "n_estimators": 1000,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbose": -1,
}

# 模型保存目录
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


# ============================================================
# 1. 目标变量构造
# ============================================================

def create_target(df: pd.DataFrame, horizon: int = PREDICT_HORIZON) -> pd.DataFrame:
    """
    构造目标变量：预测 t+horizon 日的涨跌方向 (P4补丁: Skip-1-day)

    公式 (horizon=2, Skip-1-day):
        future_return = close(t+2) / close(t+1) - 1
        target = 1 if future_return > 0 else 0

    为什么 Skip-1-day (horizon=2)?
        - t+1 预测包含 microstructure noise (bid-ask bounce)
        - 跳过1天 → 预测的是"后天的收盘 vs 明天的收盘"
        - StockGPT 论文实验证明 skip-1-day 能降低噪声
        - A股 T+1 制度: T日买入→T+1才能卖→T+2开盘可操作, 预测T+2最合理

    为什么用涨跌方向而不预测涨跌幅？
        - 方向比幅度更稳定
        - 日线级别的涨跌幅噪声很大

    参数:
        df: 含 'close', 'code', 'date' 的 DataFrame
        horizon: 预测 horizon (1=传统t+1, 2=Skip-1-day推荐)

    返回:
        df + 'future_return' 列 + 'target' 列
    """
    df = df.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 按股票分组计算未来收益（避免跨股票混算）
    df["close_future"] = df.groupby("code")["close"].shift(-horizon)
    df["close_next"] = df.groupby("code")["close"].shift(-(horizon - 1))

    # Skip-1-day 收益: close(t+2)/close(t+1) - 1
    df["future_return"] = df["close_future"] / df["close_next"] - 1

    # 二分类标签: 涨(1) / 跌(0)
    df["target"] = (df["future_return"] > 0).astype(int)

    # 清理临时列
    df = df.drop(columns=["close_future", "close_next"])

    return df


def create_multiclass_target(df: pd.DataFrame,
                              horizon: int = PREDICT_HORIZON,
                              n_bins: int = N_RETURN_BINS,
                              bin_edges: List[float] = None) -> pd.DataFrame:
    """
    构造多分类目标变量：收益区间 (P2补丁: 分布输出)

    将连续的未来收益映射到 K 个区间:
        0: 大跌 (< -5%)
        1: 小跌 (-5% ~ -2%)
        2: 微跌 (-2% ~ 0)
        3: 微涨 (0 ~ 2%)
        4: 小涨 (2% ~ 5%)
        5: 大涨 (> 5%)

    为什么用多分类而非二分类？
        - 输出 6 维概率向量 → 完整分布信息
        - 可以从分布中提取: 期望收益、方向概率、不确定性(熵)
        - 高熵 = 分布平坦 = 模型不确定 → 降低仓位
        - 低熵 = 分布尖锐 = 模型确定 → 正常仓位

    参数:
        df: 含 'close', 'code', 'date' 的 DataFrame
        horizon: 预测 horizon
        n_bins: 收益区间数
        bin_edges: 区间边界 (长度=n_bins+1)

    返回:
        df + 'future_return' + 'target_multiclass' + 'target_binary' 列
    """
    if bin_edges is None:
        bin_edges = RETURN_BIN_EDGES

    df = df.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 计算未来收益（与 create_target 相同逻辑）
    df["close_future"] = df.groupby("code")["close"].shift(-horizon)
    df["close_next"] = df.groupby("code")["close"].shift(-(horizon - 1))
    df["future_return"] = df["close_future"] / df["close_next"] - 1

    # 映射到区间
    df["target_multiclass"] = pd.cut(
        df["future_return"],
        bins=bin_edges,
        labels=False,         # 用整数标签 0~5
        include_lowest=True
    )

    # 同时保留二分类标签（方便对比）
    df["target_binary"] = (df["future_return"] > 0).astype(int)

    df = df.drop(columns=["close_future", "close_next"])

    return df


def calc_prediction_entropy(prob_vector: np.ndarray) -> float:
    """
    从多分类预测概率向量计算信息熵 (P2补丁: 不确定性量化)

    公式:
        H = -Σ p_i * log2(p_i)

    解读:
        - 最大熵 = log2(K) ≈ 2.58 (6分类, 均匀分布 → 模型完全不确定)
        - 最小熵 = 0 (所有概率集中在1个类别 → 模型完全确定)
        - 一般阈值: H > 2.0 → 高不确定性, 建议轻仓或不交易

    参数:
        prob_vector: 长度为 K 的概率向量 (sum=1)

    返回:
        信息熵值 (0 ~ log2(K))
    """
    # 防止 log(0)
    prob_vector = np.clip(prob_vector, 1e-10, 1.0)
    prob_vector = prob_vector / prob_vector.sum()  # 归一化
    entropy = -np.sum(prob_vector * np.log2(prob_vector))
    return float(entropy)


def calc_expected_return(prob_vector: np.ndarray,
                         bin_midpoints: List[float] = None) -> float:
    """
    从多分类概率向量计算期望收益 (P2补丁)

    E[r] = Σ p_i * midpoint_i

    参数:
        prob_vector: 长度为 K 的概率向量
        bin_midpoints: 各区间的中点收益

    返回:
        期望收益率
    """
    if bin_midpoints is None:
        # 对应 RETURN_BIN_EDGES 的区间中点
        bin_midpoints = [-0.075, -0.035, -0.01, 0.01, 0.035, 0.075]

    prob_vector = prob_vector / prob_vector.sum()
    expected = np.dot(prob_vector, bin_midpoints)
    return float(expected)


# ============================================================
# 2. 市场状态聚类 (P1补丁: 多模式建模)
# ============================================================

class RegimeRouter:
    """
    市场状态路由器 (P1补丁: MSIF-OEM 多模式建模)

    职责:
        1. 用全市场级因子做 K-Means 聚类 → K 种市场状态
        2. 每种状态训练一个独立的 LightGBM 模型
        3. 推理时: 先判断当前 regime → 用对应的模型预测

    为什么需要这个?
        - 牛市和熊市的因子效果完全不同
        - 例如: 动量因子在牛市中有效, 在熊市中可能完全失效
        - 一个模型学习所有状态 → 它必须学会"先判断状态再选规则"
        - 树模型能部分做到, 但不如显式建模高效

    市场状态因子 (用于聚类, 与股票级因子不同):
        - 全市场等权平均收益率 (20日)
        - 全市场波动率 (20日)
        - 全市场上涨股票占比 (20日均值)
        - 全市场成交量变化率 (20日)
    """

    def __init__(self, n_regimes: int = N_REGIMES):
        """
        参数:
            n_regimes: 聚类数 (3=牛市/震荡/熊市, 5=更细粒度)
        """
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.models: Dict[int, lgb.Booster] = {}  # regime_id → model
        self.regime_labels: Dict[int, str] = {}   # regime_id → 人类可读标签
        self.cluster_centers_: np.ndarray = None
        self._fitted = False

    def compute_market_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从全市场数据计算市场级因子

        这些因子描述"整个市场在经历什么"，而非单只股票的特征。

        参数:
            df: 含 date, code, close, volume 的全市场数据

        返回:
            DataFrame, index=date, 列=市场级因子
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 每只股票每日收益率
        df["daily_ret"] = df.groupby("code")["close"].pct_change()

        # 按日期聚合
        daily_market = df.groupby("date").agg(
            equal_weight_ret=("daily_ret", "mean"),          # 全市场等权平均收益
            volatility=("daily_ret", "std"),                  # 截面波动率
            advance_ratio=("daily_ret", lambda x: (x > 0).mean()),  # 上涨占比
            volume_sum=("volume", "sum")                       # 全市场成交量
        ).reset_index()

        # 计算滚动统计
        daily_market["market_ret_20d"] = daily_market["equal_weight_ret"].rolling(20).mean()
        daily_market["market_vol_20d"] = daily_market["equal_weight_ret"].rolling(20).std()
        daily_market["advance_ratio_20d"] = daily_market["advance_ratio"].rolling(20).mean()
        daily_market["volume_change_20d"] = (
            daily_market["volume_sum"] /
            daily_market["volume_sum"].rolling(20).mean() - 1
        )

        daily_market = daily_market.dropna()
        return daily_market

    def fit_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        用全市场数据聚类→识别市场状态

        流程:
            1. 计算 4 个市场级因子
            2. 标准化
            3. K-Means 聚类
            4. 根据聚类中心给每个 regime 打标签

        参数:
            df: 全市场数据 (含 date, code, close, volume)

        返回:
            array, 每个日期对应的 regime_id (0~K-1)
        """
        market_df = self.compute_market_factors(df)

        # 用于聚类的特征
        feature_cols = [
            "market_ret_20d",
            "market_vol_20d",
            "advance_ratio_20d",
            "volume_change_20d"
        ]
        X = market_df[feature_cols].values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 聚类
        regimes = self.kmeans.fit_predict(X_scaled)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # 给每个 regime 打人类可读标签
        self._label_regimes()

        # 构建日期→regime 映射
        regime_map = dict(zip(market_df["date"], regimes))
        self._regime_map = regime_map
        self._fitted = True

        return regimes

    def _label_regimes(self):
        """
        根据聚类中心的特征值给 regime 打标签

        判断逻辑:
            - 收益率高 + 波动率低 + 上涨占比高 → 牛市
            - 收益率低 + 波动率高 + 上涨占比低 → 熊市
            - 其他 → 震荡市
        """
        centers = self.cluster_centers_
        # 标准化后的中心, 用 ret 和 advance_ratio 的加权和来排序
        scores = centers[:, 0] * 0.4 + centers[:, 2] * 0.4 - centers[:, 1] * 0.2
        # scores 越高 → 市场越好
        sorted_indices = np.argsort(scores)  # 从差到好

        n = len(sorted_indices)
        if n == 3:
            self.regime_labels[sorted_indices[0]] = "熊市"
            self.regime_labels[sorted_indices[1]] = "震荡"
            self.regime_labels[sorted_indices[2]] = "牛市"
        elif n == 5:
            self.regime_labels[sorted_indices[0]] = "恐慌"
            self.regime_labels[sorted_indices[1]] = "熊市"
            self.regime_labels[sorted_indices[2]] = "震荡"
            self.regime_labels[sorted_indices[3]] = "复苏"
            self.regime_labels[sorted_indices[4]] = "牛市"
        else:
            for i in range(n):
                self.regime_labels[i] = f"状态{i}"

    def get_regime(self, date) -> int:
        """获取指定日期的市场状态"""
        if not self._fitted:
            raise ValueError("请先调用 fit_regimes()")
        date = pd.Timestamp(date)
        if date not in self._regime_map:
            # 找最近的已知日期
            known_dates = sorted(self._regime_map.keys())
            closest = min(known_dates, key=lambda d: abs((d - date).days))
            return self._regime_map[closest]
        return self._regime_map[date]

    def assign_regimes_to_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        给全量股票数据打上 regime 标签

        参数:
            df: 含 'date' 列的股票数据

        返回:
            df + 'regime' 列
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["regime"] = df["date"].apply(self.get_regime)
        return df


# ============================================================
# 3. 时间序列切分
# ============================================================

def time_series_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序切分 train / val / test

    为什么不能随机切分？
        - 随机切分 → 模型可能用未来数据训练，用过去数据测试
        - 导致模型学到"未来信息"，回测表现虚高
        - 时间序列的因果律：只能用已知信息预测未知

    切分方式：
        |──────── 训练集 (70%) ────────|── 验证集 (15%) ──|── 测试集 (15%) ──|
        2023-01-01 ───────────────→ 按时间顺序 ──────────────────────→ 2026-07-01

    参数:
        df: 含 'date' 列的完整数据（需已转为 datetime）

    返回:
        (train_df, val_df, test_df)，按时间排序
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    dates = sorted(df["date"].unique())
    n_total = len(dates)

    train_cut = int(n_total * TRAIN_RATIO)
    val_cut = int(n_total * (TRAIN_RATIO + VAL_RATIO))

    train_dates = set(dates[:train_cut])
    val_dates = set(dates[train_cut:val_cut])
    test_dates = set(dates[val_cut:])

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    print(f"📅 数据切分: 训练 {len(train_dates)}天 | 验证 {len(val_dates)}天 | 测试 {len(test_dates)}天")

    return train_df, val_df, test_df


# ============================================================
# 4. 数据准备
# ============================================================

def prepare_data(train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 feature_columns: List[str],
                 target_column: str = "target") -> Tuple:
    """
    准备 LightGBM 需要的训练数据格式

    要做的事：
        1. 分离 X (特征) 和 y (目标)
        2. 移除 NaN 行（因子计算初期会有 NaN，如 60 日均线需要 60 天数据）
        3. 可选的样本加权（如给近期数据更高权重）

    参数:
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        feature_columns: 使用的特征列名列表
        target_column: 目标变量列名 (默认 "target")

    返回:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # 只保留特征列和目标列都不为 NaN 的行
    all_cols = feature_columns + [target_column]
    train_clean = train_df[all_cols].dropna()
    val_clean = val_df[all_cols].dropna()
    test_clean = test_df[all_cols].dropna()

    X_train = train_clean[feature_columns].values
    y_train = train_clean[target_column].values
    X_val = val_clean[feature_columns].values
    y_val = val_clean[target_column].values
    X_test = test_clean[feature_columns].values
    y_test = test_clean[target_column].values

    print(f"📊 数据量: 训练 {len(X_train):,} | 验证 {len(X_val):,} | 测试 {len(X_test):,}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 5. 模型训练
# ============================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                params: Dict[str, Any] = None,
                model_type: str = "binary") -> lgb.Booster:
    """
    训练 LightGBM 模型

    训练策略：
        1. 用训练集训练，验证集做 Early Stopping
        2. 如果验证集 loss 在 early_stopping_rounds 内没有提升 → 停止训练
        3. 返回验证集表现最好的那一版模型

    Early Stopping 为什么重要？
        - 树模型持续训练会越来越拟合训练集的噪声
        - 验证集 loss 不降反升 = 在过拟合训练集
        - 应该停在验证集最优的那一步

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        params: LightGBM 超参 dict，不传就用默认值
        model_type: "binary" 二分类 或 "multiclass" 多分类(分布输出)

    返回:
        训练好的 LightGBM Booster 对象
    """
    if params is None:
        params = DEFAULT_PARAMS_MULTICLASS if model_type == "multiclass" else DEFAULT_PARAMS_BINARY

    # 分离 early_stopping_rounds（不是 LightGBM 原生参数）
    early_stopping = params.pop("early_stopping_rounds", 50)
    params_copy = {k: v for k, v in params.items() if k != "early_stopping_rounds"}

    # 构造 LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 训练
    model = lgb.train(
        params=params_copy,
        train_set=train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=params.get("n_estimators", 1000),
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(period=100)],
    )

    # 恢复参数
    params["early_stopping_rounds"] = early_stopping

    return model


# ============================================================
# 6. 多模式训练 (P1补丁)
# ============================================================

def train_regime_models(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        feature_columns: List[str],
                        router: RegimeRouter,
                        model_type: str = "multiclass") -> Dict[int, lgb.Booster]:
    """
    为每种市场状态训练独立的 LightGBM (P1补丁)

    流程:
        1. 按 regime 分组训练集和验证集
        2. 对每种 regime，用该 regime 下的数据训练模型
        3. 如果某 regime 的数据太少，退化为用全量数据

    参数:
        train_df: 训练集（需含 'regime' 列）
        val_df: 验证集（需含 'regime' 列）
        feature_columns: 特征列名
        router: 已拟合的 RegimeRouter
        model_type: "binary" 或 "multiclass"

    返回:
        {regime_id: trained_model}
    """
    target_col = "target_multiclass" if model_type == "multiclass" else "target"
    regimes = sorted(train_df["regime"].unique())

    print(f"\n🔀 多模式训练: 共 {len(regimes)} 种市场状态")
    for regime_id in regimes:
        label = router.regime_labels.get(regime_id, f"状态{regime_id}")
        regime_train = train_df[train_df["regime"] == regime_id]
        regime_val = val_df[val_df["regime"] == regime_id]
        print(f"   {label}: 训练样本 {len(regime_train):,} | 验证样本 {len(regime_val):,}")

    models = {}
    for regime_id in regimes:
        label = router.regime_labels.get(regime_id, f"状态{regime_id}")
        regime_train = train_df[train_df["regime"] == regime_id]
        regime_val = val_df[val_df["regime"] == regime_id]

        if len(regime_train) < 1000:
            print(f"   ⚠️ {label} 样本不足({len(regime_train)}), 使用全量数据训练")
            regime_train = train_df
            regime_val = val_df

        try:
            X_tr, y_tr, X_v, y_v, _, _ = prepare_data(
                regime_train, regime_val, regime_val, feature_columns, target_col
            )
            model = train_model(X_tr, y_tr, X_v, y_v, model_type=model_type)
            models[regime_id] = model
            print(f"   ✅ {label} 模型训练完成")
        except Exception as e:
            print(f"   ❌ {label} 训练失败: {e} → 使用全量模型")
            # 退化为全量训练
            if 0 not in models:
                X_tr, y_tr, X_v, y_v, _, _ = prepare_data(
                    train_df, val_df, val_df, feature_columns, target_col
                )
                models[regime_id] = train_model(X_tr, y_tr, X_v, y_v, model_type=model_type)

    return models


def predict_with_regime(X: np.ndarray,
                        regime_ids: np.ndarray,
                        router: RegimeRouter,
                        return_proba: bool = True) -> np.ndarray:
    """
    用多模式模型预测 (P1补丁)

    流程:
        1. 对每个样本，根据 regime_id 选择对应的模型
        2. 批量预测（同 regime 的样本一起预测，提高效率）

    参数:
        X: 特征矩阵 (n_samples, n_features)
        regime_ids: 每个样本的市场状态
        router: 含已训练模型的 RegimeRouter
        return_proba: True=返回概率, False=返回类别

    返回:
        预测结果 (n_samples, n_classes) 或 (n_samples,)
    """
    n_samples = X.shape[0]
    # 确定输出维度
    first_model = list(router.models.values())[0]
    if return_proba and hasattr(first_model, "predict"):
        test_pred = first_model.predict(X[:1])
        if test_pred.ndim == 2:
            n_outputs = test_pred.shape[1]
            result = np.zeros((n_samples, n_outputs))
        else:
            result = np.zeros(n_samples)
    else:
        result = np.zeros(n_samples)

    # 按 regime 分组预测
    for regime_id in np.unique(regime_ids):
        mask = regime_ids == regime_id
        if regime_id in router.models:
            model = router.models[regime_id]
            if return_proba:
                result[mask] = model.predict(X[mask])
            else:
                preds = model.predict(X[mask])
                if preds.ndim == 2:
                    result[mask] = np.argmax(preds, axis=1)
                else:
                    result[mask] = (preds > 0.5).astype(int)
        else:
            # fallback: 用第一个可用模型
            fallback_regime = list(router.models.keys())[0]
            model = router.models[fallback_regime]
            if return_proba:
                result[mask] = model.predict(X[mask])
            else:
                preds = model.predict(X[mask])
                if preds.ndim == 2:
                    result[mask] = np.argmax(preds, axis=1)
                else:
                    result[mask] = (preds > 0.5).astype(int)

    return result


# ============================================================
# 7. 模型持久化
# ============================================================

def save_model(model: lgb.Booster, filepath: str, metadata: dict = None):
    """
    保存模型到文件

    文件名格式：model_YYYYMMDD_HHMMSS.txt
    （LightGBM 原生格式是 .txt，不是 .pkl）

    metadata 包含：训练日期、特征列表、时间范围、超参

    参数:
        model: LightGBM Booster
        filepath: 保存路径 (.txt)
        metadata: 额外元数据
    """
    model.save_model(filepath)

    # 保存元数据为同名 JSON
    if metadata:
        meta_path = filepath.replace(".txt", "_meta.json")
        # 处理不可序列化的类型
        meta_clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                meta_clean[k] = v
            elif isinstance(v, np.ndarray):
                meta_clean[k] = v.tolist()
            else:
                meta_clean[k] = str(v)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_clean, f, ensure_ascii=False, indent=2, default=str)


def save_regime_models(router: RegimeRouter, output_dir: str):
    """
    保存多模式模型 (P1补丁)

    保存在 output_dir/regime_{id}_{label}.txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for regime_id, model in router.models.items():
        label = router.regime_labels.get(regime_id, f"state{regime_id}")
        filepath = output_dir / f"regime_{regime_id}_{label}.txt"
        save_model(model, str(filepath), {
            "regime_id": regime_id,
            "regime_label": label,
            "saved_at": datetime.now().isoformat()
        })

    # 保存 RegimeRouter 的聚类参数
    router_path = output_dir / "regime_router.pkl"
    with open(router_path, "wb") as f:
        pickle.dump({
            "kmeans": router.kmeans,
            "scaler": router.scaler,
            "regime_labels": router.regime_labels,
            "regime_map": router._regime_map,
            "n_regimes": router.n_regimes,
        }, f)

    print(f"💾 多模式模型已保存到 {output_dir}")


def load_model(filepath: str) -> lgb.Booster:
    """从文件加载模型"""
    model = lgb.Booster(model_file=filepath)
    return model


def load_regime_router(filepath: str) -> RegimeRouter:
    """加载 RegimeRouter (P1补丁)"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    router = RegimeRouter(n_regimes=data["n_regimes"])
    router.kmeans = data["kmeans"]
    router.scaler = data["scaler"]
    router.regime_labels = data["regime_labels"]
    router._regime_map = data["regime_map"]
    router.cluster_centers_ = data["kmeans"].cluster_centers_
    router._fitted = True
    return router


# ============================================================
# 8. 主流程
# ============================================================

def train_pipeline(feature_path: str,
                   feature_columns: List[str] = None,
                   model_output_dir: str = None,
                   model_type: str = "multiclass",
                   use_regime_models: bool = True) -> Tuple[Any, dict]:
    """
    【主函数】完整的训练管线

    流程：
        1. 加载特征数据
        2. create_multiclass_target() → 构造分布目标变量
        3. RegimeRouter.fit_regimes() → 识别市场状态 (P1补丁)
        4. time_series_split() → 按时间切分
        5. train_regime_models() → 多模式训练 (P1补丁)
        6. 保存所有模型

    参数:
        feature_path: 特征 Parquet 文件路径
        feature_columns: 使用的特征列（None=使用全部数值列）
        model_output_dir: 模型保存目录
        model_type: "binary" 或 "multiclass"
        use_regime_models: 是否使用多模式建模 (P1补丁)

    返回:
        (模型 or RegimeRouter, 训练元数据)
    """
    if model_output_dir is None:
        model_output_dir = str(MODEL_DIR)

    print(f"\n{'='*60}")
    print(f"🚀 开始训练管线 (horizon={PREDICT_HORIZON}, type={model_type})")
    print(f"{'='*60}")

    # ---- 1. 加载数据 ----
    print(f"\n📂 加载特征数据: {feature_path}")
    df = pd.read_parquet(feature_path)
    print(f"   数据量: {len(df):,} 行 × {len(df.columns)} 列")

    # 自动选择特征列
    if feature_columns is None:
        exclude_cols = {"date", "code", "name", "close", "open", "high", "low",
                        "volume", "amount", "turnover_rate", "is_suspended", "is_st",
                        "future_return", "target", "target_multiclass", "target_binary",
                        "regime"}
        feature_columns = [c for c in df.columns if c not in exclude_cols
                           and df[c].dtype in ("float64", "float32", "int64", "int32")]
    print(f"   使用特征: {len(feature_columns)} 个")

    # ---- 2. 构造目标变量 ----
    print(f"\n🎯 构造目标变量 (horizon={PREDICT_HORIZON})")
    if model_type == "multiclass":
        df = create_multiclass_target(df, horizon=PREDICT_HORIZON)
        target_col = "target_multiclass"
        print(f"   分布输出: {N_RETURN_BINS} 个区间 {RETURN_BIN_LABELS}")
    else:
        df = create_target(df, horizon=PREDICT_HORIZON)
        target_col = "target"

    # 统计数据分布
    if target_col == "target_multiclass":
        bin_counts = df[target_col].value_counts().sort_index()
        for i, count in bin_counts.items():
            if i < len(RETURN_BIN_LABELS):
                pct = count / len(df) * 100
                print(f"   {RETURN_BIN_LABELS[i]}: {count:,} ({pct:.1f}%)")

    # ---- 3. 市场状态聚类 (P1补丁) ----
    if use_regime_models:
        print(f"\n🔍 市场状态聚类 (K={N_REGIMES})")
        router = RegimeRouter(n_regimes=N_REGIMES)
        router.fit_regimes(df)
        df = router.assign_regimes_to_data(df)

        for rid, label in router.regime_labels.items():
            count = (df["regime"] == rid).sum()
            print(f"   {label}: {count:,} 样本 ({count/len(df)*100:.1f}%)")

    # ---- 4. 时间序列切分 ----
    train_df, val_df, test_df = time_series_split(df)

    # ---- 5. 训练 ----
    if use_regime_models:
        print(f"\n🤖 多模式训练 ({N_REGIMES} regimes × {model_type})")
        models = train_regime_models(train_df, val_df, feature_columns, router, model_type)
        router.models = models

        # 保存
        save_regime_models(router, model_output_dir)

        metadata = {
            "horizon": PREDICT_HORIZON,
            "model_type": model_type,
            "n_regimes": N_REGIMES,
            "regime_labels": router.regime_labels,
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "train_dates": (str(train_df["date"].min()), str(train_df["date"].max())),
            "test_dates": (str(test_df["date"].min()), str(test_df["date"].max())),
            "trained_at": datetime.now().isoformat(),
        }

        print(f"\n✅ 多模式训练完成!")
        return router, metadata

    else:
        print(f"\n🤖 单模型训练 ({model_type})")
        X_tr, y_tr, X_v, y_v, X_te, y_te = prepare_data(
            train_df, val_df, test_df, feature_columns, target_col
        )
        model = train_model(X_tr, y_tr, X_v, y_v, model_type=model_type)

        # 保存
        model_path = Path(model_output_dir) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "horizon": PREDICT_HORIZON,
            "model_type": model_type,
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "train_dates": (str(train_df["date"].min()), str(train_df["date"].max())),
            "test_dates": (str(test_df["date"].min()), str(test_df["date"].max())),
            "trained_at": datetime.now().isoformat(),
        }
        save_model(model, str(model_path), metadata)

        print(f"\n✅ 训练完成! 模型保存到 {model_path}")
        return model, metadata


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import sys

    feature_path = Path(__file__).resolve().parent.parent.parent / "data" / "features.parquet"

    if not feature_path.exists():
        print(f"⚠️ 特征数据不存在: {feature_path}")
        print("   请先运行 python src/features/alpha_factors.py")
        print("\n💡 可以用合成数据测试训练管线:")
        print("   python -c \"from src.models.train import *; test_with_synthetic()\"")
        sys.exit(1)

    # 运行完整训练管线
    model, meta = train_pipeline(
        str(feature_path),
        model_type="multiclass",
        use_regime_models=True
    )
