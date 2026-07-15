"""
信号生成模块 — 把模型预测转化为可执行的交易信号

作用：
    接收模型预测（支持二分类概率 或 多分类分布输出），经过多维度过滤和排序，
    输出具体的买卖指令。

    补丁已应用:
    - P2 (StockGPT S4): 分布输出 — 从多分类预测中提取方向概率+不确定性(熵)
    - P3 (Increase Alpha I2): 三元信号 — +1(买)/-1(卖)/0(观望)

信号生成逻辑：
    1. 多分类输出 → 提取方向概率 + 计算不确定性(熵) + 期望收益
    2. 不确定性过滤：高熵 → 模型不确定 → 降低仓位或过滤
    3. 方向概率筛选：P(涨) >= 阈值
    4. 排序择优：按期望收益从高到低排序
    5. 生成信号：对持仓中但不再推荐的股票发出卖出信号

交易信号的三种状态：
    - BUY  (+1): 预测会涨，且满足所有筛选条件 → 买入
    - SELL (-1): 持仓中但不再推荐，或触发止损 → 卖出
    - HOLD ( 0): 不做任何操作

依赖：
    - pandas, numpy
    - src/models/train.py (calc_prediction_entropy, calc_expected_return)

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================
# 配置
# ============================================================

MIN_PROB_THRESHOLD = 0.55       # 方向概率最低阈值
MIN_EXPECTED_RETURN = 0.002     # 最小期望收益 (0.2%/天)
MAX_SIGNALS_PER_DAY = 10        # 每日最多推送几只股票
MIN_HOLDING_DAYS = 3            # 最短持有天数（避免频繁交易）

# P2补丁: 不确定性过滤
MAX_ENTROPY = 2.0               # 熵阈值 (6分类最大熵≈2.58, 2.0=中等确定)
MAX_ENTROPY_LOW_CONFIDENCE = 2.5  # 极高不确定性 → 直接过滤


# ============================================================
# 1. 多分类输出处理 (P2补丁)
# ============================================================

def extract_direction_prob(prob_matrix: np.ndarray,
                           bin_labels: List[str] = None) -> np.ndarray:
    """
    从多分类概率向量中提取方向概率 P(涨)

    多分类输出: [P(大跌), P(小跌), P(微跌), P(微涨), P(小涨), P(大涨)]
    方向概率 = P(微涨) + P(小涨) + P(大涨) = 后3个bin的概率和

    参数:
        prob_matrix: (n_samples, n_bins) 概率矩阵
        bin_labels: 区间标签（未使用，保留扩展）

    返回:
        (n_samples,) 方向概率数组
    """
    n_bins = prob_matrix.shape[1]
    # 后一半的 bin 是"涨"方向
    up_bins_start = n_bins // 2
    direction_prob = prob_matrix[:, up_bins_start:].sum(axis=1)
    return direction_prob


def extract_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    """
    从多分类概率矩阵计算每个样本的信息熵 (P2补丁)

    公式:
        H = -Σ p_i * log2(p_i)

    解读:
        - 低熵 (~0.5): 概率集中在1个bin → 模型很确定
        - 中熵 (~1.5): 概率分散在2-3个bin → 模型有些犹豫
        - 高熵 (~2.5): 接近均匀分布 → 模型几乎在瞎猜

    参数:
        prob_matrix: (n_samples, n_bins) 概率矩阵

    返回:
        (n_samples,) 熵值数组
    """
    # 防止 log(0)
    prob = np.clip(prob_matrix, 1e-10, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    entropy = -np.sum(prob * np.log2(prob), axis=1)
    return entropy


def extract_expected_return(prob_matrix: np.ndarray,
                            bin_midpoints: List[float] = None) -> np.ndarray:
    """
    从多分类概率矩阵计算期望收益 (P2补丁)

    E[r] = Σ p_i * midpoint_i

    参数:
        prob_matrix: (n_samples, n_bins) 概率矩阵
        bin_midpoints: 各区间的中点收益, 默认对应6分类

    返回:
        (n_samples,) 期望收益数组
    """
    if bin_midpoints is None:
        # 对应 6 个区间的中点: 大跌/小跌/微跌/微涨/小涨/大涨
        bin_midpoints = [-0.075, -0.035, -0.01, 0.01, 0.035, 0.075]

    bin_midpoints = np.array(bin_midpoints)
    prob = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    expected = prob @ bin_midpoints
    return expected


# ============================================================
# 2. 置信度分级 (P2补丁: 不确定性 → 动态仓位)
# ============================================================

def assign_confidence_level(entropy: np.ndarray) -> np.ndarray:
    """
    根据预测熵分配置信度等级 (P2补丁)

    等级映射:
        - "high":   熵 < 1.5  → 概率分布尖锐 → 模型很确定 → 满仓
        - "medium": 熵 1.5~2.0 → 中等确定 → 半仓
        - "low":    熵 2.0~2.5 → 不太确定 → 轻仓(1/4)
        - "filter": 熵 > 2.5  → 接近随机 → 直接过滤不交易

    参数:
        entropy: (n_samples,) 熵值数组

    返回:
        (n_samples,) 置信度等级字符串数组
    """
    confidence = np.full(len(entropy), "medium", dtype=object)
    confidence[entropy < 1.5] = "high"
    confidence[(entropy >= 2.0) & (entropy < 2.5)] = "low"
    confidence[entropy >= 2.5] = "filter"
    return confidence


def confidence_to_position_weight(confidence: str) -> float:
    """
    置信度 → 仓位权重

    high   → 1.0 (满仓: 单票20%)
    medium → 0.5 (半仓: 单票10%)
    low    → 0.25 (轻仓: 单票5%)
    filter → 0.0 (不交易)
    """
    weights = {"high": 1.0, "medium": 0.5, "low": 0.25, "filter": 0.0}
    return weights.get(confidence, 0.5)


# ============================================================
# 3. 预测筛选
# ============================================================

def filter_by_prob(predictions: pd.DataFrame,
                   prob_column: str = "direction_prob",
                   min_prob: float = MIN_PROB_THRESHOLD) -> pd.DataFrame:
    """
    按方向概率筛选股票

    逻辑：
        - 只保留 direction_prob >= min_prob 的股票
        - 概率太低（如 50.1%）跟扔硬币没区别，不交易

    参数:
        predictions: 模型预测结果（含 direction_prob 列）
        prob_column: 概率列名
        min_prob: 最低阈值

    返回:
        筛选后的 DataFrame
    """
    return predictions[predictions[prob_column] >= min_prob].copy()


def filter_by_entropy(predictions: pd.DataFrame,
                      entropy_column: str = "entropy",
                      max_entropy: float = MAX_ENTROPY_LOW_CONFIDENCE) -> pd.DataFrame:
    """
    按不确定性过滤 (P2补丁)

    逻辑：
        - 熵 > MAX_ENTROPY_LOW_CONFIDENCE (2.5) → 过滤掉
        - 模型在瞎猜的时候不应该交易

    参数:
        predictions: 含 entropy 列的预测结果
        entropy_column: 熵列名
        max_entropy: 最大允许熵

    返回:
        过滤后的 DataFrame
    """
    if entropy_column not in predictions.columns:
        return predictions
    return predictions[predictions[entropy_column] <= max_entropy].copy()


def filter_by_expected_return(predictions: pd.DataFrame,
                              min_return: float = MIN_EXPECTED_RETURN) -> pd.DataFrame:
    """
    按期望收益过滤 (P2补丁)

    逻辑：
        - 即使方向概率 > 55%，但如果期望收益太低 (如0.05%)
        - 扣除交易成本后可能还是亏钱 → 过滤掉

    参数:
        predictions: 含 expected_return 列的预测结果
        min_return: 最小期望收益

    返回:
        过滤后的 DataFrame
    """
    if "expected_return" not in predictions.columns:
        return predictions
    return predictions[predictions["expected_return"] >= min_return].copy()


def rank_by_confidence(predictions: pd.DataFrame,
                       rank_column: str = "expected_return",
                       max_signals: int = MAX_SIGNALS_PER_DAY) -> pd.DataFrame:
    """
    按优先级排序，取 Top N

    排序逻辑：
        - 优先按期望收益从高到低排序（如果有多分类输出）
        - 否则按方向概率排序

    参数:
        predictions: 已筛选的预测结果
        rank_column: 排序依据列
        max_signals: 最多保留几只

    返回:
        排序后的 Top N
    """
    if rank_column not in predictions.columns:
        rank_column = "direction_prob"
    predictions = predictions.sort_values(rank_column, ascending=False)
    return predictions.head(max_signals).copy()


# ============================================================
# 4. 信号生成
# ============================================================

def generate_buy_signals(predictions: pd.DataFrame,
                         current_positions: List[str]) -> pd.DataFrame:
    """
    生成买入信号

    规则：
        1. 方向概率 >= 阈值
        2. 熵 <= 上限（模型不能太不确定）
        3. 不在当前持仓中（避免重复买入）
        4. 排名在 Top N
        5. 置信度 ≠ "filter"

    参数:
        predictions: 当日所有股票的预测结果
        current_positions: 当前持仓的股票代码列表

    返回:
        DataFrame，含 code, name, signal, confidence, direction_prob, ...
    """
    if predictions is None or len(predictions) == 0:
        return pd.DataFrame()

    df = predictions.copy()

    # Step 1: 不确定性过滤 (P2补丁)
    if "entropy" in df.columns:
        df = filter_by_entropy(df)
    if "confidence" not in df.columns and "entropy" in df.columns:
        df["confidence"] = assign_confidence_level(df["entropy"].values)

    # Step 2: 置信度过滤
    if "confidence" in df.columns:
        df = df[df["confidence"] != "filter"]

    # Step 3: 方向概率过滤
    if "direction_prob" in df.columns:
        df = filter_by_prob(df, "direction_prob")
    elif "pred_prob" in df.columns:
        df = filter_by_prob(df, "pred_prob")

    # Step 4: 期望收益过滤 (P2补丁)
    if "expected_return" in df.columns:
        df = filter_by_expected_return(df)

    # Step 5: 排除已持仓
    current_set = set(current_positions) if current_positions else set()
    if "code" in df.columns:
        df = df[~df["code"].isin(current_set)]

    if len(df) == 0:
        return pd.DataFrame()

    # Step 6: 排序取 Top N
    df = rank_by_confidence(df)

    # Step 7: 生成信号
    df["signal"] = 1  # BUY
    if "reason" not in df.columns:
        df["reason"] = "model_prediction"

    # 动态仓位权重 (P2补丁)
    if "confidence" in df.columns:
        df["position_weight"] = df["confidence"].apply(confidence_to_position_weight)
    else:
        df["position_weight"] = 1.0

    return df


def generate_sell_signals(current_positions: List[str],
                          recommended_stocks: List[str],
                          stop_loss_stocks: List[str] = None) -> pd.DataFrame:
    """
    生成卖出信号

    卖出条件（满足任一即卖出）：
        1. 持仓股票不在今日推荐列表中（模型不再看好了）
        2. 触发止损（由风控模块传入）
        3. （可选）达到止盈目标

    参数:
        current_positions: 当前持仓股票代码列表
        recommended_stocks: 今日推荐买入的股票代码列表
        stop_loss_stocks: 触发止损的股票代码列表

    返回:
        DataFrame，含 code, signal=-1, sell_reason
    """
    if stop_loss_stocks is None:
        stop_loss_stocks = []

    sell_records = []
    recommended_set = set(recommended_stocks) if recommended_stocks else set()
    stop_set = set(stop_loss_stocks)

    for code in current_positions:
        reasons = []
        if code in stop_set:
            reasons.append("stop_loss")
        if code not in recommended_set:
            reasons.append("no_longer_recommended")

        if reasons:
            sell_records.append({
                "code": code,
                "signal": -1,
                "sell_reason": "|".join(reasons),
            })

    if not sell_records:
        return pd.DataFrame()

    return pd.DataFrame(sell_records)


# ============================================================
# 5. 信号合并与后处理
# ============================================================

def merge_signals(buy_signals: pd.DataFrame,
                  sell_signals: pd.DataFrame) -> pd.DataFrame:
    """
    合并买入和卖出信号为统一的信号表

    参数:
        buy_signals: 买入信号
        sell_signals: 卖出信号

    返回:
        统合信号表，列: code, signal(+1/-1), reason, ...
    """
    frames = []
    if buy_signals is not None and len(buy_signals) > 0:
        frames.append(buy_signals)
    if sell_signals is not None and len(sell_signals) > 0:
        # 统一列名
        sell = sell_signals.copy()
        for col in ["direction_prob", "expected_return", "entropy", "confidence",
                     "position_weight", "name"]:
            if col not in sell.columns:
                sell[col] = None
        sell["reason"] = sell.get("sell_reason", "sell")
        frames.append(sell)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def generate_signals_summary(signals: pd.DataFrame) -> str:
    """
    生成当日信号摘要（用于 Telegram 推送和日志）

    参数:
        signals: merge_signals() 的输出

    返回:
        Markdown 格式的信号摘要
    """
    if signals is None or len(signals) == 0:
        return "📊 今日无交易信号"

    buys = signals[signals["signal"] == 1]
    sells = signals[signals["signal"] == -1]

    lines = [f"📊 今日信号: 买入 {len(buys)} 只 | 卖出 {len(sells)} 只\n"]

    if len(buys) > 0:
        lines.append("🟢 **买入信号:**")
        for _, row in buys.iterrows():
            name = row.get("name", row.get("code", "?"))
            prob = row.get("direction_prob", row.get("pred_prob", 0))
            conf = row.get("confidence", "?")
            exp_ret = row.get("expected_return", None)
            line = f"   {name} | 方向概率 {prob:.1%}"
            if exp_ret is not None:
                line += f" | 期望收益 {exp_ret:.2%}"
            line += f" | 置信度 {conf}"
            lines.append(line)

    if len(sells) > 0:
        lines.append("\n🔴 **卖出信号:**")
        for _, row in sells.iterrows():
            code = row.get("code", "?")
            reason = row.get("sell_reason", row.get("reason", "?"))
            lines.append(f"   {code} | 原因: {reason}")

    return "\n".join(lines)


# ============================================================
# 6. 主流程
# ============================================================

def generate_signals(predictions: pd.DataFrame,
                     current_positions: List[str] = None,
                     stop_loss_stocks: List[str] = None) -> pd.DataFrame:
    """
    【主函数】生成当日全部交易信号

    完整流程：
        1. 从多分类输出提取方向概率 + 熵 + 期望收益 (P2补丁)
        2. 不确定性过滤 (P2补丁)
        3. 方向概率筛选
        4. 期望收益过滤 (P2补丁)
        5. 置信度分级 + 动态仓位权重
        6. 排序取 Top N
        7. 生成买入信号
        8. 生成卖出信号
        9. 合并输出

    参数:
        predictions: 当日模型预测结果
            - 二分类模型: 含 pred_prob 列
            - 多分类模型: 含 prob_0 ~ prob_5 列 (P2补丁)
        current_positions: 当前持仓的股票代码列表
        stop_loss_stocks: 触发止损的股票代码列表

    返回:
        当日信号表，含 code, signal(+1/-1), direction_prob, entropy,
        expected_return, confidence, position_weight, reason
    """
    if current_positions is None:
        current_positions = []
    if stop_loss_stocks is None:
        stop_loss_stocks = []

    df = predictions.copy()

    # ---- 处理多分类输出 (P2补丁) ----
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if len(prob_cols) >= 3:  # 多分类输出
        prob_matrix = df[prob_cols].values.astype(float)

        # 提取方向概率
        df["direction_prob"] = extract_direction_prob(prob_matrix)

        # 计算熵 (不确定性)
        df["entropy"] = extract_entropy(prob_matrix)

        # 计算期望收益
        df["expected_return"] = extract_expected_return(prob_matrix)

        # 分配置信度
        df["confidence"] = assign_confidence_level(df["entropy"].values)

        print(f"📊 P2分布输出: {len(df)} 样本 | "
              f"方向概率均值 {df['direction_prob'].mean():.2%} | "
              f"熵均值 {df['entropy'].mean():.2f}")

    elif "pred_prob" in df.columns:
        # 二分类输出 → 方向概率 = pred_prob
        df["direction_prob"] = df["pred_prob"]
        print(f"📊 二分类输出: {len(df)} 样本 | 概率均值 {df['pred_prob'].mean():.2%}")

    # ---- 生成买卖信号 ----
    buy_signals = generate_buy_signals(df, current_positions)
    recommended = buy_signals["code"].tolist() if len(buy_signals) > 0 else []
    sell_signals = generate_sell_signals(current_positions, recommended, stop_loss_stocks)

    # ---- 合并 ----
    signals = merge_signals(buy_signals, sell_signals)

    # ---- 打印摘要 ----
    summary = generate_signals_summary(signals)
    print(summary)

    return signals


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("信号生成模块")
    print(f"筛选规则：方向概率 >= {MIN_PROB_THRESHOLD}, 每日最多 {MAX_SIGNALS_PER_DAY} 只")
    print("信号类型：+1 买入 / -1 卖出 / 0 持有")
    print("\nP2补丁: 分布输出 + 不确定性过滤")
    print(f"  熵阈值: {MAX_ENTROPY} (中等) / {MAX_ENTROPY_LOW_CONFIDENCE} (过滤)")
    print("  置信度: high→满仓 | medium→半仓 | low→轻仓 | filter→不交易")
