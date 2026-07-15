"""
Alpha 因子计算模块 — 从 OHLCV 衍生 18+ 量化因子

设计原则:
    1. 每个因子独立函数 → 方便单独测试/按需组合
    2. 向量化 → pandas rolling/ewm, 不写 for 循环
    3. groupby("code") → 每只股票独立计算, 禁止跨股票混算
    4. 禁止 look-ahead → 只用 ≤t 时刻的信息

因子: 收益(4) + 均线偏离(4) + MACD(3) + 波动率(3) + 成交量(3) + RSI(1) = 18

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 工具函数
# ============================================================

def _by_code(df: pd.DataFrame, col: str, fn, *args, **kwargs):
    """按股票代码分组后对指定列应用函数 (避免跨股票混算)"""
    return df.groupby("code")[col].transform(lambda x: fn(x, *args, **kwargs))


# ============================================================
# 1. 收益类因子
# ============================================================

def calc_ret(df: pd.DataFrame, window: int) -> pd.Series:
    """N日收益率: ret = close / close.shift(window) - 1"""
    close_shifted = df.groupby("code")["close"].shift(window)
    return df["close"] / close_shifted - 1


def calc_all_return_factors(df: pd.DataFrame) -> pd.DataFrame:
    """ret_1d, ret_5d, ret_10d, ret_20d"""
    df = df.copy()
    for w in [1, 5, 10, 20]:
        df[f"ret_{w}d"] = calc_ret(df, w)
    return df


# ============================================================
# 2. 均线偏离类因子
# ============================================================

def calc_ma_dev(df: pd.DataFrame, window: int) -> pd.Series:
    """价格相对N日均线的偏离: close / MA(N) - 1"""
    ma = df.groupby("code")["close"].transform(lambda x: x.rolling(window, min_periods=window//2).mean())
    return df["close"] / ma - 1


def calc_all_ma_factors(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """ma_dev_5, ma_dev_10, ma_dev_20, ma_dev_60"""
    if windows is None:
        windows = [5, 10, 20, 60]
    df = df.copy()
    for w in windows:
        df[f"ma_dev_{w}"] = calc_ma_dev(df, w)
    return df


# ============================================================
# 3. MACD 因子
# ============================================================

def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD = EMA(fast) - EMA(slow); signal = EMA(MACD); hist = MACD - signal"""
    df = df.copy()

    def _macd_for_stock(close: pd.Series) -> pd.DataFrame:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_line - macd_signal
        })

    result = df.groupby("code")["close"].apply(_macd_for_stock).reset_index(level=1, drop=True)
    df["macd"] = result["macd"]
    df["macd_signal"] = result["macd_signal"]
    df["macd_hist"] = result["macd_hist"]
    return df


# ============================================================
# 4. 波动率类因子
# ============================================================

def calc_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """年化历史波动率: std(daily_ret, window) × √252"""
    daily_ret = df.groupby("code")["close"].pct_change()
    vol = daily_ret.groupby(df["code"]).transform(
        lambda x: x.rolling(window, min_periods=window//2).std()
    )
    return vol * np.sqrt(252)


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    平均真实波幅: TR = max(H-L, |H-C_prev|, |L-C_prev|)
    用 Wilder 平滑 (ema, alpha=1/window)
    """
    prev_close = df.groupby("code")["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - prev_close),
            np.abs(df["low"] - prev_close)
        )
    )
    atr = tr.groupby(df["code"]).transform(
        lambda x: x.ewm(alpha=1/window, adjust=False).mean()
    )
    return atr


def calc_bollinger_width(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """布林带宽度: (upper - lower) / mid"""
    ma = df.groupby("code")["close"].transform(
        lambda x: x.rolling(window, min_periods=window//2).mean()
    )
    std = df.groupby("code")["close"].transform(
        lambda x: x.rolling(window, min_periods=window//2).std()
    )
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / ma


# ============================================================
# 5. 成交量类因子
# ============================================================

def calc_volume_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    """量比: volume / avg_volume(N)"""
    avg_vol = df.groupby("code")["volume"].transform(
        lambda x: x.rolling(window, min_periods=window//2).mean()
    )
    ratio = df["volume"] / avg_vol.replace(0, np.nan)
    return ratio


def calc_turnover_change(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """换手率变化: turnover / turnover.shift(window) - 1"""
    if "turnover_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    shifted = df.groupby("code")["turnover_rate"].shift(window)
    return df["turnover_rate"] / shifted.replace(0, np.nan) - 1


# ============================================================
# 6. RSI (Wilder 平滑)
# ============================================================

def calc_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI with Wilder smoothing (EMA, alpha=1/window)"""
    delta = df.groupby("code")["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(df["code"]).transform(
        lambda x: x.ewm(alpha=1/window, adjust=False).mean()
    )
    avg_loss = loss.groupby(df["code"]).transform(
        lambda x: x.ewm(alpha=1/window, adjust=False).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)


# ============================================================
# 7. 额外增强因子 (P1补丁: MSIF-OEM 市场级因子)
# ============================================================

def calc_market_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    市场级因子 (P1补丁: 用于 RegimeRouter 聚类, 非个股因子):
      - market_ret_20d: 全市场等权20日平均收益
      - market_vol_20d: 全市场20日波动率
      - advance_ratio_20d: 20日上涨股票占比均值
      - volume_change_20d: 全市场成交量20日变化
    """
    df = df.copy()
    df["daily_ret"] = df.groupby("code")["close"].pct_change()

    daily = df.groupby("date").agg(
        eq_ret=("daily_ret", "mean"),
        cross_vol=("daily_ret", "std"),
        advance=("daily_ret", lambda x: (x > 0).mean()),
        total_vol=("volume", "sum")
    ).reset_index()

    daily["market_ret_20d"] = daily["eq_ret"].rolling(20).mean()
    daily["market_vol_20d"] = daily["eq_ret"].rolling(20).std()
    daily["advance_ratio_20d"] = daily["advance"].rolling(20).mean()
    daily["volume_change_20d"] = daily["total_vol"] / daily["total_vol"].rolling(20).mean() - 1

    daily = daily.dropna()
    df["date"] = pd.to_datetime(df["date"])
    daily["date"] = pd.to_datetime(daily["date"])

    merge_cols = ["date", "market_ret_20d", "market_vol_20d", "advance_ratio_20d", "volume_change_20d"]
    df = df.merge(daily[merge_cols], on="date", how="left")
    df.drop(columns=["daily_ret"], inplace=True, errors="ignore")
    return df


# ============================================================
# 8. 一键计算全部因子
# ============================================================

def compute_all_factors(df: pd.DataFrame,
                        include_market: bool = True) -> pd.DataFrame:
    """
    计算全部 18+ Alpha 因子 + 可选市场级因子。

    总计: 收益(4) + 均线偏离(4) + MACD(3) + 波动率(3) + 成交量(3) + RSI(1) = 18
         + 市场级(4) = 22 (如果 include_market=True)
    """
    print(f"\n{'='*60}")
    print("🔬 因子计算管线")
    print(f"{'='*60}")

    df = df.copy()

    # 确保按code+date排序
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 1. 收益类 (4)
    print("  收益类因子...", end=" ")
    df = calc_all_return_factors(df)
    print("✅ ret_1d/5d/10d/20d")

    # 2. 均线偏离 (4)
    print("  均线偏离因子...", end=" ")
    df = calc_all_ma_factors(df)
    print("✅ ma_dev_5/10/20/60")

    # 3. MACD (3)
    print("  MACD因子...", end=" ")
    df = calc_macd(df)
    print("✅ macd/macd_signal/macd_hist")

    # 4. 波动率 (3)
    print("  波动率因子...", end=" ")
    df["volatility_20"] = calc_volatility(df, 20)
    df["atr_14"] = calc_atr(df, 14)
    df["bb_width_20"] = calc_bollinger_width(df, 20)
    print("✅ volatility_20/atr_14/bb_width_20")

    # 5. 成交量 (3)
    print("  成交量因子...", end=" ")
    df["volume_ratio_5"] = calc_volume_ratio(df, 5)
    df["volume_ratio_20"] = calc_volume_ratio(df, 20)
    df["turnover_change_5"] = calc_turnover_change(df, 5)
    print("✅ volume_ratio_5/20, turnover_change_5")

    # 6. RSI (1)
    print("  RSI因子...", end=" ")
    df["rsi_14"] = calc_rsi(df, 14)
    print("✅ rsi_14")

    # 7. 市场级因子 (4, P1补丁)
    if include_market:
        print("  市场级因子...", end=" ")
        df = calc_market_factors(df)
        print("✅ market_ret/vol/advance/volume_change")

    # 清理: 删除因子计算过程中产生的中间列
    df = df.drop(columns=["daily_ret"], errors="ignore")

    # 统计
    factor_cols = [c for c in df.columns if c not in
                   ("date", "code", "name", "open", "high", "low", "close",
                    "volume", "amount", "is_suspended", "is_st", "turnover_rate")]
    n_factors = len(factor_cols)

    print(f"\n✅ 因子计算完成: {n_factors} 个因子, {len(df):,} 行")
    return df


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    clean_path = Path(__file__).resolve().parent.parent.parent / "data" / "a_stock_daily_clean.parquet"
    feature_path = Path(__file__).resolve().parent.parent.parent / "data" / "features.parquet"

    if clean_path.exists():
        df = pd.read_parquet(clean_path)
        df = compute_all_factors(df)
        df.to_parquet(feature_path, index=False)
        print(f"💾 特征保存: {feature_path}")
    else:
        print(f"⚠️ 清洗数据不存在: {clean_path}")
        print("   请先运行 python src/data/cleaner.py")
