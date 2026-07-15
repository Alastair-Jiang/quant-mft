"""
数据清洗模块 — A股日线数据预处理

作用：把 fetcher.py 拉下来的原始 CSV 处理成可直接用于特征工程的干净数据。

清洗流程：
    1. 数据加载 + 基本过滤 (新股剔除、列检查)
    2. ST股票标记与过滤 (ST涨跌停5%≠普通10%)
    3. 停牌检测 (禁止前向填充！)
    4. 复权一致性检查 (high≥close≥low)
    5. 微盘股/流动性过滤 (P5补丁: StockGPT S5)
    6. 极值处理 (MAD Winsorization, 金融fat-tail)
    7. 缺失值处理 (区分停牌 vs 真缺失)

作者: 蒋东旭
日期: 2026-07-15
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 配置
# ============================================================
WINSORIZE_METHOD = "mad"
WINSORIZE_THRESHOLD = 5.0
MAX_SUSPEND_DAYS = 60
MIN_TRADING_DAYS = 200
EXCLUDE_ST = True
MIN_MARKET_CAP_PCT = 10
MIN_VOLUME_PCT = 5


# ============================================================
# 1. 数据加载
# ============================================================

def load_raw_data(filepath: str) -> pd.DataFrame:
    """加载 fetcher.py 生成的原始 CSV, 做基本类型转换"""
    df = pd.read_csv(filepath, encoding="utf-8-sig", dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"]).reset_index(drop=True)
    print(f"📂 加载数据: {len(df):,} 行, {df['code'].nunique()} 只股票")
    return df


# ============================================================
# 2. 极值处理 — MAD Winsorization
# ============================================================

def winsorize_mad(series: pd.Series, threshold: float = 5.0) -> pd.Series:
    """
    MAD (Median Absolute Deviation) 极值处理。
    比 3σ 更适合金融数据——收益率是 fat-tail 分布, 不是正态。
    """
    s = series.dropna()
    if len(s) < 10:
        return series
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad < 1e-10:
        return series
    upper = med + threshold * mad
    lower = med - threshold * mad
    return series.clip(lower=lower, upper=upper)


def apply_winsorize(df: pd.DataFrame, columns: List[str], method: str = "mad") -> pd.DataFrame:
    """
    对指定列做 Winsorization。
    只处理收益率/成交量变化率等比率列, 不处理价格本身。
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df.groupby("code")[col].transform(
                lambda x: winsorize_mad(x, WINSORIZE_THRESHOLD)
            )
    return df


# ============================================================
# 3. ST 股票处理
# ============================================================

def mark_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """标记 ST/*ST 股票 (涨跌停规则5%≠普通10%, 必须单独处理)"""
    df = df.copy()
    df["is_st"] = df["name"].str.contains(r"\*?ST", na=False)
    st_count = df.groupby("code")["is_st"].max().sum()
    print(f"🏷️ ST股票: {st_count} 只")
    return df


def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """排除 ST 股票"""
    if not EXCLUDE_ST:
        return df
    st_codes = df.groupby("code")["is_st"].max()
    exclude = st_codes[st_codes].index.tolist()
    df = df[~df["code"].isin(exclude)].copy()
    print(f"  过滤后剩余 {df['code'].nunique()} 只 (剔除{len(exclude)}只ST)")
    return df


# ============================================================
# 4. 停牌检测
# ============================================================

def detect_suspension(df: pd.DataFrame) -> pd.DataFrame:
    """
    检测停牌日: 生成完整 code×date 矩阵, 缺失行=停牌。
    ⚠️ 不做前向填充! 填充会制造假信号。
    """
    df = df.copy()
    all_dates = sorted(df["date"].unique())
    all_codes = sorted(df["code"].unique())
    # 只对日期跨度大的股票填充 (避免内存爆炸, 用groupby方式)
    date_set = set(all_dates)

    suspension_records = []
    for code, group in df.groupby("code"):
        code_dates = set(group["date"])
        missing_dates = date_set - code_dates
        if missing_dates:
            for d in missing_dates:
                suspension_records.append({"date": d, "code": code})

    if suspension_records:
        susp_df = pd.DataFrame(suspension_records)
        susp_df["is_suspended"] = True
        susp_df["name"] = "未知"
        df = pd.concat([df, susp_df], ignore_index=True)
        df["is_suspended"] = df["is_suspended"].fillna(False)
    else:
        df["is_suspended"] = False

    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    susp_pct = df["is_suspended"].mean() * 100
    print(f"⏸️ 停牌日占比: {susp_pct:.1f}%")
    return df


def filter_long_suspensions(df: pd.DataFrame, max_days: int = MAX_SUSPEND_DAYS) -> pd.DataFrame:
    """剔除连续停牌超 max_days 的股票 (重大重组→行为不可预测)"""
    df = df.copy()
    df["suspend_group"] = (
        (df["is_suspended"] != df["is_suspended"].shift()).cumsum()
    )
    bad_codes = set()
    for code, group in df.groupby("code"):
        susp = group[group["is_suspended"]]
        for _, sg in susp.groupby("suspend_group"):
            if len(sg) > max_days:
                bad_codes.add(code)
                break
    if bad_codes:
        df = df[~df["code"].isin(bad_codes)]
        print(f"  剔除长停牌股票: {len(bad_codes)} 只")
    return df


# ============================================================
# 5. 复权一致性检查
# ============================================================

def check_adjust_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查并修复复权价格逻辑: high≥max(open,close), low≤min(open,close)
    """
    df = df.copy()
    bad_mask = (
        (df["high"] < df[["open", "close"]].max(axis=1)) |
        (df["low"] > df[["open", "close"]].min(axis=1)) |
        (df["close"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0)
    )
    n_bad = bad_mask.sum()
    if n_bad > 0:
        print(f"⚠️ 复权不一致行: {n_bad} ({n_bad/len(df)*100:.2f}%) → 修复")
        df.loc[bad_mask, "high"] = df.loc[bad_mask, ["open", "close", "high"]].max(axis=1)
        df.loc[bad_mask, "low"] = df.loc[bad_mask, ["open", "close", "low"]].min(axis=1)
    else:
        print("✅ 复权一致性检查通过")
    return df


# ============================================================
# 6. 微盘股/流动性过滤 (P5补丁)
# ============================================================

def filter_by_volume(df: pd.DataFrame, min_pct: float = MIN_VOLUME_PCT) -> pd.DataFrame:
    """
    按日均成交额过滤低流动性股票 (P5补丁: StockGPT S5)。
    不用市值(需要额外API), 直接用成交额分位数。
    """
    df = df.copy()
    avg_amount = df.groupby("code")["amount"].mean()
    threshold = avg_amount.quantile(min_pct / 100)
    bad_codes = avg_amount[avg_amount < threshold].index.tolist()
    if bad_codes:
        df = df[~df["code"].isin(bad_codes)]
        print(f"💧 低流动性过滤: 剔除 {len(bad_codes)} 只 (日均成交额<{threshold/1e4:.0f}万)")
    return df


# ============================================================
# 7. 缺失值处理
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理非停牌引起的缺失: 新股过滤 + 列级缺失填充"""
    df = df.copy()
    # 过滤交易天数过少的新股
    trading_days = df.groupby("code").size()
    new_codes = trading_days[trading_days < MIN_TRADING_DAYS].index.tolist()
    if new_codes:
        df = df[~df["code"].isin(new_codes)]
        print(f"🆕 剔除新股(交易日<{MIN_TRADING_DAYS}): {len(new_codes)} 只")

    # 关键列用前值填充 (按code分组, 不跨股票)
    fill_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df.groupby("code")[col].transform(lambda x: x.fillna(method="ffill"))

    df = df.dropna(subset=["close", "volume"])
    return df


# ============================================================
# 8. 新股过滤
# ============================================================

def filter_new_stocks(df: pd.DataFrame, min_days: int = MIN_TRADING_DAYS) -> pd.DataFrame:
    """剔除上市不足 min_days 天的次新股 (数据太少, 因子计算会大量NaN)"""
    df = df.copy()
    code_counts = df.groupby("code").size()
    valid = code_counts[code_counts >= min_days].index.tolist()
    removed = len(code_counts) - len(valid)
    if removed > 0:
        df = df[df["code"].isin(valid)]
        print(f"🔻 剔除次新股: {removed} 只 (交易日<{min_days})")
    return df


# ============================================================
# 9. 主流程
# ============================================================

def clean_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    一键清洗: 加载→ST标记→停牌→复权检查→流动性→缺失值→极值→保存
    """
    print(f"\n{'='*60}")
    print("🧹 数据清洗管线")
    print(f"{'='*60}")

    # 1. 加载
    df = load_raw_data(input_path)

    # 2. ST 标记 + 过滤
    df = mark_st_stocks(df)
    df = filter_st_stocks(df)

    # 3. 停牌检测
    df = detect_suspension(df)
    df = filter_long_suspensions(df)

    # 4. 复权一致性
    df = check_adjust_consistency(df)

    # 5. 微盘/流动性过滤 (P5)
    if "amount" in df.columns:
        df = filter_by_volume(df)

    # 6. 新股过滤
    df = filter_new_stocks(df)

    # 7. 缺失值
    df = handle_missing_values(df)

    # 8. Winsorization: 对收益率/变化率类列做极值处理
    # (此时还没有收益率列, 清洗阶段主要处理 volume 和 amount)
    ratio_cols = ["volume", "amount"]
    df = apply_winsorize(df, ratio_cols, method=WINSORIZE_METHOD)

    # 9. 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n✅ 清洗完成: {len(df):,} 行, {df['code'].nunique()} 只股票")
    print(f"   日期: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   输出: {output_path}")
    return df


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    raw_path = Path(__file__).resolve().parent.parent.parent / "data" / "a_stock_daily.csv"
    clean_path = Path(__file__).resolve().parent.parent.parent / "data" / "a_stock_daily_clean.parquet"

    if raw_path.exists():
        clean_pipeline(str(raw_path), str(clean_path))
    else:
        print(f"⚠️ 原始数据不存在: {raw_path}")
        print("   请先运行 python src/data/fetcher.py")
