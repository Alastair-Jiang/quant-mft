"""
数据获取模块 - A股日线数据

用 akshare 拉取A股全市场（沪深京）日线OHLCV数据，存为CSV。

依赖：
- akshare: 免费金融数据接口，底层调东方财富/新浪API
- pandas: 数据处理，可以把DataFrame想象成"Python里的Excel表格"

作者: 蒋东旭
日期: 2026-07-14
"""

import akshare as ak
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# 配置参数
# ============================================================

# 数据存储目录（项目根目录下的 data 文件夹）
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# 每只股票拉多久的历史数据（一般A股3-5年就够了）
YEARS_OF_HISTORY = 3

# 每次请求间隔（秒），避免被封IP
# 东方财富是免费接口，请求太频繁会被ban
REQUEST_DELAY = 2


# ============================================================
# 核心函数
# ============================================================

def fetch_all_stock_codes():
    """
    获取所有A股股票代码列表

    akshare 的 stock_info_a_code_name()
    → 返回一个DataFrame，格式大概是这样：

    | code   | name     |
    |--------|----------|
    | 000001 | 平安银行 |
    | 000002 | 万科A    |
    | 600519 | 贵州茅台 |
    | ...    | ...      |

    注：包含沪深京三地交易所（沪市6开头，深市0/3开头，京市8开头）
    """
    print("📋 正在获取A股股票列表...")

    # akshare 函数：获取沪深京股票代码和名称
    df = ak.stock_info_a_code_name()

    print(f"   共获取 {len(df)} 只股票")
    return df


def fetch_one_stock_daily(code, name, start_date="20230101", end_date=None):
    """
    拉取单只股票的日线数据

    【类比】就像在东方财富网页上查一只股票的历史K线，然后导出Excel

    参数:
        code: 股票代码，如 "000001"
        name: 股票名称，如 "平安银行"
        start_date: 起始日期 "YYYYMMDD"
        end_date: 结束日期，留空=今天

    返回:
        DataFrame，列: date, open, high, low, close, volume, amount, code, name
    """

    # 如果没有指定结束日期，就用今天
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # 根据代码前缀判断交易所（决定akshare用哪个函数参数）
    if code.startswith("6"):
        symbol = f"sh{code}"  # 上海交易所 → sh
    else:
        symbol = f"sz{code}"  # 深圳/北京 → sz

    print(f"   📊 {code} {name} ({start_date} → {end_date})")

    # ---- 核心：akshare 日线数据接口 ----
    # stock_zh_a_hist 参数说明：
    #   symbol: 股票代码（带前缀sh/sz）
    #   period: "daily"=日线, "weekly"=周线, "monthly"=月线
    #   start_date/end_date: 日期范围
    #   adjust: "" = 不复权, "qfq" = 前复权, "hfq" = 后复权
    #   【重要】前复权会让历史价格按最新股本调整，更适合ML训练

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权：保证历史价格跟现在价格可比较
    )

    # 如果数据为空（比如新股还没交易），返回空DataFrame
    if df is None or len(df) == 0:
        print(f"   ⚠️ {code} {name} 无数据（可能停牌或新股）")
        return pd.DataFrame()

    # 加上股票代码和名称，方便之后多只股票合并
    df["code"] = code
    df["name"] = name

    # akshare 返回的列名是中文的，统一改成英文方便使用
    # 原始列名：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    # 我们保留核心的 OHLCV（Open/High/Low/Close/Volume）
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    })

    # 只保留需要的列
    columns_needed = ["date", "open", "high", "low", "close", "volume", "amount", "code", "name"]
    df = df[columns_needed]

    return df


def fetch_all_daily(start_date=None):
    """
    【主函数】拉取全市场日线数据

    流程：
    1. 先拿到所有股票代码
    2. 一只一只拉日线
    3. 合并成一个大的CSV保存

    参数:
        start_date: 起始日期，默认 = YEARS_OF_HISTORY年前
    """

    # ---- 1. 确定起始日期 ----
    if start_date is None:
        today = datetime.now()
        start = today - timedelta(days=YEARS_OF_HISTORY * 365)
        start_date = start.strftime("%Y%m%d")

    print(f"\n{'='*60}")
    print(f"🚀 开始拉取A股全市场日线数据（从 {start_date} 至今）")
    print(f"{'='*60}\n")

    # ---- 2. 获取股票列表 ----
    stocks = fetch_all_stock_codes()

    # ---- 3. 确保数据目录存在 ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 4. 逐只拉取数据 ----
    all_data = []  # 用一个列表收集每只股票的DataFrame

    for i, row in stocks.iterrows():
        code = row["code"]   # 股票代码列
        name = row["name"]   # 股票名称列

        try:
            df = fetch_one_stock_daily(code, name, start_date=start_date)
            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            # 某只股票拉失败了不要中断整个流程
            print(f"   ❌ {code} {name} 拉取失败: {e}")

    # ---- 5. 合并所有股票数据 ----
    if not all_data:
        print("\n❌ 没有拉到任何数据，请检查网络连接")
        return None

    # pd.concat = 把多个DataFrame纵向拼接（行堆叠）
    # 就像把多个Excel表首尾相连拼成一个大表
    full_data = pd.concat(all_data, ignore_index=True)

    # 按股票代码和日期排序
    full_data = full_data.sort_values(["code", "date"])

    # ---- 6. 保存为CSV ----
    output_path = DATA_DIR / "a_stock_daily.csv"
    full_data.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ---- 7. 输出统计 ----
    print(f"\n{'='*60}")
    print(f"✅ 数据拉取完成！")
    print(f"   股票数量: {full_data['code'].nunique()}")
    print(f"   总行数:   {len(full_data):,}")
    print(f"   日期范围: {full_data['date'].min()} → {full_data['date'].max()}")
    print(f"   保存路径: {output_path}")
    print(f"{'='*60}\n")

    return full_data


# ============================================================
# 入口：直接运行 python fetcher.py 时执行
# ============================================================
if __name__ == "__main__":
    import time

    start_time = time.time()

    # 跑！拉全市场数据
    data = fetch_all_daily()

    elapsed = time.time() - start_time
    print(f"⏱️ 总耗时: {elapsed/60:.1f} 分钟")
