"""
Telegram 双向聊天 Bot — 接收指令, 查询状态, 触发操作

作用：
    你可以在手机上给 bot 发命令，bot 会回复你。
    比如在吃饭时想看看训练状态，不用跑回电脑前，掏出手机发 /status 就行。

和 notifier.py 的区别：
    - notifier.py: 单向推送 (系统 → 你)
    - chatbot.py:  双向对话 (你 ←→ 系统)

原理：
    python-telegram-bot 的 Application 会持续轮询 Telegram 服务器，
    检测到新消息后判断是不是命令（以 / 开头），是就调用对应的处理函数。

支持的命令：
    /ping        — 检查 bot 是否在线
    /status      — 查看系统状态 (GPU、实验、模型、断点)
    /experiments — 实验排名 (Top 5, 按 AUC 排序)
    /models      — 已保存的模型文件列表
    /checkpoint  — 未完成的训练断点
    /signals     — 最新交易信号
    /help        — 显示所有可用命令
    随便说话      — 关键词智能引导

运行方式:
    python src/pipeline/chatbot.py
    Ctrl+C 停止

扩展:
    后期可以加 /train 触发训练, /signals 获取今日信号等。

依赖:
    python-telegram-bot (已安装)

作者: 蒋东旭
日期: 2026-07-15
"""

import sys
import io
# 修复 Windows GBK 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from datetime import datetime
import pandas as pd
import torch

# 确保项目根目录在 Python 搜索路径中
# (直接运行 chatbot.py 时, src/ 不在 sys.path)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

# 复用 notifier.py 的配置加载函数
from src.pipeline.notifier import _load_telegram_config


# ============================================================
# 构建 Application (带自定义 HTTP 设置)
# ============================================================

def build_application(token: str) -> Application:
    """
    构建 Telegram Application，配置超时和连接池。

    单独写成函数而非在 main() 里直接写，是为了方便后期加代理等配置。
    """
    # connect_timeout: 建立连接的超时 (秒), 默认5秒在墙内偶尔不够
    # read_timeout:   等待响应的超时 (秒)
    # write_timeout:  发送数据的超时 (秒)
    # pool_timeout:   从连接池获取连接的超时
    request = HTTPXRequest(
        connect_timeout=15.0,   # 从5秒提高到15秒
        read_timeout=30.0,      # 从默认5秒提高到30秒
        write_timeout=15.0,
        pool_timeout=5.0,
    )
    return Application.builder().token(token).request(request).build()

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_LOG = DATA_DIR / "experiments.csv"
SIGNALS_FILE = DATA_DIR / "signals.csv"


# ============================================================
# 命令处理函数
# ============================================================

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /ping — 验证 bot 是否在线

    最简单的命令，收到 /ping 就回 pong。
    类比: 像 ping 8.8.8.8 测网络通不通一样，用来确认 bot 还活着。
    """
    await update.message.reply_text("🏓 pong!")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /help — 显示所有可用命令

    用户忘了命令时发 /help 就能看到列表。
    """
    text = (
        "<b>📋 可用命令</b>\n"
        "\n"
        "/ping — 检查 bot 是否在线\n"
        "/status — 系统状态 (GPU / 实验 / 模型 / 断点)\n"
        "/experiments — 实验排名 (Top 5)\n"
        "/models — 已保存的模型列表\n"
        "/checkpoint — 未完成的训练断点\n"
        "/signals — 最新交易信号\n"
        "/help — 显示此帮助信息"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /status — 查看当前系统状态

    回复内容:
        - GPU 可用性 + 型号
        - 已完成的实验数量
        - 已保存的模型数量
        - 当前时间
        - 是否有未完成的断点

    为什么要查状态？
        你可能在出门前想知道训练跑得怎么样了，
        掏出手机发 /status 就全知道了，不用远程连回电脑。
    """
    lines = ["<b>📊 系统状态</b>\n"]

    # ---- GPU ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f"🖥 GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        lines.append("🖥 GPU: 无 (CPU模式)")

    # ---- 已完成实验 ----
    if EXPERIMENT_LOG.exists():
        df = pd.read_csv(EXPERIMENT_LOG)
        n_exp = len(df)
        lines.append(f"📈 已完成实验: {n_exp} 组")
        if "tf_best_val_auc" in df.columns and n_exp > 0:
            best_row = df.loc[df["tf_best_val_auc"].idxmax()]
            lines.append(f"   最佳: {best_row['experiment']} (AUC={best_row['tf_best_val_auc']:.4f})")
    else:
        lines.append("📈 已完成实验: 0 组")

    # ---- 已保存模型 ----
    if MODEL_DIR.exists():
        tf_models = list(MODEL_DIR.glob("transformer_*.pt"))
        lgb_models = list(MODEL_DIR.glob("lightgbm_*.txt"))
        lines.append(f"💾 已保存模型: {len(tf_models)} Transformer + {len(lgb_models)} LightGBM")

    # ---- 未完成断点 ----
    ckpt_dir = MODEL_DIR / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.pt"))
        if ckpts:
            lines.append(f"⏳ 未完成断点: {len(ckpts)} 个")
            for ckpt in ckpts[:3]:  # 最多显示3个
                lines.append(f"   - {ckpt.stem}")

    # ---- 时间 ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n🕐 {now}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


def _fmt_size(size_bytes: int) -> str:
    """把字节数转成人类可读的文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


async def cmd_experiments(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /experiments — 实验排名 (Top 5, 按 Transformer AUC 排序)

    数据来源: data/experiments.csv
    """
    if not EXPERIMENT_LOG.exists():
        await update.message.reply_text("📊 暂无实验记录\n   先跑一次训练，结果会自动记录到 experiments.csv")
        return

    df = pd.read_csv(EXPERIMENT_LOG)
    if len(df) == 0:
        await update.message.reply_text("📊 实验记录为空")
        return

    lines = [f"<b>📊 实验排名</b> (共 {len(df)} 组)\n"]

    if "tf_best_val_auc" in df.columns:
        df_sorted = df.sort_values("tf_best_val_auc", ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
            medal = ["🥇", "🥈", "🥉", "4.", "5."][i]
            lines.append(
                f"{medal} <code>{row['experiment']}</code>\n"
                f"   TF_AUC <b>{row['tf_best_val_auc']:.4f}</b> | "
                f"LGB_AUC {row['lgb_val_auc']:.4f} | "
                f"参数 {row['n_params']:,}"
            )
    else:
        lines.append("⚠️ 实验记录格式不符 (缺少 tf_best_val_auc 列)")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /models — 已保存的模型文件列表
    """
    if not MODEL_DIR.exists():
        await update.message.reply_text("💾 models/ 目录不存在")
        return

    tf_models = sorted(MODEL_DIR.glob("transformer_*.pt"))
    lgb_models = sorted(MODEL_DIR.glob("lightgbm_*.txt"))

    if not tf_models and not lgb_models:
        await update.message.reply_text("💾 还没有保存的模型\n   跑一次训练就会自动保存")
        return

    lines = ["<b>💾 已保存模型</b>\n"]

    if tf_models:
        lines.append(f"<b>Transformer ({len(tf_models)} 个):</b>")
        for m in tf_models:
            lines.append(f"  - {m.name} ({_fmt_size(m.stat().st_size)})")

    if lgb_models:
        lines.append(f"\n<b>LightGBM ({len(lgb_models)} 个):</b>")
        for m in lgb_models:
            lines.append(f"  - {m.name} ({_fmt_size(m.stat().st_size)})")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_checkpoint(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /checkpoint — 查看未完成的训练断点
    """
    ckpt_dir = MODEL_DIR / "checkpoints"

    if not ckpt_dir.exists():
        await update.message.reply_text("✅ 无未完成断点\n   断节目录尚未创建")
        return

    ckpts = list(ckpt_dir.glob("*.pt"))
    if not ckpts:
        await update.message.reply_text("✅ 无未完成断点\n   所有训练都已正常完成")
        return

    lines = [f"<b>⏳ 未完成断点</b> ({len(ckpts)} 个)\n"]

    for ckpt in ckpts:
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=True)
            epoch = data.get("epoch", "?")
            best_epoch = data.get("best_epoch", "?")
            best_loss = data.get("best_val_loss", float("nan"))
            name = ckpt.stem
            lines.append(
                f"  <code>{name}</code>\n"
                f"   epoch {epoch} | 最佳@epoch {best_epoch} (loss={best_loss:.4f})"
            )
        except Exception:
            lines.append(f"  <code>{ckpt.stem}</code> (无法读取详情)")

    lines.append(f"\n💡 下次运行训练时会自动从断点继续")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /signals — 最新交易信号

    数据来源: data/signals.csv (由 strategy/signal_generator.py 生成)
    """
    if not SIGNALS_FILE.exists():
        await update.message.reply_text(
            "📈 暂无交易信号\n\n"
            "<i>signals.csv 不存在</i> — 需要先跑策略生成信号:\n"
            "<code>python src/strategy/signal_generator.py</code>"
        )
        return

    df = pd.read_csv(SIGNALS_FILE)
    if len(df) == 0:
        await update.message.reply_text("📈 信号文件为空，暂无信号")
        return

    # 取最近日期的最新一批信号（最多显示 5 个）
    if "date" in df.columns:
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].head(5)
        date_str = str(latest_date)
    else:
        latest = df.head(5)
        date_str = "最近"

    lines = [f"<b>📈 最新信号</b> ({date_str})\n"]

    for _, row in latest.iterrows():
        code = row.get("code", "?")
        name = row.get("name", "")
        prob = row.get("prob", 0)
        direction = row.get("direction", "?")
        emoji = "📈" if "涨" in str(direction) else "📉"
        lines.append(
            f"{emoji} <code>{code}</code> {name}\n"
            f"   方向: {direction} | 概率: <b>{prob:.1%}</b>"
        )
        if "expected_return" in row:
            lines[-1] += f" | 预期收益: <b>{row['expected_return']:+.2%}</b>"

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    处理非命令消息 (普通文字)

    策略: 不忽略，给一个友好的提示告诉用户可以做什么。
    如果消息里提到特定关键词 (如"训练"、"状态")，可以智能引导。
    """
    text = update.message.text.strip().lower()

    # 关键词智能引导
    if any(kw in text for kw in ["状态", "status", "怎么样", "进度"]):
        await cmd_status(update, context)
    elif any(kw in text for kw in ["实验", "experiment", "排名", "auc"]):
        await cmd_experiments(update, context)
    elif any(kw in text for kw in ["模型", "model", "保存"]):
        await cmd_models(update, context)
    elif any(kw in text for kw in ["断点", "checkpoint", "续训"]):
        await cmd_checkpoint(update, context)
    elif any(kw in text for kw in ["信号", "signal", "买卖"]):
        await cmd_signals(update, context)
    elif any(kw in text for kw in ["训练", "跑", "train", "开始"]):
        await update.message.reply_text(
            "💡 目前还不支持通过聊天触发训练，请去电脑上运行。\n"
            "试试 /status 查看当前状态?"
        )
    elif any(kw in text for kw in ["帮助", "help", "命令", "功能"]):
        await cmd_help(update, context)
    else:
        await update.message.reply_text(
            "👋 你好! 试试这些命令:\n\n"
            "/ping — 检查在线状态\n"
            "/status — 查看系统状态\n"
            "/experiments — 实验排名\n"
            "/models — 模型列表\n"
            "/help — 显示帮助"
        )


# ============================================================
# 主程序
# ============================================================

def main():
    """启动聊天 Bot (阻塞运行，Ctrl+C 停止)"""
    cfg = _load_telegram_config()
    token = cfg.get("bot_token", "")

    if not token:
        print("❌ 未配置 bot_token! 请检查 config/local.yaml")
        return

    print(f"{'='*50}")
    print(f"🤖 Telegram 聊天 Bot 启动中...")
    print(f"{'='*50}")

    # 创建 Application
    # Application 是 python-telegram-bot v20+ 的核心类,
    # 负责管理网络连接、轮询消息、分发到 handler
    app = build_application(token)

    # 注册命令处理器
    # CommandHandler: 匹配 /xxx 格式的消息
    # MessageHandler:  匹配普通文字消息 (非命令)
    # filters:         过滤条件 (如 filters.TEXT 只处理文字)
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("experiments", cmd_experiments))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("checkpoint", cmd_checkpoint))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot 已上线! 在 Telegram 里给你的 bot 发消息试试")
    print("   命令: /ping /status /experiments /models /checkpoint /signals /help")
    print("   Ctrl+C 停止\n")

    # 开始轮询 (阻塞)
    # run_polling() 会一直运行, 直到 Ctrl+C
    # poll_interval=1.0 → 每秒检查一次新消息
    # drop_pending_updates=True → 启动时丢弃 bot 离线期间积累的消息
    app.run_polling(poll_interval=1.0, drop_pending_updates=True)


if __name__ == "__main__":
    main()
