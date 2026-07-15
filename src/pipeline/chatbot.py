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
    /ping      — 检查 bot 是否在线
    /status    — 查看系统状态 (GPU、实验数、模型数)
    /help      — 显示所有可用命令
    随便说话   — 友好提示

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

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# 复用 notifier.py 的配置加载函数
from src.pipeline.notifier import _load_telegram_config

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_LOG = DATA_DIR / "experiments.csv"


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
        "/status — 查看系统状态 (GPU / 实验数 / 模型数)\n"
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
    app = Application.builder().token(token).build()

    # 注册命令处理器
    # CommandHandler: 匹配 /xxx 格式的消息
    # MessageHandler:  匹配普通文字消息 (非命令)
    # filters:         过滤条件 (如 filters.TEXT 只处理文字)
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Bot 已上线! 在 Telegram 里给你的 bot 发消息试试")
    print("   命令: /ping /status /help")
    print("   Ctrl+C 停止\n")

    # 开始轮询 (阻塞)
    # run_polling() 会一直运行, 直到 Ctrl+C
    # poll_interval=1.0 → 每秒检查一次新消息
    # drop_pending_updates=True → 启动时丢弃 bot 离线期间积累的消息
    app.run_polling(poll_interval=1.0, drop_pending_updates=True)


if __name__ == "__main__":
    main()
