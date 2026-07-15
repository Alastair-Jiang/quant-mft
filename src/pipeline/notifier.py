"""
Telegram 通知模块 — 训练完成/信号推送/异常告警

作用：
    通过 Telegram Bot 向用户推送消息。训练跑完了、出信号了、报错了，
    都会自动发到手机上，不用一直盯着终端。

为什么用 Telegram？
    - 免费、API 简单、国内不用翻墙就能用（对比 Discord/Slack）
    - 手机上实时收到推送
    - 可以双向交互（后期可以发指令给 bot）

用法：
    1. 先在 Telegram 搜索 @BotFather，创建 bot 拿到 token
    2. 把 token 写到 config/local.yaml → telegram.bot_token
    3. 给 bot 发任意一条消息（比如 "hi"）
    4. 运行 python -c "from src.pipeline.notifier import *; init_chat_id()"
       这会自动获取你的 chat_id 并保存到配置
    5. 之后就可以用 send_message() 发消息了

配置：
    config/local.yaml:
        telegram:
            bot_token: "123456:ABC..."
            chat_id: 123456789    # 你的 Telegram 用户 ID

依赖：
    python-telegram-bot (pip install python-telegram-bot)

作者: 蒋东旭
日期: 2026-07-15
"""

import sys
import io
# 修复 Windows GBK 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yaml
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

# ============================================================
# 配置加载
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _load_telegram_config() -> dict:
    """
    加载 Telegram 配置，local.yaml 覆盖 default.yaml

    为什么这样设计？
        - token 是密钥，存在 local.yaml（Git 忽略）
        - default.yaml 只放模板/默认值，可以提交到 GitHub
        - 代码里只调这个函数，不用到处重复读文件
    """
    config = {}

    # 先加载默认配置
    default_path = PROJECT_ROOT / "config" / "default.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            default_cfg = yaml.safe_load(f) or {}
            config.update(default_cfg.get("telegram", {}))

    # 再用本地配置覆盖（含 token）
    local_path = PROJECT_ROOT / "config" / "local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
            config.update(local_cfg.get("telegram", {}))

    return config


def _save_chat_id(chat_id: int):
    """把 chat_id 持久化到 local.yaml，下次不用重新获取"""
    local_path = PROJECT_ROOT / "config" / "local.yaml"

    # 读取现有内容
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # 更新 chat_id
    if "telegram" not in cfg:
        cfg["telegram"] = {}
    cfg["telegram"]["chat_id"] = chat_id

    with open(local_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


# ============================================================
# 1. 获取 chat_id
# ============================================================

async def _get_chat_id_async() -> Optional[int]:
    """
    从 bot 的 recent updates 中提取 chat_id。

    前提：你先给 bot 发过至少一条消息（任意内容）。
    然后运行这个函数，bot 会读取消息列表，找到你的 chat_id。

    为什么不用 /getUpdates 而用 python-telegram-bot？
        - python-telegram-bot 封装了所有 API 调用细节
        - 不需要手动拼 HTTP 请求
        - 出错时自动重试
    """
    from telegram import Bot
    from telegram.error import TelegramError

    cfg = _load_telegram_config()
    token = cfg.get("bot_token", "")
    if not token:
        print("❌ 未配置 bot_token，请在 config/local.yaml 中填写")
        return None

    bot = Bot(token=token)

    try:
        updates = await bot.get_updates(timeout=10)
        if not updates:
            print("⚠️ 没有收到任何消息。请先给你的 bot 发一条消息（比如 'hi'），然后重新运行。")
            return None

        # 取最近一条消息的发送者
        chat_id = updates[-1].message.chat_id
        sender = updates[-1].message.from_user
        print(f"✅ 获取到 chat_id: {chat_id}")
        print(f"   发送者: @{sender.username if sender.username else '未知'} ({sender.first_name})")
        return chat_id

    except TelegramError as e:
        print(f"❌ Telegram API 错误: {e}")
        print("   请检查: 1) token 是否正确 2) 网络是否能连上 api.telegram.org")
        return None


def init_chat_id():
    """
    初始化 chat_id：从 bot 消息历史自动获取并保存到配置。

    使用方式（终端运行）：
        python -c "from src.pipeline.notifier import init_chat_id; init_chat_id()"
    """
    chat_id = asyncio.run(_get_chat_id_async())
    if chat_id is not None:
        _save_chat_id(chat_id)
        print(f"💾 chat_id 已保存到 config/local.yaml")
        print(f"   现在可以用 send_message() 发消息了！")


# ============================================================
# 2. 发送消息
# ============================================================

def _send_sync(text: str, parse_mode: str = "HTML") -> bool:
    """
    同步发送消息（内部实现）。

    为什么用 HTML 格式？
        - Telegram 支持 <b>粗体</b> <i>斜体</i> <code>代码</code>
        - 比 Markdown 更不容易因为特殊字符报错
    """
    cfg = _load_telegram_config()
    token = cfg.get("bot_token", "")
    chat_id = cfg.get("chat_id", None)

    if not token:
        print("⚠️ Telegram: 未配置 bot_token")
        return False
    if not chat_id:
        print("⚠️ Telegram: 未配置 chat_id，请先运行 init_chat_id()")
        return False

    async def _send():
        from telegram import Bot
        from telegram.error import TelegramError
        bot = Bot(token=token)
        try:
            await bot.send_message(chat_id=chat_id, text=text,
                                   parse_mode=parse_mode,
                                   disable_web_page_preview=True)
            return True
        except TelegramError as e:
            print(f"⚠️ Telegram 发送失败: {e}")
            return False

    return asyncio.run(_send())


def send_message(text: str) -> bool:
    """
    发送一条纯文本消息到 Telegram。

    参数:
        text: 消息内容（支持 HTML 标签: <b> <i> <code> <pre> <a>）

    返回:
        True=发送成功, False=失败

    使用示例:
        send_message("训练完成！AUC: 0.53")
        send_message("<b>⚠️ 异常：</b>数据获取失败")
    """
    timestamp = datetime.now().strftime("%H:%M")
    msg = f"[{timestamp}] {text}"
    return _send_sync(msg)


def send_training_done(experiment_name: str, auc: float, acc: float,
                       epoch: int, elapsed: str, gpu_info: str = "") -> bool:
    """
    发送"训练完成"通知（格式化消息）。

    消息示例:
        ┌─────────────────────────┐
        │ ✅ 训练完成              │
        │ 实验: stockgpt_baseline  │
        │ AUC: 0.5234 | Acc: 51.2% │
        │ Epoch: 245 | 耗时: 2.3h  │
        │ GPU: RTX 5060 Ti (16GB)  │
        └─────────────────────────┘
    """
    lines = [
        f"<b>✅ 训练完成</b>",
        f"实验: <code>{experiment_name}</code>",
        f"AUC: <b>{auc:.4f}</b> | Acc: <b>{acc:.1%}</b>",
        f"Epoch: {epoch} | 耗时: {elapsed}",
    ]
    if gpu_info:
        lines.append(f"GPU: {gpu_info}")

    return _send_sync("\n".join(lines))


def send_all_experiments_done(n_done: int, n_total: int, elapsed: str,
                                top_result: str = "") -> bool:
    """
    发送"全部实验完成"通知。
    """
    lines = [
        f"<b>🏁 全部实验完成!</b>",
        f"完成: {n_done}/{n_total} | 总耗时: {elapsed}",
    ]
    if top_result:
        lines.append(f"最佳: {top_result}")

    return _send_sync("\n".join(lines))


def send_error(module: str, error_msg: str) -> bool:
    """
    发送错误告警通知。

    当训练/数据获取/任何关键流程出错时调用。
    """
    text = (
        f"<b>🚨 错误告警</b>\n"
        f"模块: <code>{module}</code>\n"
        f"错误: {error_msg[:500]}"  # 截断过长消息
    )
    return _send_sync(text)


def send_signal(code: str, name: str, prob: float, direction: str,
                expected_return: float, regime: str = "") -> bool:
    """
    推送交易信号（阶段4每日运行用）。

    消息示例:
        📈 买入信号: 000001 平安银行
        方向: 涨 | 概率: 58.3% | 预期收益: 1.2%
        市场状态: 震荡市
    """
    emoji = "📈" if direction == "涨" else "📉"
    lines = [
        f"{emoji} <b>信号: {code} {name}</b>",
        f"方向: {direction} | 概率: <b>{prob:.1%}</b> | 预期收益: <b>{expected_return:+.2%}</b>",
    ]
    if regime:
        lines.append(f"市场状态: {regime}")

    return _send_sync("\n".join(lines))


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        # python src/pipeline/notifier.py init
        init_chat_id()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # python src/pipeline/notifier.py test → 发送测试消息
        ok = send_message("🧪 这是一条测试消息。如果你看到它，说明 Telegram Bot 配置成功！")
        if ok:
            print("✅ 测试消息发送成功")
        else:
            print("❌ 发送失败，请检查 token 和 chat_id")
    else:
        print("Telegram 通知模块")
        print("  用法:")
        print("    python src/pipeline/notifier.py init    → 获取 chat_id")
        print("    python src/pipeline/notifier.py test    → 发送测试消息")
