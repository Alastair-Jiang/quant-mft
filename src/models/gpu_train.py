"""
GPU 训练脚本 — 5060 Ti 16GB 主力训练

双线对比:
  线路A (GPU): 合成数据生成 → Transformer 预训练 → 分布预测
  线路B (CPU): 合成数据生成 → LightGBM baseline → 性能对比

合成数据设计 (对应 docs/reference-model-analysis.md 的合成数据方案):
  - 第一层: HMM 市场状态 (牛市/震荡/熊市)
  - 第二层: 状态相关因子→收益映射 (规律不唯一, 随状态切换)
  - 第三层: GARCH(1,1) + t-分布噪声 (fat tail + volatility clustering)
  - 第四层: 截面相关性 (行业 + 市场因子)

训练产物:
  - models/transformer_*.pt     (Transformer 模型权重)
  - models/lightgbm_*.txt       (LightGBM 模型)
  - data/experiments.csv        (实验记录)

作者: 蒋东旭
日期: 2026-07-15
"""

import sys
import io
# 修复 Windows GBK 终端编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
import json
import time
import warnings
from datetime import datetime
from typing import Tuple, Dict, List, Optional
warnings.filterwarnings("ignore")

# Telegram 通知（可选，未配置则跳过）
try:
    from src.pipeline.notifier import send_training_done, send_all_experiments_done, send_error
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False

# ============================================================
# 全局配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 合成数据规模 — 对齐 32GB RAM + 16GB VRAM
N_DAYS = 20_000
N_STOCKS = 500
N_REGIMES = 3
N_FEATURES = 20
SEQ_LEN = 128          # 序列长度 (平衡显存)
N_RETURN_BINS = 6

# Transformer 架构
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.1
BATCH_SIZE = 1024      # 默认batch
LEARNING_RATE = 1e-4
N_EPOCHS = 30
USE_AMP = True
NUM_WORKERS = 0  # Windows spawn 模式不支持多进程

# LightGBM
LGB_PARAMS = {
    "objective": "multiclass", "num_class": N_RETURN_BINS,
    "metric": "multi_logloss", "num_leaves": 31, "learning_rate": 0.01,
    "n_estimators": 500, "min_child_samples": 100, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "random_state": 42, "verbose": -1,
}

# 实验记录
EXPERIMENT_LOG = DATA_DIR / "experiments.csv"

# 断点续训
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"    # 断点保存目录
CHECKPOINT_EVERY_N_EPOCHS = 5                  # 每5个epoch存一次断点


# ============================================================
# Part A: 合成数据生成器
# ============================================================

class SyntheticMarketGenerator:
    """
    分层合成市场数据生成器。

    四层结构:
      1. HMM 市场状态序列 (牛市/震荡/熊市)
      2. 状态相关因子→收益映射 (规律随状态切换)
      3. GARCH(1,1) + Student-t 噪声 (fat tail + volatility clustering)
      4. 截面相关性 (市场共同因子 + 行业因子)
    """

    def __init__(self, n_days: int = N_DAYS, n_stocks: int = N_STOCKS,
                 n_features: int = N_FEATURES, n_regimes: int = N_REGIMES,
                 seed: int = 42):
        self.n_days = n_days
        self.n_stocks = n_stocks
        self.n_features = n_features
        self.n_regimes = n_regimes
        self.rng = np.random.RandomState(seed)

        # HMM 状态转移矩阵 (高自相关: 状态有惯性)
        self.trans_mat = np.array([
            [0.95, 0.04, 0.01],  # 牛市: 5%概率切换
            [0.03, 0.92, 0.05],  # 震荡: 8%概率切换
            [0.02, 0.06, 0.92],  # 熊市: 8%概率切换
        ])

        # 每个状态的因子权重 (规律不唯一!)
        self.regime_weights = {
            0: np.array([0.3, 0.15, -0.05, 0.1, 0.05, -0.02, 0.08, 0.12, 0.03, -0.04,
                         0.06, 0.02, 0.07, -0.03, 0.04, 0.01, 0.05, -0.01, 0.02, 0.03]),  # 牛市: 动量+趋势主导
            1: np.array([-0.1, 0.05, 0.02, -0.15, 0.08, -0.06, -0.03, 0.04, -0.08, -0.12,
                         0.02, -0.01, 0.05, 0.03, -0.02, 0.01, -0.04, 0.02, -0.01, 0.01]),  # 震荡: 反转+均线回归
            2: np.array([-0.2, -0.1, 0.05, -0.05, -0.15, 0.02, -0.08, -0.06, 0.01, -0.03,
                        -0.1, 0.03, -0.05, 0.02, -0.04, 0.01, -0.07, 0.01, -0.02, 0.01]),  # 熊市: 波动率+防御主导
        }

        # 每只股票对因子的敏感度 (截面异质性)
        self.stock_betas = self.rng.normal(1.0, 0.3, (n_stocks, n_features))
        self.stock_betas = np.clip(self.stock_betas, 0.3, 2.0)

    def _generate_regime_sequence(self) -> np.ndarray:
        """HMM 生成市场状态序列"""
        regimes = np.zeros(self.n_days, dtype=int)
        regimes[0] = self.rng.choice(self.n_regimes, p=[0.3, 0.5, 0.2])
        for t in range(1, self.n_days):
            regimes[t] = self.rng.choice(self.n_regimes, p=self.trans_mat[regimes[t-1]])
        return regimes

    def _generate_factor_series(self, regimes: np.ndarray) -> np.ndarray:
        """
        生成因子时间序列 (带状态相关的均值偏移)。
        因子 = 状态偏移 + AR(1) + 噪声
        """
        regime_offsets = {
            0: 0.001,   # 牛市: 因子略偏正
            1: 0.0,     # 震荡: 零均值
            2: -0.001,  # 熊市: 因子略偏负
        }

        factors = np.zeros((self.n_days, self.n_features))
        # 初始值
        factors[0] = self.rng.normal(0, 0.01, self.n_features)

        for t in range(1, self.n_days):
            offset = regime_offsets[regimes[t]]
            ar_term = 0.3 * factors[t-1]    # AR(1), 弱自相关
            noise = self.rng.normal(offset, 0.01, self.n_features)
            factors[t] = ar_term + noise

        return factors

    def _generate_garch_noise(self, n: int, regimes: np.ndarray) -> np.ndarray:
        """GARCH(1,1) + Student-t(ν=3) 噪声 (fat tail + volatility clustering)"""
        # 每个状态的基准波动率
        regime_sigma = {0: 0.008, 1: 0.012, 2: 0.020}

        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = 0.01 ** 2

        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        nu = 3  # t-分布自由度 (越小越肥尾)

        for t in range(1, n):
            sigma2[t] = omega + alpha * (returns[t-1]**2 / sigma2[t-1]) * sigma2[t-1] + beta * sigma2[t-1]
            sigma = np.sqrt(sigma2[t]) * regime_sigma[regimes[t]] / 0.01
            returns[t] = sigma * self.rng.standard_t(nu)

        return returns

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        生成完整合成数据集。

        返回:
          features_df: (n_days × n_stocks, n_features+4) 因子 + 元数据
          returns_df:  (n_days × n_stocks, 3) 真实收益 + 信号 + 噪声
          regimes:     (n_days,) 真实市场状态 (用于验证)
          noise_levels:(n_days,) 每日信噪比 (用于验证)
        """
        print("\n🏭 生成合成市场数据...")

        # 1. 状态序列
        regimes = self._generate_regime_sequence()
        regime_counts = pd.Series(regimes).value_counts()
        regime_names = {0: "牛市", 1: "震荡", 2: "熊市"}
        for r, c in regime_counts.items():
            print(f"   {regime_names[r]}: {c} 天 ({c/len(regimes)*100:.1f}%)")

        # 2. 因子序列
        print(f"   生成因子: {N_FEATURES} 个 × {self.n_days} 天...")
        market_factors = self._generate_factor_series(regimes)

        # 3. GARCH 噪声 (全市场级)
        print("   生成噪声: GARCH(1,1)+t(3)...")
        market_noise = self._generate_garch_noise(self.n_days, regimes)

        # 4. 构建每只股票的数据
        print(f"   构建截面: {self.n_stocks} 只股票 × {self.n_days} 天...")
        all_features = []
        all_returns = []
        all_regimes = []
        all_noise_levels = []

        for day in tqdm(range(self.n_days), desc="生成截面数据", unit="天"):
            regime = regimes[day]
            weights = self.regime_weights[regime]
            f_day = market_factors[day]  # (n_features,)
            n_day = market_noise[day]     # scalar

            # 每只股票的因子暴露 = 市场因子 × 股票敏感度 + 个股噪声
            stock_factors = (f_day * self.stock_betas +
                            self.rng.normal(0, 0.002, (self.n_stocks, self.n_features)))

            # 预期收益 = 因子权重 · 因子值 (截面: 每只股票不同)
            expected_rets = stock_factors @ weights  # (n_stocks,)

            # 实际收益 = 预期收益 + 噪声 × 个股波动率
            idio_noise = self.rng.standard_t(3, self.n_stocks) * 0.005
            actual_rets = expected_rets + market_noise[day] * 0.5 + idio_noise

            # 信噪比
            signal_var = np.var(expected_rets)
            noise_var = np.var(actual_rets - expected_rets)
            snr = signal_var / (signal_var + noise_var + 1e-10)

            all_features.append(stock_factors)
            all_returns.append(np.column_stack([actual_rets, expected_rets, np.full_like(actual_rets, market_noise[day])]))
            all_regimes.extend([regime] * self.n_stocks)
            all_noise_levels.extend([snr] * self.n_stocks)

        features = np.vstack(all_features)
        returns = np.vstack(all_returns)

        # 转 DataFrame
        feat_cols = [f"factor_{i}" for i in range(self.n_features)]
        features_df = pd.DataFrame(features, columns=feat_cols)
        features_df["day"] = np.repeat(np.arange(self.n_days), self.n_stocks)
        features_df["stock_id"] = np.tile(np.arange(self.n_stocks), self.n_days)
        features_df["regime"] = all_regimes
        features_df["snr"] = all_noise_levels

        returns_df = pd.DataFrame(returns, columns=["actual_return", "expected_return", "market_noise"])
        # 裁剪极端值 + 填充 NaN, 防止 pd.cut 产生 NaN label
        returns_df["actual_return"] = returns_df["actual_return"].clip(-0.3, 0.3).fillna(0)
        returns_df["target_bin"] = pd.cut(
            returns_df["actual_return"],
            bins=[-np.inf, -0.05, -0.02, 0, 0.02, 0.05, np.inf],
            labels=False
        ).fillna(3).astype(int)  # NaN → 微涨(3)兜底
        returns_df["target_dir"] = (returns_df["actual_return"] > 0).astype(int)

        print(f"   ✅ 合成数据: {len(features_df):,} 样本 | 信噪比均值 {np.mean(all_noise_levels):.3f}\n")
        return features_df, returns_df, regimes, np.array(all_noise_levels)


# ============================================================
# Part B: Transformer 模型 (StockGPT 风格)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)])


class StockTransformer(nn.Module):
    """
    Decoder-only Transformer for stock return prediction.
    类似 StockGPT 架构: 输入特征序列 → 预测下一个收益区间。

    参数规模: ~0.8M (轻量, 适合 16GB VRAM 做大量实验)
    """

    def __init__(self, n_features: int = N_FEATURES, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 n_bins: int = N_RETURN_BINS, dropout: float = DROPOUT,
                 seq_len: int = SEQ_LEN):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_bins),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.head(x[:, -1, :])  # 只用最后一个时间步
        return x


# ============================================================
# Part C: 数据集
# ============================================================

class StockDataset(Dataset):
    """惰性加载: 存索引不存数据, 避免 474GB 爆炸"""

    def __init__(self, features_df, returns_df, seq_len=SEQ_LEN, n_features=N_FEATURES,
                 n_stocks=N_STOCKS, n_days=N_DAYS):
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_stocks = n_stocks
        self.n_days = n_days

        feat_cols = [f"factor_{i}" for i in range(n_features)]

        # 只存一份原始数组, 不复制; 清理残留 NaN
        raw_feat = features_df[feat_cols].values.reshape(n_days, n_stocks, n_features).astype(np.float32)
        raw_feat = np.nan_to_num(raw_feat, nan=0.0)
        self.feat_array = raw_feat
        raw_ret = returns_df["target_bin"].values.reshape(n_days, n_stocks).astype(np.float32)
        raw_ret = np.nan_to_num(raw_ret, nan=3.0)
        self.ret_array = raw_ret.astype(np.int64)
        raw_dir = returns_df["target_dir"].values.reshape(n_days, n_stocks).astype(np.float32)
        raw_dir = np.nan_to_num(raw_dir, nan=0.0)
        self.dir_array = raw_dir.astype(np.int64)

        # 构建索引: [(stock_id, t_start), ...]
        self.indices = []
        for s in range(n_stocks):
            for t in range(seq_len, n_days - 1):
                self.indices.append((s, t))

        print(f"   Dataset: {len(self.indices):,} 样本 (seq={seq_len}, feat={n_features}, "
              f"内存: {self.feat_array.nbytes/1024**2:.0f}MB)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s, t = self.indices[idx]
        # 切片 → 不复制, 但 PyTorch tensor 会复制
        seq = torch.tensor(self.feat_array[t - self.seq_len:t, s, :])
        target_bin = torch.tensor(self.ret_array[t + 1, s])
        target_dir = torch.tensor(self.dir_array[t + 1, s])
        return seq, target_bin, target_dir


# ============================================================
# Part C-extra: 断点续训
# ============================================================

def save_checkpoint(checkpoint_name: str, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler,
                    scaler, best_val_loss: float, best_epoch: int,
                    best_state: dict, patience_counter: int,
                    history: dict):
    """
    保存训练断点到磁盘, 用于中断后从断点继续。

    保存的内容比模型文件多：
        - 优化器状态 → 恢复后 momentum/Adam 的二阶矩不会丢失
        - scheduler 状态 → 学习率曲线继续而非从头衰减
        - AMP scaler 状态 → 混合精度损失缩放比例继续
        - early stopping 计数器 → 不会因为中断重置计数值

    参数:
        checkpoint_name: 断点文件名 (不含路径)
        其余: 训练循环中的当前状态变量
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"

    scaler_state = scaler.state_dict() if scaler is not None else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler_state,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "patience_counter": patience_counter,
        "history": history,
    }
    torch.save(checkpoint, path)


def load_checkpoint(checkpoint_name: str) -> dict | None:
    """
    尝试加载之前的断点。

    返回:
        checkpoint dict 或 None (没有断点)
    """
    path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    if not path.exists():
        return None
    print(f"📂 发现断点: {path.name} ({path.stat().st_size/1024:.0f}KB)")
    return torch.load(path, map_location=DEVICE)


# ============================================================
# Part D: 训练循环
# ============================================================

def train_transformer(model, train_loader, val_loader, n_epochs=N_EPOCHS,
                      lr=LEARNING_RATE, device=DEVICE,
                      use_amp: bool = USE_AMP,
                      checkpoint_name: str = None,
                      resume: bool = True) -> Dict:
    """
    训练 Transformer, 返回最佳模型和训练历史。

    断点续训机制:
        - 如果 resume=True 且 checkpoint 文件存在 → 恢复状态从断点继续
        - 每 CHECKPOINT_EVERY_N_EPOCHS 自动存一次断点
        - 训练完成(含 early stop)后自动删除断点文件
        - Ctrl+C 中断时自动保存断点, 下次自动续训

    参数:
        checkpoint_name: 断点文件名 (不含路径和扩展名), None=不启用断点
        resume: 是否尝试从断点恢复
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 初始状态 ----
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 50    # 50 epoch不降才停
    min_delta = 1e-5            # 低于此视为无改善
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": [], "epoch_time": []}
    start_epoch = 0

    # ---- 尝试从断点恢复 ----
    if resume and checkpoint_name is not None:
        checkpoint = load_checkpoint(checkpoint_name)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint["scaler_state_dict"] is not None and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            best_val_loss = checkpoint["best_val_loss"]
            best_epoch = checkpoint["best_epoch"]
            best_state = checkpoint["best_state"]
            patience_counter = checkpoint["patience_counter"]
            history = checkpoint["history"]
            start_epoch = checkpoint["epoch"] + 1  # 从下一个epoch开始
            print(f"🔄 从断点恢复: epoch {start_epoch}/{n_epochs} | "
                  f"最佳@epoch {best_epoch+1} (val_loss={best_val_loss:.4f}) | "
                  f"耐心计数 {patience_counter}/{early_stop_patience}")
            # checkpoint加载后立即删除, 避免重复使用
            (CHECKPOINT_DIR / f"{checkpoint_name}.pt").unlink()
            print(f"🗑 已清理断点文件, 训练完成后会重新保存")

    print(f"\n🚀 Transformer 训练 | {sum(p.numel() for p in model.parameters()):,} 参数 | {device}")
    print(f"   AMP混合精度: {use_amp} | Batch: {train_loader.batch_size} | Seq: {model.seq_len}")
    print(f"   EarlyStopping: patience={early_stop_patience} min_delta={min_delta}")
    if checkpoint_name:
        print(f"   断点: 每{CHECKPOINT_EVERY_N_EPOCHS}epoch自动保存")
    print(f"{'='*60}")

    try:
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()

            # Train — 每个epoch一个batch级进度条
            model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{n_epochs} Train",
                              unit="batch", leave=False,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for x, y_bin, _ in train_pbar:
                x, y_bin = x.to(device, non_blocking=True), y_bin.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y_bin)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * x.size(0)
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss /= len(train_loader.dataset)

            # Val — batch级进度条
            model.eval()
            val_loss = 0.0
            all_probs, all_labels_bin, all_labels_dir = [], [], []
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{n_epochs} Val  ",
                            unit="batch", leave=False,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    for x, y_bin, y_dir in val_pbar:
                        x, y_bin = x.to(device, non_blocking=True), y_bin.to(device, non_blocking=True)
                        logits = model(x)
                        loss = criterion(logits, y_bin)
                        val_loss += loss.item() * x.size(0)
                        probs = F.softmax(logits, dim=-1)
                        all_probs.append(probs.cpu().numpy())
                        all_labels_bin.append(y_bin.cpu().numpy())
                        all_labels_dir.append(y_dir.numpy())
            val_loss /= len(val_loader.dataset)

            probs = np.vstack(all_probs)
            labels_bin = np.concatenate(all_labels_bin)
            labels_dir = np.concatenate(all_labels_dir)

            # 方向准确率 (涨 vs 跌)
            dir_prob = probs[:, 3:].sum(axis=1)  # 后3个bin=涨
            val_acc = accuracy_score(labels_dir, dir_prob > 0.5)
            val_auc = roc_auc_score(labels_dir, dir_prob)

            scheduler.step(val_loss)
            epoch_time = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)
            history["epoch_time"].append(epoch_time)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
                improved = "*"
            else:
                patience_counter += 1
                improved = " "

            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{n_epochs} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | acc {val_acc:.3f} | auc {val_auc:.4f} | "
                  f"lr {lr_now:.1e} | {epoch_time:.0f}s{improved}")

            # ---- 定期保存断点 ----
            if checkpoint_name is not None and (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
                save_checkpoint(checkpoint_name, epoch, model, optimizer, scheduler,
                                scaler, best_val_loss, best_epoch, best_state,
                                patience_counter, history)
                print(f"   💾 断点已保存 (epoch {epoch+1})")

            # Early stopping: patience轮不改善则停止
            if patience_counter >= early_stop_patience:
                print(f"⏹ EarlyStopping at epoch {epoch+1}: val_loss未改善{early_stop_patience}轮 "
                      f"(best@{best_epoch+1}: {best_val_loss:.4f})")
                break

    except KeyboardInterrupt:
        # Ctrl+C 优雅中断 → 保存断点
        if checkpoint_name is not None:
            save_checkpoint(checkpoint_name, epoch, model, optimizer, scheduler,
                            scaler, best_val_loss, best_epoch, best_state,
                            patience_counter, history)
            print(f"\n⏸ Ctrl+C 中断! 断点已保存到 {CHECKPOINT_DIR / checkpoint_name}.pt")
            print(f"   📍 当前进度: epoch {epoch+1}/{n_epochs} | 最佳@epoch {best_epoch+1}")
            print(f"   💡 下次运行会自动从断点继续")
        raise

    # 训练完成 → 清理断点
    if checkpoint_name is not None:
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"🗑 训练完成, 断点文件已清理")

    # 恢复最佳模型
    model.load_state_dict(best_state)
    print(f"✅ 最佳 epoch={best_epoch+1}/{epoch+1} | val_loss: {best_val_loss:.4f} | "
          f"val_auc: {max(history['val_auc']):.4f} | "
          f"总耗时: {sum(history['epoch_time']):.0f}s")
    return history


# ============================================================
# Part E: LightGBM Baseline
# ============================================================

def train_lightgbm_baseline(features_df, returns_df, train_days, val_days) -> Tuple[lgb.Booster, Dict]:
    """LightGBM 基线训练 (多分类, CPU)"""
    feat_cols = [f"factor_{i}" for i in range(N_FEATURES)]

    train_mask = features_df["day"].isin(train_days)
    val_mask = features_df["day"].isin(val_days)

    X_tr = features_df.loc[train_mask, feat_cols].values
    y_tr = returns_df.loc[train_mask, "target_bin"].values
    X_v = features_df.loc[val_mask, feat_cols].values
    y_v = returns_df.loc[val_mask, "target_bin"].values

    # 去除 NaN
    valid_tr = ~np.isnan(y_tr)
    valid_v = ~np.isnan(y_v)

    print(f"\n🌲 LightGBM 训练 | 训练 {valid_tr.sum():,} | 验证 {valid_v.sum():,}")

    train_data = lgb.Dataset(X_tr[valid_tr], label=y_tr[valid_tr])
    val_data = lgb.Dataset(X_v[valid_v], label=y_v[valid_v], reference=train_data)

    t0 = time.time()
    model = lgb.train(
        LGB_PARAMS, train_data,
        valid_sets=[val_data], valid_names=["val"],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=100)],
    )
    train_time = time.time() - t0

    # 评估
    probs = model.predict(X_v[valid_v])
    dir_prob = probs[:, 3:].sum(axis=1)
    y_dir = (returns_df.loc[val_mask, "actual_return"].values[valid_v] > 0).astype(int)
    auc = roc_auc_score(y_dir, dir_prob)
    acc = accuracy_score(y_dir, dir_prob > 0.5)

    metrics = {"auc": auc, "accuracy": acc, "train_time": train_time}
    print(f"✅ LightGBM val_auc: {auc:.4f} | val_acc: {acc:.3f} | {train_time:.0f}s")

    return model, metrics


# ============================================================
# Part F: 超参搜索实验
# ============================================================

def run_experiment(name: str, d_model: int, n_layers: int, n_heads: int,
                   lr: float, batch_size: int, n_epochs: int,
                   dataset, feature_df, returns_df,
                   train_days, val_days, seq_len=SEQ_LEN) -> Dict:
    """运行一组实验: Transformer + LightGBM 对比"""

    print(f"\n{'='*60}")
    print(f"🔬 实验: {name}")
    print(f"   Transformer: d={d_model} L={n_layers} H={n_heads} lr={lr} bs={batch_size}")
    print(f"{'='*60}")

    # 为每个实验重建 DataLoader (确保 batch_size 真正生效)
    n_train = int(len(dataset) * 0.8)
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True)

    # Transformer
    model = StockTransformer(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                             seq_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")

    tf_history = train_transformer(model, train_loader, val_loader,
                                   n_epochs=n_epochs, lr=lr, device=DEVICE,
                                   checkpoint_name=f"transformer_{name}")

    # LightGBM baseline
    lgb_model, lgb_metrics = train_lightgbm_baseline(feature_df, returns_df, train_days, val_days)

    # 记录
    result = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "lr": lr, "batch_size": batch_size, "n_epochs": n_epochs,
        "n_params": n_params,
        "tf_best_val_auc": float(max(tf_history["val_auc"])),
        "tf_best_val_acc": float(max(tf_history["val_acc"])),
        "lgb_val_auc": float(lgb_metrics["auc"]),
        "lgb_val_acc": float(lgb_metrics["accuracy"]),
        "lgb_train_time": float(lgb_metrics["train_time"]),
        "total_epoch_time": float(sum(tf_history["epoch_time"])),
        "device": str(DEVICE),
    }

    # 保存模型
    torch.save(model.state_dict(), MODEL_DIR / f"transformer_{name}.pt")
    lgb_model.save_model(str(MODEL_DIR / f"lightgbm_{name}.txt"))

    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   GPU 显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB used / "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB total")

    # 追加实验记录
    _log_experiment(result)

    # Telegram 通知
    if _TELEGRAM_AVAILABLE:
        try:
            elapsed = f"{result['total_epoch_time']/3600:.1f}h" if result['total_epoch_time'] > 3600 else f"{result['total_epoch_time']:.0f}s"
            send_training_done(
                experiment_name=name,
                auc=result["tf_best_val_auc"],
                acc=result["tf_best_val_acc"],
                epoch=n_epochs,
                elapsed=elapsed,
                gpu_info=f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU",
            )
        except Exception:
            pass  # 通知失败不影响训练

    return result


def _log_experiment(result: Dict):
    """追加到 experiments.csv"""
    df_new = pd.DataFrame([result])
    if EXPERIMENT_LOG.exists():
        df_old = pd.read_csv(EXPERIMENT_LOG)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(EXPERIMENT_LOG, index=False)


# ============================================================
# Part G: 主程序
# ============================================================

def main():
    print(f"\n{'='*70}")
    print(f"🚀 6小时 GPU 训练启动")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB)")
    else:
        print(f"   ⚠️ 未检测到 GPU, 使用 CPU 训练")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    t_start = time.time()

    # ---- 1. 生成合成数据 ----
    generator = SyntheticMarketGenerator(n_days=N_DAYS, n_stocks=N_STOCKS, n_features=N_FEATURES)
    features_df, returns_df, regimes, noise_levels = generator.generate()

    # ---- 2. 构建 Dataset ----
    print("📦 构建 PyTorch Dataset...")
    dataset = StockDataset(features_df, returns_df)
    total_samples = len(dataset)

    # 训练/验证日期范围 (LightGBM用)
    n_train_days = int(N_DAYS * 0.8)
    train_days = range(SEQ_LEN, n_train_days)
    val_days = range(n_train_days, N_DAYS - 1)

    # ---- 3. 定义实验列表 ----
    experiment_configs = [
        # (name, d_model, n_layers, n_heads, lr, batch_size, n_epochs)
        ("stockgpt_baseline", 128, 4, 4, 1e-4, 1536, 500),
        ("deep_l8",           128, 8, 8, 5e-5, 768,  500),
        ("wide_d384",         384, 4, 8, 1e-4, 768,  500),
        ("xl_d384_l12",       384, 12, 12, 3e-5, 384, 500),
        ("fast_bigbatch",     128, 4, 4, 5e-4, 2048, 500),
    ]

    # ---- 4. 运行实验 (跳过已完成的) ----
    experiments = []
    for config in experiment_configs:
        name = config[0]
        tf_model_path = MODEL_DIR / f"transformer_{name}.pt"
        lgb_model_path = MODEL_DIR / f"lightgbm_{name}.txt"

        # 检查是否已完成
        if tf_model_path.exists() and lgb_model_path.exists():
            print(f"\n⏭ {name}: 模型已存在, 跳过")
            # 从已有记录中恢复实验数据
            if EXPERIMENT_LOG.exists():
                df_old = pd.read_csv(EXPERIMENT_LOG)
                matched = df_old[df_old["experiment"] == name]
                if len(matched) > 0:
                    experiments.append(matched.iloc[-1].to_dict())
            continue

        # 检查是否有未完成的断点
        ckpt_path = CHECKPOINT_DIR / f"transformer_{name}.pt"
        if ckpt_path.exists():
            print(f"   📂 发现未完成断点, 将从断点继续")

        try:
            results = run_experiment(
                name, config[1], config[2], config[3],  # d_model, n_layers, n_heads
                config[4], config[5], config[6],         # lr, batch_size, n_epochs
                dataset=dataset,
                feature_df=features_df, returns_df=returns_df,
                train_days=train_days, val_days=val_days
            )
            experiments.append(results)
        except KeyboardInterrupt:
            print(f"\n⏸ 用户中断, 当前实验断点已保存")
            print(f"   已完成实验: {len(experiments)}/{len(experiment_configs)}")
            print(f"   💡 下次运行会自动从断点继续")
            break

    # ---- 5. 总结报告 ----
    total_time = time.time() - t_start
    hours = total_time / 3600

    # 合并本次新跑的 + 从实验记录加载的已有结果
    if EXPERIMENT_LOG.exists():
        df_log = pd.read_csv(EXPERIMENT_LOG)
        # 用新跑的结果覆盖同名旧记录
        if experiments:
            df_new = pd.DataFrame(experiments)
            new_names = set(df_new["experiment"].tolist())
            df_log = df_log[~df_log["experiment"].isin(new_names)]
            df_log = pd.concat([df_log, df_new], ignore_index=True)
        experiments_all = df_log.to_dict("records")
    else:
        experiments_all = experiments

    n_total = len(experiment_configs)
    n_done = len(experiments_all)

    print(f"\n{'='*70}")
    if n_done >= n_total:
        print(f"🏁 全部实验完成!")
    else:
        print(f"⏸ 训练中断/部分完成")
    print(f"   总耗时: {hours:.1f} 小时")
    print(f"   完成实验: {n_done}/{n_total}")
    print(f"   实验记录: {EXPERIMENT_LOG}")
    print(f"   模型保存: {MODEL_DIR}")
    if CHECKPOINT_DIR.exists():
        remaining = list(CHECKPOINT_DIR.glob("*.pt"))
        if remaining:
            print(f"   未完成断点: {len(remaining)} 个")
            for ckpt in remaining:
                print(f"      ⏳ {ckpt.stem}")
    print(f"{'='*70}")

    # 打印排名
    if experiments_all:
        df = pd.DataFrame(experiments_all)
        if "tf_best_val_auc" in df.columns:
            print("\n📊 实验结果排名 (按 Transformer val_auc):")
            df_sorted = df.sort_values("tf_best_val_auc", ascending=False)
            for i, (_, row) in enumerate(df_sorted.iterrows()):
                print(f"   {i+1}. {row['experiment']:20s} | TF_AUC {row['tf_best_val_auc']:.4f} "
                      f"| LGB_AUC {row['lgb_val_auc']:.4f} | Params {row['n_params']:,}")

        if "lgb_val_auc" in df.columns:
            print(f"\n📊 实验结果排名 (按 LightGBM val_auc):")
            df_sorted = df.sort_values("lgb_val_auc", ascending=False)
            for i, (_, row) in enumerate(df_sorted.iterrows()):
                print(f"   {i+1}. {row['experiment']:20s} | LGB_AUC {row['lgb_val_auc']:.4f} "
                      f"| TF_AUC {row['tf_best_val_auc']:.4f} | {row['lgb_train_time']:.0f}s")

    # Telegram 汇总通知
    if _TELEGRAM_AVAILABLE and experiments_all:
        try:
            df = pd.DataFrame(experiments_all)
            if "tf_best_val_auc" in df.columns:
                best_row = df.loc[df["tf_best_val_auc"].idxmax()]
                top = f"{best_row['experiment']} (AUC={best_row['tf_best_val_auc']:.4f})"
            else:
                top = ""
            send_all_experiments_done(
                n_done=n_done,
                n_total=n_total,
                elapsed=f"{hours:.1f}h",
                top_result=top,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
