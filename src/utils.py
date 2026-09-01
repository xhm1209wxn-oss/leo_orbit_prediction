"""
项目的实用函数
包含日志、设备设置、可复现性等工具
"""

import torch
import numpy as np
import random
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO",
                  console: bool = True):
    """
    配置日志

    参数：
        log_level: 日志等级（DEBUG/INFO/WARNING/ERROR）
        console: 是否输出到控制台
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logging.info(f"Logging initialized (level={log_level}, console={console})")


def setup_reproducibility(seed: int = 42, deterministic: bool = True):
    """
    设置实验的可复现性

    参数：
        seed: 随机种子
        deterministic: 是否使用确定性算法（速度略慢但可复现）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logging.info(f"Random seeds set to {seed} (deterministic={deterministic})")


def setup_device(device: str = "auto") -> torch.device:
    """
    设置计算设备

    参数：
        device: 设备指定（auto/cpu/cuda:0/cuda:1/...）

    返回：
        torch.device 对象
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)

    if device_obj.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device_obj)
        gpu_memory = torch.cuda.get_device_properties(device_obj).total_memory / 1e9
        logging.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logging.info("Using CPU")

    return device_obj


def ensure_directory(path: str) -> Path:
    """
    确保目录存在，不存在则创建

    参数：
        path: 目录路径

    返回：
        Path 对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
class EarlyStopping:
    """提前停止处理器"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        参数：
            patience: 等待改进的轮次数
            min_delta: 判定改进的最小变化（用于早停判断）
            mode: 'min' 或 'max'（指示更小或更大更优）
        """
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_score = None  # 用于早停判断（需满足 min_delta）
        self.best_epoch = 0  # 用于早停判断
        self.num_bad_epochs = 0

        self.best_score_absolute = None
        self.best_epoch_absolute = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        检查是否应提前停止

        参数：
            score: 当前指标（损失或评价指标）
            epoch: 当前轮次索引（从0开始）

        返回：
            True 表示应停止，否则 False
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_score_absolute = score
            self.best_epoch_absolute = epoch
        else:
            if self.mode == 'min':
                better_absolute = score < self.best_score_absolute
            else:
                better_absolute = score > self.best_score_absolute
            if better_absolute:
                self.best_score_absolute = score
                self.best_epoch_absolute = epoch

            if self.mode == 'min':
                improved = score < (self.best_score - self.min_delta)
            else:
                improved = score > (self.best_score + self.min_delta)

            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                logging.info(
                    f"Early stopping triggered at round {epoch + 1} "
                    f"(no improvement for {self.num_bad_epochs} consecutive rounds, "
                    f"patience={self.patience}, min_delta={self.min_delta})"
                )
                return True

        return False
