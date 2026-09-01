"""
用于结果展示的可视化函数
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def _compact_plot_series(values, width: int = 3):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or width <= 1:
        return values
    kernel = np.ones(int(width), dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode='valid')


def plot_training_history(history: dict, save_path: str = None, dpi: int = 300, best_epoch: int = None):
    """
    绘制训练过程曲线

    参数：
        history: 训练记录字典
        save_path: 图像保存路径
        dpi: 图像 DPI
        best_epoch: 最佳轮次索引（可选，用于标记早停位置）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    rounds = np.arange(1, len(history['train_loss']) + 1)
    ax.plot(rounds, history['train_loss'], 'b-o', label='Training Loss',
            linewidth=2, markersize=4)

    if history.get('val_loss') and len(history['val_loss']) > 0:
        val_rounds = np.asarray(history.get('val_rounds', []), dtype=int)
        if val_rounds.size == 0:
            val_rounds = rounds[:len(history['val_loss'])]
        ax.plot(val_rounds, history['val_loss'], 'r-s', label='Validation Loss',
                linewidth=2, markersize=4)

        if best_epoch is not None:
            best_round = best_epoch + 1
            matching = np.where(val_rounds == best_round)[0]
            if matching.size > 0:
                best_idx = int(matching[0])
                best_val_loss = history['val_loss'][best_idx]
                ax.plot(best_round, best_val_loss, 'g*', markersize=15,
                       label=f'Best (Round {best_round})', zorder=5)
                ax.axvline(x=best_round, color='green', linestyle='--',
                          alpha=0.5, linewidth=1.5)
    else:
        logger.info("val_loss is empty, plotting train_loss only")

    ax.set_xlabel('Collaborative Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Grouped Collaborative Training Progress', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

    plt.close()


def plot_pm_comparison(metrics: dict, save_path: str = None, dpi: int = 300):
    """
    绘制 PM 对比柱状图

    参数：
        metrics: 指标字典
        save_path: 图像保存路径
        dpi: 图像 DPI
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    pm_per_dir = metrics['pm_per_dir']
    pm_mean = metrics['pm_mean']

    dirs_short = ['R', 'T', 'N', 'Mean']
    x = np.arange(len(dirs_short))
    pm_vals = list(pm_per_dir) + [pm_mean]

    bars = ax.bar(x, pm_vals, color=['#3498db', '#2ecc71', '#9b59b6', '#e67e22'],
                  edgecolor='black', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(dirs_short, fontsize=12)
    ax.set_ylabel('PM (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Performance Metric by Direction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, pm_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1.0,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"PM comparison plot saved to {save_path}")

    plt.close()


def plot_rtn_error_histograms(true_err: np.ndarray,
                              residual: np.ndarray,
                              save_path: str = None,
                              dpi: int = 300,
                              bins: int = 50,
                              clip_percentile: float = 99.0):
    """
    绘制 RTN 误差分布直方图（基线 vs 残差）

    参数：
        true_err: 真实误差 (N,3)，单位 km
        residual: 残差误差 (N,3)，单位 km
        save_path: 图像保存路径
        dpi: 图像 DPI
        bins: 直方图分箱数量
        clip_percentile: x 轴范围使用 |误差| 的该分位数（对称截断，避免极端值撑大坐标）
    """
    dirs = ['Radial (R)', 'Along-track (T)', 'Normal (N)']
    colors = {'baseline': '#2980b9', 'residual': '#e67e22'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle('RTN Error Distribution (km)', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes):
        combined = np.concatenate([true_err[:, i], residual[:, i]])
        max_abs = np.percentile(np.abs(combined), clip_percentile)
        x_lim = max(1e-6, max_abs) * 1.05  # 留 5% 边距

        ax.hist(true_err[:, i], bins=bins, alpha=0.6, color=colors['baseline'],
                label='Baseline', edgecolor='black', linewidth=0.5)
        ax.hist(residual[:, i], bins=bins, alpha=0.6, color=colors['residual'],
                label='Residual', edgecolor='black', linewidth=0.5)
        ax.set_title(dirs[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Error (km)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(-x_lim, x_lim)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.6, linestyle='--')
        ax.text(0.98, 0.95, f'|x|<=p{clip_percentile:.0f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"RTN error histograms saved to {save_path}")

    plt.close()


def plot_dt_distribution(split_dt_seconds: Dict[str, np.ndarray],
                         save_path: str = None,
                         dpi: int = 300,
                         bins: int = 80):
    """
    绘制 dt 分布图（按 train/val/test 叠加）

    参数：
        split_dt_seconds: 各 split 的 dt 数组（单位秒）
        save_path: 图像保存路径
        dpi: 图像 DPI
        bins: 直方图分箱数量
    """
    colors = {
        'train': '#1f77b4',
        'val': '#ff7f0e',
        'test': '#2ca02c',
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    has_any = False

    for split_name in ['train', 'val', 'test']:
        values = split_dt_seconds.get(split_name)
        if values is None or len(values) == 0:
            continue

        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        if not valid.any():
            continue

        dt_hours = values[valid] / 3600.0
        has_any = True
        ax.hist(
            dt_hours,
            bins=bins,
            alpha=0.45,
            density=False,
            color=colors.get(split_name, None),
            label=f"{split_name} (n={len(dt_hours):,})",
            edgecolor='black',
            linewidth=0.4,
        )

        q50, q90, q99 = np.percentile(dt_hours, [50, 90, 99])
        logger.info(
            "dt distribution [%s]: n=%d, mean=%.2f h, median=%.2f h, p90=%.2f h, p99=%.2f h",
            split_name,
            len(dt_hours),
            float(dt_hours.mean()),
            float(q50),
            float(q90),
            float(q99),
        )

    if not has_any:
        logger.warning("No valid dt values available, skipping dt distribution plot.")
        plt.close(fig)
        return

    ax.set_xlabel('dt (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('dt Distribution by Split', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"dt distribution plot saved to {save_path}")

    plt.close(fig)
