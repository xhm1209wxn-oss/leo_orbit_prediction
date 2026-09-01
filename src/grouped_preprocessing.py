"""Grouped preprocessing utilities for collaborative residual learning.

The grouped training simulation computes normalization and validation summaries
for satellite groups in one process. Group partitions are modeling constructs,
not separate data silos.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroupStatistics:
    """Summary statistics for one satellite group."""
    group_id: int
    n_samples: int
    X_sum: np.ndarray       # 特征和（用于计算均值）
    X_sq_sum: np.ndarray    # 特征平方和（用于计算方差）
    y_sum: np.ndarray       # 标签和
    y_sq_sum: np.ndarray    # 标签平方和

    def to_dict(self) -> Dict:
        """Convert the group summary to a serializable dictionary."""
        return {
            'group_id': self.group_id,
            'n_samples': self.n_samples,
            'X_sum': self.X_sum.tolist(),
            'X_sq_sum': self.X_sq_sum.tolist(),
            'y_sum': self.y_sum.tolist(),
            'y_sq_sum': self.y_sq_sum.tolist()
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GroupStatistics':
        """Rebuild a group summary from a dictionary."""
        return cls(
            group_id=d['group_id'],
            n_samples=d['n_samples'],
            X_sum=np.array(d['X_sum']),
            X_sq_sum=np.array(d['X_sq_sum']),
            y_sum=np.array(d['y_sum']),
            y_sq_sum=np.array(d['y_sq_sum'])
        )


def compute_group_statistics(
    X: np.ndarray, y: np.ndarray, group_id: int
) -> GroupStatistics:
    """Compute the sufficient normalization statistics for one satellite group."""
    n = len(X)
    return GroupStatistics(
        group_id=group_id,
        n_samples=n,
        X_sum=X.sum(axis=0).astype(np.float64),
        X_sq_sum=(X ** 2).sum(axis=0).astype(np.float64),
        y_sum=y.sum(axis=0).astype(np.float64),
        y_sq_sum=(y ** 2).sum(axis=0).astype(np.float64)
    )


def aggregate_group_statistics(
    group_stats: List[GroupStatistics],
    eps: float = 1e-8
) -> Dict[str, np.ndarray]:
    """Aggregate group summaries to obtain shared normalization parameters."""
    if not group_stats:
        raise ValueError("No group statistics provided")

    total_n = sum(s.n_samples for s in group_stats)
    if total_n == 0:
        raise ValueError("Total sample count is zero")

    X_sum_total = sum(s.X_sum for s in group_stats)
    X_sq_sum_total = sum(s.X_sq_sum for s in group_stats)

    X_mean = X_sum_total / total_n
    X_var = (X_sq_sum_total / total_n) - (X_mean ** 2)
    X_std = np.sqrt(np.maximum(X_var, 0)) + eps  # 确保非负

    y_sum_total = sum(s.y_sum for s in group_stats)
    y_sq_sum_total = sum(s.y_sq_sum for s in group_stats)

    y_mean = y_sum_total / total_n
    y_var = (y_sq_sum_total / total_n) - (y_mean ** 2)
    y_std = np.sqrt(np.maximum(y_var, 0)) + eps

    logger.info(
        "Group statistics aggregated from %d satellite groups, total samples: %d",
        len(group_stats), total_n,
    )

    return {
        'X_mean': X_mean.astype(np.float32),
        'X_std': X_std.astype(np.float32),
        'y_mean': y_mean.astype(np.float32),
        'y_std': y_std.astype(np.float32),
        'total_samples': total_n,
        'n_groups': len(group_stats)
    }


@dataclass
class GroupValidationMetrics:
    """Validation summary for one satellite group."""
    group_id: int
    n_samples: int
    residual_sq_sum: np.ndarray   # (y_pred - y_true)² 的和
    residual_abs_sum: np.ndarray  # |y_pred - y_true| 的和
    true_sq_sum: np.ndarray       # y_true² 的和（用于 Baseline RMS）
    true_abs_sum: np.ndarray      # |y_true| 的和（用于 Baseline MAE）
    loss_sum: float               # 损失和


def compute_group_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_id: int
) -> GroupValidationMetrics:
    """Compute validation metrics for one satellite group."""
    n = len(y_true)
    residual = y_pred - y_true

    return GroupValidationMetrics(
        group_id=group_id,
        n_samples=n,
        residual_sq_sum=(residual ** 2).sum(axis=0).astype(np.float64),
        residual_abs_sum=np.abs(residual).sum(axis=0).astype(np.float64),
        true_sq_sum=(y_true ** 2).sum(axis=0).astype(np.float64),
        true_abs_sum=np.abs(y_true).sum(axis=0).astype(np.float64),
        loss_sum=float((residual ** 2).mean())
    )


def aggregate_group_validation_metrics(
    group_metrics: List[GroupValidationMetrics]
) -> Dict[str, float]:
    """Aggregate validation metrics over all satellite groups."""
    if not group_metrics:
        raise ValueError("No group metrics provided")

    total_n = sum(m.n_samples for m in group_metrics)
    if total_n == 0:
        raise ValueError("Total sample count is zero")

    residual_sq_sum_total = sum(m.residual_sq_sum for m in group_metrics)
    residual_abs_sum_total = sum(m.residual_abs_sum for m in group_metrics)

    true_sq_sum_total = sum(m.true_sq_sum for m in group_metrics)
    true_abs_sum_total = sum(m.true_abs_sum for m in group_metrics)

    loss_sum_total = sum(m.loss_sum * m.n_samples for m in group_metrics)

    residual_rms = np.sqrt(residual_sq_sum_total / total_n)
    residual_mae = residual_abs_sum_total / total_n

    baseline_rms = np.sqrt(true_sq_sum_total / total_n)
    baseline_mae = true_abs_sum_total / total_n

    pm_per_dir = (1.0 - residual_rms / (baseline_rms + 1e-10)) * 100.0
    pm_mae_per_dir = (1.0 - residual_mae / (baseline_mae + 1e-10)) * 100.0

    return {
        'baseline_rms_r': float(baseline_rms[0]),
        'baseline_rms_t': float(baseline_rms[1]),
        'baseline_rms_n': float(baseline_rms[2]),
        'baseline_rms_mean': float(baseline_rms.mean()),
        'residual_rms_r': float(residual_rms[0]),
        'residual_rms_t': float(residual_rms[1]),
        'residual_rms_n': float(residual_rms[2]),
        'residual_rms_mean': float(residual_rms.mean()),
        'pm_r': float(pm_per_dir[0]),
        'pm_t': float(pm_per_dir[1]),
        'pm_n': float(pm_per_dir[2]),
        'pm_mean': float(pm_per_dir.mean()),
        'baseline_mae_r': float(baseline_mae[0]),
        'baseline_mae_t': float(baseline_mae[1]),
        'baseline_mae_n': float(baseline_mae[2]),
        'residual_mae_r': float(residual_mae[0]),
        'residual_mae_t': float(residual_mae[1]),
        'residual_mae_n': float(residual_mae[2]),
        'pm_mae_r': float(pm_mae_per_dir[0]),
        'pm_mae_t': float(pm_mae_per_dir[1]),
        'pm_mae_n': float(pm_mae_per_dir[2]),
        'pm_mae_mean': float(pm_mae_per_dir.mean()),
        'loss': loss_sum_total / total_n,
        'total_samples': total_n,
        'n_groups': len(group_metrics)
    }
