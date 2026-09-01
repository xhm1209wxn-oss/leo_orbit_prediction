"""
用于轨道误差预测的评估指标
计算 TLE 误差校正的性能指标（PM）
"""

import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)
_EPS = 1e-12


def evaluate_pm_global(model: torch.nn.Module,
                       val_loader,
                       device: torch.device,
                       norm_params: Dict = None) -> Dict:
    """
    在验证集上评估性能指标（PM），同时提供基于 RMS 与 MAE 的两套指标

    PM_rms = (1 - Residual_RMS / Baseline_RMS) × 100%
    PM_mae = (1 - Residual_MAE / Baseline_MAE) × 100%

    参数：
        model: 已训练模型
        val_loader: 验证集数据加载器
        device: 设备
        norm_params: 标准化参数（可选，用于反标准化）

    返回：
        评估指标字典
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            preds = model(batch_x)

            all_true.append(batch_y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    true_err = np.vstack(all_true)  # (N, 3) [R,T,N]，已标准化
    pred_err = np.vstack(all_pred)  # (N, 3)，已标准化

    if norm_params is not None:
        y_mean = norm_params['y_mean']
        y_std = norm_params['y_std']

        true_err = true_err * y_std + y_mean  # 返回 km
        pred_err = pred_err * y_std + y_mean

        logger.info("✓ Denormalized predictions back to original scale (km)")

    residual = true_err - pred_err

    baseline_rms = np.sqrt(np.mean(true_err ** 2, axis=0))
    resid_rms = np.sqrt(np.mean(residual ** 2, axis=0))

    baseline_mae = np.mean(np.abs(true_err), axis=0)
    resid_mae = np.mean(np.abs(residual), axis=0)

    pm_per_dir = (1.0 - resid_rms / baseline_rms) * 100.0
    pm_mean = pm_per_dir.mean()
    pm_mae_per_dir = np.where(
        baseline_mae > 0,
        (1.0 - resid_mae / baseline_mae) * 100.0,
        0.0
    )
    pm_mae_mean = pm_mae_per_dir.mean()

    baseline_norm = np.linalg.norm(true_err, axis=1)
    resid_norm = np.linalg.norm(residual, axis=1)
    baseline_rms_3d = np.sqrt(np.mean(baseline_norm ** 2))
    resid_rms_3d = np.sqrt(np.mean(resid_norm ** 2))
    baseline_mae_3d = baseline_norm.mean()
    resid_mae_3d = resid_norm.mean()
    pm_rms_3d = (1.0 - resid_rms_3d / baseline_rms_3d) * 100.0 if baseline_rms_3d > 0 else 0.0
    pm_mae_3d = (1.0 - resid_mae_3d / baseline_mae_3d) * 100.0 if baseline_mae_3d > 0 else 0.0

    baseline_sq_sum = np.sum(true_err ** 2, axis=0)
    resid_sq_sum = np.sum(residual ** 2, axis=0)
    eag_per_dir = 10.0 * np.log10((baseline_sq_sum + _EPS) / (resid_sq_sum + _EPS))
    eag_mean = eag_per_dir.mean()

    csr_per_dir = (np.mean(np.abs(residual) < np.abs(true_err), axis=0) * 100.0).astype(np.float64)
    csr_mean = csr_per_dir.mean()

    baseline_sq_sum_3d = np.sum(baseline_norm ** 2)
    resid_sq_sum_3d = np.sum(resid_norm ** 2)
    eag_3d = 10.0 * np.log10((baseline_sq_sum_3d + _EPS) / (resid_sq_sum_3d + _EPS))
    csr_3d = float(np.mean(resid_norm < baseline_norm) * 100.0)

    metrics = {
        'baseline_rms': baseline_rms,
        'resid_rms': resid_rms,
        'pm_per_dir': pm_per_dir,
        'pm_mean': pm_mean,
        'baseline_mae': baseline_mae,
        'resid_mae': resid_mae,
        'pm_mae_per_dir': pm_mae_per_dir,
        'pm_mae_mean': pm_mae_mean,
        'baseline_norm_mean': baseline_norm.mean(),
        'resid_norm_mean': resid_norm.mean(),
        'baseline_rms_3d': baseline_rms_3d,
        'resid_rms_3d': resid_rms_3d,
        'baseline_mae_3d': baseline_mae_3d,
        'resid_mae_3d': resid_mae_3d,
        'pm_rms_3d': pm_rms_3d,
        'pm_mae_3d': pm_mae_3d,
        'eag_per_dir': eag_per_dir,
        'eag_mean': eag_mean,
        'eag_3d': eag_3d,
        'csr_per_dir': csr_per_dir,
        'csr_mean': csr_mean,
        'csr_3d': csr_3d,
        'true_err': true_err,
        'residual': residual
    }

    return metrics


def print_pm_results(metrics: Dict):
    """
    以表格形式打印 PM 结果

    参数：
        metrics: evaluate_pm_global 输出的指标字典
    """
    baseline_rms = metrics['baseline_rms']
    resid_rms = metrics['resid_rms']
    pm_per_dir = metrics['pm_per_dir']
    pm_mean = metrics['pm_mean']

    baseline_mae = metrics['baseline_mae']
    resid_mae = metrics['resid_mae']
    pm_mae_per_dir = metrics['pm_mae_per_dir']
    pm_mae_mean = metrics['pm_mae_mean']
    baseline_rms_3d = metrics['baseline_rms_3d']
    resid_rms_3d = metrics['resid_rms_3d']
    pm_rms_3d = metrics['pm_rms_3d']
    baseline_mae_3d = metrics['baseline_mae_3d']
    resid_mae_3d = metrics['resid_mae_3d']
    pm_mae_3d = metrics['pm_mae_3d']
    eag_per_dir = metrics.get('eag_per_dir')
    eag_mean = metrics.get('eag_mean')
    eag_3d = metrics.get('eag_3d')
    csr_per_dir = metrics.get('csr_per_dir')
    csr_mean = metrics.get('csr_mean')
    csr_3d = metrics.get('csr_3d')

    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE METRICS (Real TLE Dataset, RTN frame)")
    logger.info("=" * 80)
    logger.info(f"{'Direction':<20} {'Baseline RMS (km)':<22} {'Residual RMS (km)':<22} {'PM_RMS (%)':<12}")
    logger.info(f"{'Radial (R)':<20} {baseline_rms[0]:>18.4f} {resid_rms[0]:>22.4f} {pm_per_dir[0]:>10.4f}")
    logger.info(f"{'Along-track (T)':<20} {baseline_rms[1]:>18.4f} {resid_rms[1]:>22.4f} {pm_per_dir[1]:>10.4f}")
    logger.info(f"{'Normal (N)':<20} {baseline_rms[2]:>18.4f} {resid_rms[2]:>22.4f} {pm_per_dir[2]:>10.4f}")
    logger.info(f"{'Mean':<20} {baseline_rms.mean():>18.4f} {resid_rms.mean():>22.4f} {pm_mean:>10.4f}")
    logger.info(f"{'3D norm':<20} {baseline_rms_3d:>18.4f} {resid_rms_3d:>22.4f} {pm_rms_3d:>10.4f}")
    logger.info("=" * 80)

    logger.info(f"{'Direction':<20} {'Baseline MAE (km)':<22} {'Residual MAE (km)':<22} {'PM_MAE (%)':<12}")
    logger.info(f"{'Radial (R)':<20} {baseline_mae[0]:>18.4f} {resid_mae[0]:>22.4f} {pm_mae_per_dir[0]:>10.4f}")
    logger.info(f"{'Along-track (T)':<20} {baseline_mae[1]:>18.4f} {resid_mae[1]:>22.4f} {pm_mae_per_dir[1]:>10.4f}")
    logger.info(f"{'Normal (N)':<20} {baseline_mae[2]:>18.4f} {resid_mae[2]:>22.4f} {pm_mae_per_dir[2]:>10.4f}")
    logger.info(f"{'Mean':<20} {baseline_mae.mean():>18.4f} {resid_mae.mean():>22.4f} {pm_mae_mean:>10.4f}")
    logger.info(f"{'3D norm':<20} {baseline_mae_3d:>18.4f} {resid_mae_3d:>22.4f} {pm_mae_3d:>10.4f}")
    logger.info("=" * 80)

    if eag_per_dir is not None and csr_per_dir is not None:
        logger.info(f"{'Direction':<20} {'EAG (dB)':<22} {'CSR (%)':<12}")
        logger.info(f"{'Radial (R)':<20} {float(eag_per_dir[0]):>18.4f} {float(csr_per_dir[0]):>22.4f}")
        logger.info(f"{'Along-track (T)':<20} {float(eag_per_dir[1]):>18.4f} {float(csr_per_dir[1]):>22.4f}")
        logger.info(f"{'Normal (N)':<20} {float(eag_per_dir[2]):>18.4f} {float(csr_per_dir[2]):>22.4f}")
        logger.info(f"{'Mean':<20} {float(eag_mean):>18.4f} {float(csr_mean):>22.4f}")
        logger.info(f"{'3D norm':<20} {float(eag_3d):>18.4f} {float(csr_3d):>22.4f}")
        logger.info("=" * 80)
