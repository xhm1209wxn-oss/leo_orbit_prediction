"""
预处理模块
负责 SGP4 传播、轨道根数计算与误差数据集构建
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)


def parse_epoch(epoch_string: str) -> datetime:
    """
    将历元字符串解析为 datetime

    参数：
        epoch_string: 多种格式的历元字符串

    返回：
        datetime 对象
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S.%f',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(epoch_string, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse epoch string: {epoch_string}")


def propagate_tle_sgp4(tle_line1: str,
                       tle_line2: str,
                       start_time: datetime,
                       duration_hours: float = 48,
                       step_minutes: float = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 SGP4 传播一组 TLE

    参数：
        tle_line1: TLE 第一行
        tle_line2: TLE 第二行
        start_time: 传播起始时间
        duration_hours: 传播持续时间（小时）
        step_minutes: 时间步长（分钟）

    返回：
        (times, positions, velocities) 数组
    """
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)

    num_steps = int((duration_hours * 60) / step_minutes)
    times = [start_time + timedelta(minutes=i * step_minutes) for i in range(num_steps)]

    positions = []
    velocities = []

    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        error_code, r, v = satellite.sgp4(jd, fr)

        if error_code == 0:
            positions.append(r)
            velocities.append(v)
        else:
            positions.append([np.nan, np.nan, np.nan])
            velocities.append([np.nan, np.nan, np.nan])

    return np.array(times), np.array(positions), np.array(velocities)


def compute_orbital_elements(position: np.ndarray,
                             velocity: np.ndarray,
                             mu: float = 398600.4418) -> Dict[str, float]:
    """
    根据位置与速度计算经典轨道根数

    参数：
        position: 位置向量 [x, y, z]（km）
        velocity: 速度向量 [vx, vy, vz]（km/s）
        mu: 万有引力参数 (km^3/s^2)

    返回：
        轨道根数字典
    """
    r = position
    v = velocity

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)

    e_vec = ((v_mag ** 2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    energy = v_mag ** 2 / 2 - mu / r_mag

    if e != 1.0:
        a = -mu / (2 * energy)
    else:
        a = np.inf

    i = np.arccos(h[2] / h_mag)

    if n_mag != 0:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0

    if n_mag != 0 and e != 0:
        omega = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0

    if e != 0:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0

    return {
        'a': a,
        'e': e,
        'i': np.degrees(i),
        'omega': np.degrees(omega),
        'Omega': np.degrees(Omega),
        'nu': np.degrees(nu)
    }


def compute_argument_of_latitude_sin_cos(position: np.ndarray,
                                         velocity: np.ndarray) -> Tuple[float, float]:
    """
    根据状态向量计算 argument of latitude u 的 sin/cos。

    这里优先用升交线与当前位置的几何关系直接计算，避免近圆轨道下
    先分别算 omega 和 nu 再相加带来的数值不稳定。
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)

    r_mag = np.linalg.norm(r)
    if r_mag <= 0:
        return 0.0, 1.0

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    if h_mag <= 0:
        return 0.0, 1.0

    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n)

    if n_mag <= 1e-12:
        u = np.arctan2(r[1], r[0])
        return float(np.sin(u)), float(np.cos(u))

    cos_u = np.dot(n, r) / (n_mag * r_mag)
    cos_u = float(np.clip(cos_u, -1.0, 1.0))

    sin_u = np.dot(np.cross(n, r), h) / (n_mag * r_mag * h_mag)
    sin_u = float(np.clip(sin_u, -1.0, 1.0))

    norm = np.hypot(sin_u, cos_u)
    if norm <= 0:
        return 0.0, 1.0
    return sin_u / norm, cos_u / norm


def eci_to_rtn(r_ref: np.ndarray, v_ref: np.ndarray, r_other: np.ndarray) -> np.ndarray:
    """
    将位置差从 ECI 转换到 RTN 坐标系

    参数：
        r_ref: 参考位置（km）
        v_ref: 参考速度（km/s）
        r_other: 其他位置（km）

    返回：
        RTN 坐标系下的位置差 [R, T, N]（km）
    """
    r_ref = np.asarray(r_ref, dtype=np.float64)
    v_ref = np.asarray(v_ref, dtype=np.float64)

    r_hat = r_ref / np.linalg.norm(r_ref)  # 径向
    h = np.cross(r_ref, v_ref)
    h_hat = h / np.linalg.norm(h)  # 法向
    t_hat = np.cross(h_hat, r_hat)  # 切向

    R_mat = np.vstack([r_hat, t_hat, h_hat]).T

    d_r = np.asarray(r_other, dtype=np.float64) - r_ref

    return R_mat.T @ d_r


def _feature_range_signature(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        return np.empty(0, dtype=np.float64)
    return np.nanmax(features, axis=0) - np.nanmin(features, axis=0)


def extract_ballistic_coefficient(tle_line1: str) -> float:
    """
    从 TLE 第一行提取 B* 阻力系数

    参数：
        tle_line1: TLE 第一行

    返回：
        B* 系数
    """
    try:
        field = tle_line1[53:61]
        if len(field) != 8:
            return 0.0

        mantissa_sign = -1.0 if field[0] == '-' else 1.0
        mantissa_digits = field[1:6]
        exp_sign = -1 if field[6] == '-' else 1
        exp_digits = field[7]

        if not mantissa_digits.strip() or not exp_digits.strip():
            return 0.0

        mantissa = int(mantissa_digits) / 1e5
        exponent = exp_sign * int(exp_digits)
        return mantissa_sign * mantissa * (10.0 ** exponent)
    except Exception:
        return 0.0


def propagate_satrec_to_time(sat: Satrec, t_dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 Satrec 传播到指定时间

    参数：
        sat: Satrec 对象
        t_dt: 目标时间

    返回：
        (position, velocity)，若出错则返回 (None, None)
    """
    jd, fr = jday(t_dt.year, t_dt.month, t_dt.day,
                  t_dt.hour, t_dt.minute, t_dt.second)
    err, r, v = sat.sgp4(jd, fr)

    if err != 0:
        return None, None

    return np.array(r, dtype=np.float64), np.array(v, dtype=np.float64)


def cross_satellite_iqr_filter(error_datasets: Dict[str, Dict],
                               k: float = 3.0,
                               metric: str = 'T_median') -> Tuple[Dict[str, Dict], List[str]]:
    """
    跨卫星级别的 IQR 异常过滤

    基于每颗卫星的汇总误差指标（如 T 方向的中位数误差），
    在全局卫星维度上进行 IQR 过滤，识别并剔除整颗卫星数据异常的情况。

    参数：
        error_datasets: {sat_name: {'X': features, 'y': labels}} 的字典
        k: IQR 倍数，默认 3.0（外围栅栏）
        metric: 用于计算卫星级别异常的指标
            - 'T_median': T 方向误差的中位数（默认）
            - 'T_mean': T 方向误差的均值
            - 'total_rms': RTN 总 RMS 误差

    返回：
        (filtered_datasets, removed_satellites)
        - filtered_datasets: 过滤后的数据集字典
        - removed_satellites: 被剔除的卫星名称列表
    """
    if len(error_datasets) < 4:
        logger.warning("Too few satellites for cross-satellite IQR filter, skipping.")
        return error_datasets, []

    sat_metrics = {}
    for sat_name in sorted(error_datasets.keys()):
        data = error_datasets[sat_name]
        y = data.get('y')
        if y is None or len(y) == 0:
            continue

        if metric == 'T_median':
            val = np.median(np.abs(y[:, 1]))
        elif metric == 'T_mean':
            val = np.abs(np.mean(y[:, 1]))
        elif metric == 'total_rms':
            val = np.sqrt(np.mean(y ** 2))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        sat_metrics[sat_name] = val

    if len(sat_metrics) < 4:
        logger.warning("Too few satellites with valid metrics for IQR filter, skipping.")
        return error_datasets, []

    values = np.array(list(sat_metrics.values()))
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1

    if IQR < 1e-10:
        logger.info("Cross-satellite IQR is near zero, skipping filter.")
        return error_datasets, []

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    removed_satellites = []
    for sat_name in sorted(sat_metrics.keys()):
        val = sat_metrics[sat_name]
        if val < lower or val > upper:
            removed_satellites.append(sat_name)
    removed_satellites = sorted(removed_satellites)

    filtered_datasets = {
        sat_name: error_datasets[sat_name]
        for sat_name in sorted(error_datasets.keys())
        if sat_name not in removed_satellites
    }

    if removed_satellites:
        logger.info(f"Cross-satellite IQR filter (k={k}, metric={metric}):")
        logger.info(f"  Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
        logger.info(f"  Valid range: [{lower:.4f}, {upper:.4f}]")
        logger.info(f"  Removed {len(removed_satellites)}/{len(sat_metrics)} satellites:")
        for sat_name in removed_satellites[:10]:  # 只显示前 10 颗
            logger.info(f"    - {sat_name}: {sat_metrics[sat_name]:.4f} km")
        if len(removed_satellites) > 10:
            logger.info(f"    ... and {len(removed_satellites) - 10} more")

        removed_samples = sum(
            len(error_datasets[sat]['y'])
            for sat in removed_satellites
            if 'y' in error_datasets[sat]
        )
        total_samples = sum(
            len(data['y'])
            for data in error_datasets.values()
            if 'y' in data and len(data['y']) > 0
        )
        logger.info(f"  Removed {removed_samples}/{total_samples} samples "
                    f"({100 * removed_samples / total_samples:.1f}%)")
    else:
        logger.info(f"Cross-satellite IQR filter (k={k}): no satellites removed.")

    return filtered_datasets, removed_satellites


def iqr_outlier_filter(X: np.ndarray, y: np.ndarray, k: float = 3.0,
                       min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于箱线图IQR的异常值过滤（按卫星独立计算）

    参数：
        X: 特征数组 (N, D)
        y: 标签数组 (N, 3)，RTN误差
        k: IQR倍数，默认3.0（外围栅栏）
        min_samples: 最小样本数，样本不足时跳过过滤

    返回：
        过滤后的 (X, y, mask)
    """
    mask = np.ones(len(y), dtype=bool)

    if len(y) < min_samples:
        return X, y, mask

    for dim in range(3):
        vals = y[:, dim]
        Q1 = np.percentile(vals, 25)
        Q3 = np.percentile(vals, 75)
        IQR = Q3 - Q1

        if IQR < 1e-10:
            continue

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        outliers = (vals < lower) | (vals > upper)
        mask[outliers] = False

    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.debug(f"IQR filter (k={k}): removed {n_removed}/{len(y)} samples "
                     f"({100*n_removed/len(y):.1f}%)")

    return X[mask], y[mask], mask


def build_tle_error_dataset(tle_list: List[Dict],
                            max_dt: float = 7 * 24 * 3600,
                            seed: int = 42,
                            orbit_samples_n: int = 8,
                            delta_n_threshold: float = 0.003,
                            iqr_k: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于 TLE 交叉验证构建误差数据集

    - 对每个有效配对 (i, j)，以 t_j 为中心取一个轨道周期时间窗并均匀采样
    - 采样时刻：t_{j,k} = t_j - T/2 + k*T/n, k=0..n（共 n+1 个点）
    - 每个采样点使用 TLE_j 状态作为伪真值，TLE_i 外推作为预测
    - 检查路径上所有相邻 TLE 的 mean motion 连续性

    参数：
        tle_list: TLE 记录列表
        max_dt: 最大时间差（秒）
        seed: 随机种子（向后兼容）
        orbit_samples_n: 周期窗均匀采样分段数 n（k=0..n，共 n+1 个采样点）
        delta_n_threshold: 相邻 TLE 的 mean motion 允许变化上限（rev/day），超出则跳过
        iqr_k: IQR异常值过滤倍数，默认3.0；设为0或负数则禁用IQR过滤

    返回：
        (X, y, timestamps) 特征、标签和时间戳（按时间戳升序）
    """
    tle_list = sorted(tle_list, key=lambda d: d['EPOCH'])

    deduped = []
    last_epoch_sec = None
    for rec in tle_list:
        epoch_sec = parse_epoch(rec['EPOCH']).replace(microsecond=0)
        if epoch_sec == last_epoch_sec:
            continue
        deduped.append(rec)
        last_epoch_sec = epoch_sec
    tle_list = deduped

    n = len(tle_list)

    skip_steps = [1]
    feature_dim = 19

    X_list, y_list, timestamp_list, skip_type_list = [], [], [], []

    sats = [Satrec.twoline2rv(rec['TLE_LINE1'], rec['TLE_LINE2']) for rec in tle_list]
    epochs = [parse_epoch(rec['EPOCH']) for rec in tle_list]
    bstars = [extract_ballistic_coefficient(rec['TLE_LINE1']) for rec in tle_list]
    mean_motion_rad_s = []
    for rec in tle_list:
        try:
            n_rev_day = float(rec['TLE_LINE2'].split()[7][:11])  # rev/day
        except (ValueError, IndexError):
            n_rev_day = np.nan
        if np.isfinite(n_rev_day) and n_rev_day > 0:
            mean_motion_rad_s.append(float(n_rev_day * 2.0 * np.pi / 86400.0))
        else:
            mean_motion_rad_s.append(np.nan)

    for i in range(n - 1):
        sat_i = sats[i]
        t_i = epochs[i]
        b_i = bstars[i]
        n_i_rad_s = mean_motion_rad_s[i]
        r_i, v_i = propagate_satrec_to_time(sat_i, t_i)
        if r_i is None or not np.isfinite(n_i_rad_s):
            continue

        for skip in skip_steps:
            j = i + skip

            if j >= n:
                break

            valid_path = True
            for k in range(i, j):
                line2_k = tle_list[k]['TLE_LINE2'].split()
                line2_k1 = tle_list[k + 1]['TLE_LINE2'].split()
                delta_n = abs(float(line2_k1[7][:11]) - float(line2_k[7][:11]))
                if delta_n > delta_n_threshold:
                    valid_path = False
                    break

            if not valid_path:
                continue

            sat_j = sats[j]
            t_j = epochs[j]
            line2_j = tle_list[j]['TLE_LINE2'].split()

            dt_ij = (t_j - t_i).total_seconds()

            if dt_ij <= 0 or dt_ij > max_dt:
                continue

            try:
                n_mean = float(line2_j[7][:11])  # rev/day
            except (ValueError, IndexError):
                continue

            if n_mean <= 0:
                continue

            orbit_period_seconds = 86400.0 / n_mean
            n_segments = max(1, int(orbit_samples_n))
            start_offset = -0.5 * orbit_period_seconds
            step_seconds = orbit_period_seconds / n_segments

            for k_idx in range(n_segments + 1):
                t_jk = t_j + timedelta(seconds=float(start_offset + k_idx * step_seconds))

                dt_ik = (t_jk - t_i).total_seconds()
                if dt_ik <= 0:
                    continue
                phase_ik = float(n_i_rad_s * dt_ik)
                sin_n_dt = float(np.sin(phase_ik))
                cos_n_dt = float(np.cos(phase_ik))

                r_jk, v_jk = propagate_satrec_to_time(sat_j, t_jk)
                if r_jk is None:
                    continue

                r_ik, v_ik = propagate_satrec_to_time(sat_i, t_jk)
                if r_ik is None:
                    continue

                sin_u_ik, cos_u_ik = compute_argument_of_latitude_sin_cos(r_ik, v_ik)
                h_ik = float(np.linalg.norm(r_ik) - 6378.137)

                e_rtn = eci_to_rtn(r_jk, v_jk, r_ik)

                x_parts = [
                    [dt_ik],
                    [sin_n_dt],
                    [cos_n_dt],
                    r_i, v_i,              # ECI坐标系
                    r_ik, v_ik,            # ECI坐标系
                    [sin_u_ik],
                    [cos_u_ik],
                    [h_ik],
                    [b_i],
                ]

                x_vec = np.hstack(x_parts)

                X_list.append(x_vec.astype(np.float32))
                y_list.append(e_rtn.astype(np.float32))
                timestamp_list.append(t_jk)
                skip_type_list.append(skip)

    if len(X_list) == 0:
        X = np.empty((0, feature_dim), dtype=np.float32)
        y = np.empty((0, 3), dtype=np.float32)
        timestamps = np.array([], dtype='datetime64[s]')
        return X, y, timestamps

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    timestamps = np.array(timestamp_list, dtype='datetime64[s]')
    skip_types = np.array(skip_type_list, dtype=np.int32)

    if len(skip_types) > 0:
        logger.debug(f"Sample distribution by skip type:")
        for skip in sorted(set(skip_types)):
            count = (skip_types == skip).sum()
            percentage = 100 * count / len(skip_types)

            mask = skip_types == skip
            avg_dt_days = X[mask, 0].mean() / 86400 if count > 0 else 0

            logger.debug(f"  Skip {skip} (i -> i+{skip}): {count} samples ({percentage:.1f}%), "
                         f"avg dt = {avg_dt_days:.2f} days")

    if iqr_k > 0 and len(y) > 0:
        n_before = len(y)
        X, y, mask = iqr_outlier_filter(X, y, k=iqr_k)
        n_after = len(y)

        if n_before > n_after:
            timestamps = timestamps[mask]
            logger.debug(f"  IQR filter (k={iqr_k}): {n_before} -> {n_after} samples "
                         f"(removed {n_before - n_after}, {100*(n_before-n_after)/n_before:.1f}%)")

    if len(timestamps) > 1:
        sort_idx = np.argsort(timestamps, kind='mergesort')
        if not np.array_equal(sort_idx, np.arange(len(timestamps))):
            if logger.isEnabledFor(logging.DEBUG):
                ts_seconds = timestamps.astype('datetime64[s]').astype(np.int64)
                reverse_pairs = int(np.sum(np.diff(ts_seconds) < 0))
                logger.debug(f"  Sorted timestamps: fixed {reverse_pairs} reverse transitions")
            X = X[sort_idx]
            y = y[sort_idx]
            timestamps = timestamps[sort_idx]

    return X, y, timestamps


def normalize_data(X: np.ndarray, y: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用预计算的参数对特征与标签进行标准化

    参数：
        X: 特征数组
        y: 标签数组
        params: 标准化参数

    返回：
        (X_normalized, y_normalized)
    """
    X_norm = (X - params['X_mean']) / params['X_std']
    y_norm = (y - params['y_mean']) / params['y_std']

    return X_norm.astype(np.float32), y_norm.astype(np.float32)


def denormalize_predictions(y_norm: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    将预测从标准化值还原到原始量纲

    参数：
        y_norm: 标准化后的预测
        params: 标准化参数

    返回：
        反标准化后的预测
    """
    return y_norm * params['y_std'] + params['y_mean']


def save_normalization_params(params: Dict[str, np.ndarray], save_path: str):
    """
    将标准化参数保存到文件

    参数：
        params: 标准化参数
        save_path: 保存路径
    """
    import json
    from pathlib import Path

    params_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in params.items()
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(params_serializable, f, indent=2)

    logger.info(f"Normalization parameters saved to {save_path}")


def load_normalization_params(load_path: str) -> Dict[str, np.ndarray]:
    """
    从文件加载标准化参数

    参数：
        load_path: 参数文件路径

    返回：
        标准化参数
    """
    import json

    with open(load_path, 'r') as f:
        params_dict = json.load(f)

    params = {
        k: np.array(v, dtype=np.float64)
        for k, v in params_dict.items()
    }

    logger.info(f"Normalization parameters loaded from {load_path}")

    return params
