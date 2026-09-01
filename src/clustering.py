"""
卫星聚类模块
支持基于训练集 T 方向误差统计特征的 KMeans / GMM / Agglomerative 聚类，
并可按“每颗卫星训练样本数”做容量约束重分配，平衡卫星组训练贡献。
"""

import logging
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


T_ERROR_FEATURE_COLUMNS = [
    'T_mean',
    'T_std',
    'T_rms',
    'T_abs_mean',
    'T_median',
    'T_p90_abs',
]

ORBITAL_ALT_INC_FEATURE_COLUMNS = [
    'Altitude_km',
    'Inclination_deg',
]


def build_t_error_stat_feature_table(
    train_error_datasets: Dict[str, Dict[str, np.ndarray]],
    orbital_elements: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    基于训练集每颗卫星 T 方向误差构建统计特征表。
    """
    rows = []
    for sat_name in sorted(train_error_datasets.keys()):
        sat_data = train_error_datasets[sat_name]
        y = sat_data.get('y')
        if y is None or len(y) == 0:
            continue

        t_vals = np.asarray(y)[:, 1]
        t_abs = np.abs(t_vals)

        row = {
            'Satellite': sat_name,
            'T_mean': float(np.mean(t_vals)),
            'T_std': float(np.std(t_vals)),
            'T_rms': float(np.sqrt(np.mean(t_vals ** 2))),
            'T_abs_mean': float(np.mean(t_abs)),
            'T_median': float(np.median(t_vals)),
            'T_p90_abs': float(np.percentile(t_abs, 90)),
        }

        if orbital_elements is not None and sat_name in orbital_elements:
            row.update({
                'Altitude_km': float(orbital_elements[sat_name]['a_mean'] - 6378.137),
                'Inclination_deg': float(orbital_elements[sat_name]['i_mean']),
                'Eccentricity': float(orbital_elements[sat_name]['e_mean']),
                'RAAN_deg': float(orbital_elements[sat_name]['Omega_mean']),
            })

        rows.append(row)

    if not rows:
        raise ValueError("No valid satellites found for T-error-stat clustering")

    return pd.DataFrame(rows).sort_values("Satellite").reset_index(drop=True)


def build_orbital_alt_inc_feature_table(
    orbital_elements: Dict[str, Dict[str, float]],
    satellite_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    基于每颗卫星的平均轨道高度和倾角构建聚类特征表。
    """
    rows = []
    target_satellites = sorted(satellite_names) if satellite_names is not None else sorted(orbital_elements.keys())

    for sat_name in target_satellites:
        if sat_name not in orbital_elements:
            raise ValueError(f"Missing orbital elements for satellite: {sat_name}")

        sat_elements = orbital_elements[sat_name]
        rows.append({
            'Satellite': sat_name,
            'Altitude_km': float(sat_elements['a_mean'] - 6378.137),
            'Inclination_deg': float(sat_elements['i_mean']),
        })

    if not rows:
        raise ValueError("No valid satellites found for orbital_alt_inc clustering")

    return pd.DataFrame(rows).sort_values("Satellite").reset_index(drop=True)


def get_feature_columns(feature_mode: str = 't_error_stats') -> List[str]:
    """
    返回当前聚类模式使用的特征列。
    """
    if feature_mode == 't_error_stats':
        return T_ERROR_FEATURE_COLUMNS.copy()
    if feature_mode == 'orbital_alt_inc':
        return ORBITAL_ALT_INC_FEATURE_COLUMNS.copy()
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def build_feature_matrix(feature_table: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    """
    对聚类特征做标准化，得到可直接用于聚类的特征矩阵。
    """
    raw_matrix = feature_table[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    return scaler.fit_transform(raw_matrix)


def _has_all_clusters(labels: np.ndarray, n_clusters: int) -> bool:
    """
    判断硬标签是否覆盖所有簇。
    """
    return np.unique(labels).size == int(n_clusters)


def _compute_label_centers(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """
    用每簇样本均值作为兼容中心，供无原生中心的聚类方法复用接口。
    """
    centers = []
    for cluster_id in range(n_clusters):
        cluster_points = feature_matrix[labels == cluster_id]
        if cluster_points.size == 0:
            raise ValueError(
                f"Cluster {cluster_id} is empty; cannot compute cluster center"
            )
        centers.append(np.mean(cluster_points, axis=0))
    return np.asarray(centers, dtype=np.float64)


def _repair_empty_gmm_clusters(
    labels: np.ndarray,
    responsibilities: np.ndarray,
    n_clusters: int
) -> Optional[np.ndarray]:
    """
    用 GMM 后验责任概率为没有样本的分量补一个样本，尽量保持 GMM 语义。
    """
    repaired = np.asarray(labels, dtype=np.int64).copy()
    counts = np.bincount(repaired, minlength=n_clusters).astype(np.int64, copy=False)
    empty_clusters = [c for c in range(n_clusters) if counts[c] == 0]

    for dst in empty_clusters:
        donor_candidates = np.where(counts[repaired] > 1)[0]
        if donor_candidates.size == 0:
            return None

        best_idx = int(donor_candidates[np.argmax(responsibilities[donor_candidates, dst])])
        src = int(repaired[best_idx])
        if counts[src] <= 1:
            return None

        repaired[best_idx] = dst
        counts[src] -= 1
        counts[dst] += 1

    if not _has_all_clusters(repaired, n_clusters):
        return None
    return repaired


def _perform_gmm_clustering(
    feature_matrix: np.ndarray,
    n_clusters: int,
    max_attempts: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行 GMM 聚类，并保证最终硬标签覆盖所有分量。
    """
    best_full_clustering = None
    best_full_labels = None
    best_full_score = None
    best_repair_clustering = None
    best_repair_labels = None
    best_repair_score = None

    for attempt_idx in range(max_attempts):
        clustering = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            n_init=1,
            random_state=42 + attempt_idx
        )
        labels = clustering.fit_predict(feature_matrix).astype(np.int64, copy=False)
        score = float(clustering.score(feature_matrix))

        if _has_all_clusters(labels, n_clusters):
            if best_full_clustering is None or score > best_full_score:
                best_full_clustering = clustering
                best_full_labels = labels.copy()
                best_full_score = score
            continue

        if best_repair_clustering is None or score > best_repair_score:
            best_repair_clustering = clustering
            best_repair_labels = labels.copy()
            best_repair_score = score

    if best_full_clustering is not None:
        return best_full_labels, best_full_clustering.means_

    responsibilities = best_repair_clustering.predict_proba(feature_matrix)
    repaired_labels = _repair_empty_gmm_clusters(
        labels=best_repair_labels,
        responsibilities=responsibilities,
        n_clusters=n_clusters
    )
    if repaired_labels is None:
        raise ValueError(
            "GMM clustering produced empty clusters and responsibility-based repair failed. "
            "Please reduce clustering.n_clusters or change clustering.method."
        )

    logger.warning(
        "GMM produced empty hard clusters after %d initialization attempts; "
        "applied responsibility-based repair to retain %d satellite groups.",
        max_attempts,
        n_clusters
    )
    return repaired_labels, best_repair_clustering.means_


def perform_clustering(
    feature_matrix: np.ndarray,
    n_clusters: int = 2,
    method: str = 'kmeans'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行聚类，返回标签与簇中心。
    """
    method = str(method).lower()

    if method == 'kmeans':
        clustering = KMeans(
            n_clusters=n_clusters,
            n_init=20,
            random_state=42
        )
        labels = clustering.fit_predict(feature_matrix)
        return labels, clustering.cluster_centers_

    if method == 'gmm':
        return _perform_gmm_clustering(
            feature_matrix=feature_matrix,
            n_clusters=n_clusters
        )

    if method == 'agglomerative':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(feature_matrix).astype(np.int64, copy=False)
        centers = _compute_label_centers(
            feature_matrix=feature_matrix,
            labels=labels,
            n_clusters=n_clusters
        )
        return labels, centers

    raise ValueError(f"Unsupported clustering method: {method}")


def _compute_cluster_stats(
    labels: np.ndarray,
    sample_weights: np.ndarray,
    n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每簇卫星数与样本负载。
    """
    counts = np.bincount(labels, minlength=n_clusters).astype(np.int64, copy=False)
    loads = np.bincount(labels, weights=sample_weights, minlength=n_clusters).astype(np.float64, copy=False)
    return counts, loads


def _total_capacity_violation(loads: np.ndarray, lower: float, upper: float) -> float:
    overload = np.maximum(0.0, loads - upper)
    underload = np.maximum(0.0, lower - loads)
    return float(np.sum(overload + underload))


def _distance_matrix_to_centers(
    feature_matrix: np.ndarray,
    centers: np.ndarray
) -> np.ndarray:
    """
    计算样本到每个簇中心的平方欧氏距离。
    """
    diff = feature_matrix[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _fill_empty_clusters(
    labels: np.ndarray,
    counts: np.ndarray,
    loads: np.ndarray,
    dist_sq: np.ndarray,
    sample_weights: np.ndarray
) -> bool:
    """
    如果存在空簇，从负载较高簇中迁移“代价最小”的卫星填充空簇。
    """
    n_clusters = len(counts)
    for dst in range(n_clusters):
        if counts[dst] > 0:
            continue

        src_candidates = [c for c in range(n_clusters) if counts[c] > 1]
        if not src_candidates:
            return False
        src = max(src_candidates, key=lambda c: loads[c])

        src_indices = np.where(labels == src)[0]
        if src_indices.size == 0:
            return False

        best_idx = None
        best_delta = None
        for i in src_indices:
            delta = float(dist_sq[i, dst] - dist_sq[i, src])
            if best_idx is None or delta < best_delta:
                best_idx = int(i)
                best_delta = delta

        if best_idx is None:
            return False

        w = float(sample_weights[best_idx])
        labels[best_idx] = dst
        counts[src] -= 1
        counts[dst] += 1
        loads[src] -= w
        loads[dst] += w

    return True


def _find_best_capacity_move(
    labels: np.ndarray,
    counts: np.ndarray,
    loads: np.ndarray,
    dist_sq: np.ndarray,
    sample_weights: np.ndarray,
    src_clusters: List[int],
    dst_clusters: List[int],
    lower: float,
    upper: float
) -> Optional[Tuple[int, int, int, float]]:
    """
    搜索一个最优迁移动作 (sat_idx, src, dst, cost_delta)。
    目标优先减少容量违规，再尽量减少聚类失真增量。
    """
    best_move = None
    best_violation_drop = None
    best_cost_delta = None
    eps = 1e-9

    for src in src_clusters:
        if counts[src] <= 1:
            continue
        src_indices = np.where(labels == src)[0]
        if src_indices.size == 0:
            continue

        for sat_idx in src_indices:
            w = float(sample_weights[sat_idx])
            src_after = float(loads[src] - w)
            if src_after < lower - eps:
                continue

            src_dist = float(dist_sq[sat_idx, src])

            for dst in dst_clusters:
                if dst == src:
                    continue
                dst_after = float(loads[dst] + w)
                if dst_after > upper + eps:
                    continue

                before_local = (
                    max(0.0, loads[src] - upper) + max(0.0, lower - loads[src]) +
                    max(0.0, loads[dst] - upper) + max(0.0, lower - loads[dst])
                )
                after_local = (
                    max(0.0, src_after - upper) + max(0.0, lower - src_after) +
                    max(0.0, dst_after - upper) + max(0.0, lower - dst_after)
                )
                violation_drop = float(before_local - after_local)
                cost_delta = float(dist_sq[sat_idx, dst] - src_dist)

                if best_move is None:
                    best_move = (int(sat_idx), int(src), int(dst), cost_delta)
                    best_violation_drop = violation_drop
                    best_cost_delta = cost_delta
                    continue

                if violation_drop > best_violation_drop + eps:
                    best_move = (int(sat_idx), int(src), int(dst), cost_delta)
                    best_violation_drop = violation_drop
                    best_cost_delta = cost_delta
                    continue

                if abs(violation_drop - best_violation_drop) <= eps and cost_delta < best_cost_delta:
                    best_move = (int(sat_idx), int(src), int(dst), cost_delta)
                    best_violation_drop = violation_drop
                    best_cost_delta = cost_delta

    if best_move is None:
        return None
    if best_violation_drop is not None and best_violation_drop < -1e-9:
        return None
    return best_move


def _repair_assignment_with_capacity_bounds(
    initial_labels: np.ndarray,
    feature_matrix: np.ndarray,
    centers: np.ndarray,
    sample_weights: np.ndarray,
    lower: float,
    upper: float
) -> Optional[np.ndarray]:
    """
    在给定容量上下界下修复分配。
    成功则返回新标签，失败返回 None。
    """
    labels = np.asarray(initial_labels, dtype=np.int64).copy()
    n_clusters = centers.shape[0]
    dist_sq = _distance_matrix_to_centers(feature_matrix, centers)

    counts, loads = _compute_cluster_stats(labels, sample_weights, n_clusters)

    if not _fill_empty_clusters(labels, counts, loads, dist_sq, sample_weights):
        return None

    max_iters = max(2000, int(feature_matrix.shape[0] * 5))
    eps = 1e-6

    for _ in range(max_iters):
        violation = _total_capacity_violation(loads, lower, upper)
        if violation <= eps:
            return labels

        over_clusters = [c for c in range(n_clusters) if loads[c] > upper + eps]
        under_clusters = [c for c in range(n_clusters) if loads[c] < lower - eps]

        if over_clusters and under_clusters:
            move = _find_best_capacity_move(
                labels=labels,
                counts=counts,
                loads=loads,
                dist_sq=dist_sq,
                sample_weights=sample_weights,
                src_clusters=over_clusters,
                dst_clusters=under_clusters,
                lower=lower,
                upper=upper
            )
        elif over_clusters:
            dst_clusters = [c for c in range(n_clusters) if loads[c] < upper - eps]
            move = _find_best_capacity_move(
                labels=labels,
                counts=counts,
                loads=loads,
                dist_sq=dist_sq,
                sample_weights=sample_weights,
                src_clusters=over_clusters,
                dst_clusters=dst_clusters,
                lower=lower,
                upper=upper
            )
        else:
            src_clusters = [c for c in range(n_clusters) if loads[c] > lower + eps]
            move = _find_best_capacity_move(
                labels=labels,
                counts=counts,
                loads=loads,
                dist_sq=dist_sq,
                sample_weights=sample_weights,
                src_clusters=src_clusters,
                dst_clusters=under_clusters,
                lower=lower,
                upper=upper
            )

        if move is None:
            return None

        sat_idx, src, dst, _ = move
        if src == dst:
            return None
        if counts[src] <= 1:
            return None

        w = float(sample_weights[sat_idx])
        labels[sat_idx] = dst
        counts[src] -= 1
        counts[dst] += 1
        loads[src] -= w
        loads[dst] += w

    return None


def balance_cluster_assignment_by_samples(
    feature_matrix: np.ndarray,
    initial_labels: np.ndarray,
    centers: np.ndarray,
    sample_weights: np.ndarray,
    tolerance: float = 0.15,
    relax_step: float = 0.05,
    max_relax_steps: int = 3
) -> np.ndarray:
    """
    在 KMeans 初始聚类后，按样本负载做容量约束重分配。

    - 初始约束：每簇负载在 [target*(1-tolerance), target*(1+tolerance)]。
    - 若不可行，逐步放宽 tolerance，最多放宽 max_relax_steps 次。
    - 若仍不可行，回退到原始 KMeans 分配。
    """
    labels = np.asarray(initial_labels, dtype=np.int64)
    n_clusters = centers.shape[0]
    if n_clusters <= 1:
        return labels.copy()

    if sample_weights.shape[0] != labels.shape[0]:
        raise ValueError("sample_weights length mismatch with labels")
    if np.any(sample_weights <= 0):
        raise ValueError("sample_weights must be positive")

    total_load = float(np.sum(sample_weights))
    target = total_load / float(n_clusters)

    for relax_idx in range(int(max_relax_steps) + 1):
        current_tol = float(tolerance + relax_idx * relax_step)
        lower = target * max(0.0, (1.0 - current_tol))
        upper = target * (1.0 + current_tol)

        candidate = _repair_assignment_with_capacity_bounds(
            initial_labels=labels,
            feature_matrix=feature_matrix,
            centers=centers,
            sample_weights=sample_weights,
            lower=lower,
            upper=upper
        )
        if candidate is not None:
            return candidate

    logger.warning(
        "Balanced reassignment failed after relax steps; fallback to raw KMeans labels."
    )
    return labels.copy()


def create_clustering_results(
    sat_names: List[str],
    cluster_labels: np.ndarray,
    feature_table: pd.DataFrame
) -> pd.DataFrame:
    """
    创建聚类结果表，并带上当前聚类特征/辅助列。
    """
    results = pd.DataFrame({
        'Satellite': sat_names,
        'Cluster': cluster_labels,
    })
    merged = results.merge(feature_table, on='Satellite', how='left')
    base_cols = ['Satellite', 'Cluster']
    other_cols = [c for c in merged.columns if c not in base_cols]
    return merged[base_cols + other_cols]


def get_group_satellite_lists(clustering_results: pd.DataFrame) -> List[List[str]]:
    """
    获取每个卫星组（聚类）的卫星列表。
    """
    group_lists = []
    n_clusters = clustering_results['Cluster'].nunique()

    for cluster_id in sorted(clustering_results['Cluster'].unique().tolist()):
        sat_list = sorted(
            clustering_results[
                clustering_results['Cluster'] == cluster_id
            ]['Satellite'].tolist()
        )
        group_lists.append(sat_list)

    return group_lists


def run_clustering_pipeline(
    orbital_elements: Dict[str, Dict[str, float]],
    config,
    train_error_datasets: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    运行聚类流程。

    返回：
        (clustering_results, feature_matrix)
    """
    feature_mode = getattr(config.clustering, 'feature_mode', 't_error_stats')
    method = getattr(config.clustering, 'method', 'kmeans')
    if method in {'agglomerative', 'gmm'} and bool(getattr(config.clustering, 'balance_by_samples', True)):
        raise ValueError("clustering.balance_by_samples must be false when clustering.method is 'gmm' or 'agglomerative'")
    if feature_mode == 't_error_stats':
        if train_error_datasets is None:
            raise ValueError("train_error_datasets is required for t_error_stats clustering")
        feature_table = build_t_error_stat_feature_table(
            train_error_datasets=train_error_datasets,
            orbital_elements=orbital_elements
        )
    elif feature_mode == 'orbital_alt_inc':
        feature_table = build_orbital_alt_inc_feature_table(
            orbital_elements=orbital_elements,
            satellite_names=(list(train_error_datasets.keys()) if train_error_datasets is not None else None)
        )
    else:
        raise ValueError(
            f"Unsupported clustering.feature_mode: {feature_mode}"
        )

    sat_names = feature_table['Satellite'].tolist()
    feature_columns = get_feature_columns(feature_mode)
    feature_matrix = build_feature_matrix(feature_table, feature_columns)

    raw_labels, centers = perform_clustering(
        feature_matrix,
        n_clusters=config.clustering.n_clusters,
        method=method,
    )

    final_labels = raw_labels
    sample_weights = None
    if (
        getattr(config.clustering, 'balance_by_samples', True) and
        int(config.clustering.n_clusters) > 1
    ):
        if train_error_datasets is None:
            raise ValueError("train_error_datasets is required when balance_by_samples is enabled")
        sample_weights = np.asarray(
            [len(train_error_datasets[s]['X']) for s in sat_names],
            dtype=np.float64
        )
        final_labels = balance_cluster_assignment_by_samples(
            feature_matrix=feature_matrix,
            initial_labels=raw_labels,
            centers=centers,
            sample_weights=sample_weights,
            tolerance=float(getattr(config.clustering, 'balance_tolerance', 0.15)),
            relax_step=float(getattr(config.clustering, 'balance_relax_step', 0.05)),
            max_relax_steps=int(getattr(config.clustering, 'balance_max_relax_steps', 3))
        )

        n_clusters = int(config.clustering.n_clusters)
        _, raw_loads = _compute_cluster_stats(raw_labels, sample_weights, n_clusters)
        _, final_loads = _compute_cluster_stats(final_labels, sample_weights, n_clusters)
        raw_ratio = float(raw_loads.max() / max(raw_loads.min(), 1e-12))
        final_ratio = float(final_loads.max() / max(final_loads.min(), 1e-12))
        logger.info(
            "Cluster load balance (train samples) improved: raw max/min=%.3f -> balanced max/min=%.3f",
            raw_ratio,
            final_ratio
        )

    clustering_results = create_clustering_results(
        sat_names,
        final_labels,
        feature_table
    )

    logger.info(
        "Clustering completed: feature_mode=%s, method=%s, satellites=%d, features=%s",
        feature_mode,
        method,
        len(sat_names),
        ",".join(feature_columns)
    )

    return clustering_results, feature_matrix
