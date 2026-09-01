"""
用于 TLE 误差预测的 PyTorch 数据集类
支持滑动窗口序列构建，使 LSTM/Attention 能建模时序依赖
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def _build_sequences_per_satellite(X_list: List[np.ndarray],
                                   y_list: List[np.ndarray],
                                   sequence_length: int
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    按卫星独立构建滑动窗口序列，避免跨卫星边界

    参数：
        X_list: 每颗卫星的特征数组列表，各元素形状 (n_i, features)
        y_list: 每颗卫星的标签数组列表，各元素形状 (n_i, 3)
        sequence_length: 滑动窗口长度

    返回：
        (X_seq, y_seq)
        - X_seq: (total_sequences, sequence_length, features)
        - y_seq: (total_sequences, 3)  标签取窗口最后一步
    """
    seq_X_list, seq_y_list = [], []

    for X_sat, y_sat in zip(X_list, y_list):
        n = len(X_sat)
        n_seq = n - sequence_length + 1
        if n_seq <= 0:
            if n > 0:
                pad_len = sequence_length - n
                X_padded = np.vstack([np.zeros((pad_len, X_sat.shape[1]), dtype=X_sat.dtype), X_sat])
                seq_X_list.append(X_padded[np.newaxis, :, :])  # (1, seq_len, features)
                seq_y_list.append(y_sat[-1:])  # (1, 3)
            continue

        stride_n, stride_f = X_sat.strides
        X_windows = np.lib.stride_tricks.as_strided(
            X_sat,
            shape=(n_seq, sequence_length, X_sat.shape[1]),
            strides=(stride_n, stride_n, stride_f)
        ).copy()  # copy 以确保内存连续

        y_windows = y_sat[sequence_length - 1:]

        seq_X_list.append(X_windows)
        seq_y_list.append(y_windows)

    if not seq_X_list:
        features = X_list[0].shape[1] if X_list else 19
        return (np.empty((0, sequence_length, features), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32))

    return np.vstack(seq_X_list), np.vstack(seq_y_list)


class TLEErrorDataset(Dataset):
    """
    来源于 TLE 交叉验证的 (D 维特征 -> 3 维 RTN 误差) 数据集
    支持滑动窗口序列（sequence_length > 1 时按卫星独立构建）

    特征（固定 19 维，ECI 坐标系）：
        - dt, sin(n_i*dt_ik), cos(n_i*dt_ik), r_i(3), v_i(3), r_ij(3), v_ij(3),
          sin(u_ik), cos(u_ik), h_ik, B*_i

    标签（3 维）：
        - RTN 误差：[R, T, N]（km）
    """

    def __init__(self,
                 error_datasets_dict: Dict[str, Dict],
                 mean: np.ndarray = None,
                 std: np.ndarray = None,
                 y_mean: np.ndarray = None,
                 y_std: np.ndarray = None,
                 normalize: bool = True,
                 sequence_length: int = 1):
        """
        参数：
            error_datasets_dict: {sat_name: {'X': features, 'y': labels}} 的字典
            sequence_length: 滑动窗口长度，>1 时构建时序序列
        """
        X_list, y_list = [], []

        for sat_name, data in error_datasets_dict.items():
            X = np.asarray(data.get('X'), dtype=np.float32)
            y = np.asarray(data.get('y'), dtype=np.float32)

            if X.size == 0 or y.size == 0:
                logger.warning(f"Satellite {sat_name} has no samples, skipping.")
                continue

            if len(X) != len(y):
                raise ValueError(f"Satellite {sat_name} has mismatched X/y lengths: {len(X)} vs {len(y)}.")

            X_list.append(X)
            y_list.append(y)

        if not X_list:
            raise ValueError("No samples available after filtering empty satellite datasets.")

        self.sequence_length = sequence_length

        if normalize:
            X_flat = np.vstack(X_list)
            y_flat = np.vstack(y_list)
            self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else X_flat.mean(axis=0)
            self.std = (np.asarray(std, dtype=np.float32) if std is not None else X_flat.std(axis=0)) + 1e-6
            self.y_mean = np.asarray(y_mean, dtype=np.float32) if y_mean is not None else y_flat.mean(axis=0)
            self.y_std = (np.asarray(y_std, dtype=np.float32) if y_std is not None else y_flat.std(axis=0)) + 1e-6

            X_list = [(X - self.mean) / self.std for X in X_list]
            y_list = [(y - self.y_mean) / self.y_std for y in y_list]
        else:
            self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else np.zeros(X_list[0].shape[1], dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32) if std is not None else np.ones(X_list[0].shape[1], dtype=np.float32)
            self.y_mean = np.asarray(y_mean, dtype=np.float32) if y_mean is not None else np.zeros(y_list[0].shape[1], dtype=np.float32)
            self.y_std = np.asarray(y_std, dtype=np.float32) if y_std is not None else np.ones(y_list[0].shape[1], dtype=np.float32)

        if sequence_length > 1:
            self.X, self.y = _build_sequences_per_satellite(X_list, y_list, sequence_length)
        else:
            self.X = np.vstack(X_list).astype(np.float32)
            self.y = np.vstack(y_list).astype(np.float32)

        logger.info(f"Dataset initialized: {len(self)} samples, "
                   f"X.shape={self.X.shape}, target_dim={self.y.shape[1]}")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std


class SatelliteDataset(Dataset):
    """
    单卫星组（多颗卫星）的数据集
    支持滑动窗口序列（按卫星独立构建，避免跨卫星边界）
    """

    def __init__(self,
                 X_per_sat: List[np.ndarray],
                 y_per_sat: List[np.ndarray],
                 mean: np.ndarray,
                 std: np.ndarray,
                 y_mean: np.ndarray = None,
                 y_std: np.ndarray = None,
                 sequence_length: int = 1):
        """
        参数：
            X_per_sat: 每颗卫星的特征数组列表
            y_per_sat: 每颗卫星的标签数组列表
            mean: 全局特征均值
            std: 全局特征标准差
            y_mean: 全局标签均值
            y_std: 全局标签标准差
            sequence_length: 滑动窗口长度
        """
        self.sequence_length = sequence_length

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32) + 1e-6
        y_mean = np.asarray(y_mean, dtype=np.float32) if y_mean is not None else np.zeros(3, dtype=np.float32)
        y_std = np.asarray(y_std, dtype=np.float32) if y_std is not None else np.ones(3, dtype=np.float32)
        y_std = y_std + 1e-6

        X_normed = [(X.astype(np.float32) - mean) / std for X in X_per_sat]
        y_normed = [(y.astype(np.float32) - y_mean) / y_std for y in y_per_sat]

        if sequence_length > 1:
            self.X, self.y = _build_sequences_per_satellite(X_normed, y_normed, sequence_length)
        else:
            self.X = np.vstack(X_normed).astype(np.float32)
            self.y = np.vstack(y_normed).astype(np.float32)

        logger.debug(f"Satellite dataset: {len(self)} samples, X.shape={self.X.shape}")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def create_group_dataset(error_datasets_dict: Dict[str, Dict],
                         mean: np.ndarray,
                         std: np.ndarray,
                         y_mean: np.ndarray,
                         y_std: np.ndarray,
                         sequence_length: int = 1) -> TLEErrorDataset:
    """
    创建一组卫星的数据集（用于验证集/测试集）

    参数：
        error_datasets_dict: {sat_name: {'X': features, 'y': labels}}
        mean, std, y_mean, y_std: 全局标准化参数
        sequence_length: 滑动窗口长度

    返回：
        TLEErrorDataset
    """
    return TLEErrorDataset(
        error_datasets_dict,
        mean=mean, std=std,
        y_mean=y_mean, y_std=y_std,
        sequence_length=sequence_length
    )


def create_group_datasets(error_datasets: Dict[str, Dict],
                          satellite_lists: List[List[str]],
                          mean: np.ndarray,
                          std: np.ndarray,
                          y_mean: np.ndarray,
                          y_std: np.ndarray,
                          sequence_length: int = 1) -> List[Dataset]:
    """
    为每个卫星组创建数据集（使用共享归一化参数）。

    参数：
        error_datasets: 按卫星存储的误差数据字典
        satellite_lists: 每个卫星组对应的卫星名称列表集合
        mean: 用于标准化的特征均值
        std: 用于标准化的特征标准差
        y_mean: 用于标准化的标签均值
        y_std: 用于标准化的标签标准差
        sequence_length: 滑动窗口长度

    返回：
        各卫星组的 SatelliteDataset 列表
    """
    group_datasets = []

    for group_idx, sat_list in enumerate(satellite_lists):
        X_per_sat, y_per_sat = [], []

        for sat_name in sat_list:
            if sat_name in error_datasets:
                sat_X = np.asarray(error_datasets[sat_name]['X'])
                sat_y = np.asarray(error_datasets[sat_name]['y'])

                if sat_X.size == 0 or sat_y.size == 0:
                    logger.warning(f"Satellite {sat_name} has no samples, skipping for group {group_idx}.")
                    continue

                X_per_sat.append(sat_X)
                y_per_sat.append(sat_y)

        if not X_per_sat:
            raise ValueError(f"Group {group_idx} has no samples after filtering; check data generation.")

        group_dataset = SatelliteDataset(
            X_per_sat, y_per_sat,
            mean, std, y_mean, y_std,
            sequence_length=sequence_length
        )
        group_datasets.append(group_dataset)

        logger.info(f"Group {group_idx}: {len(group_dataset)} samples "
                    f"from {len(sat_list)} satellites (seq_len={sequence_length})")

    return group_datasets


def create_group_dataloaders(group_datasets: List[Dataset],
                             batch_size: int = 256,
                             shuffle: bool = True,
                             num_workers: int = 0) -> List[DataLoader]:
    """
    为各卫星组创建数据加载器。

    参数：
        group_datasets: 卫星组数据集列表
        batch_size: 批大小
        shuffle: 是否打乱数据

    返回：
        各卫星组对应的 DataLoader 列表
    """
    group_loaders = []

    for group_idx, dataset in enumerate(group_datasets):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        group_loaders.append(loader)

        logger.debug(f"Group {group_idx} loader: {len(dataset)} samples, "
                    f"{len(loader)} batches")

    return group_loaders
