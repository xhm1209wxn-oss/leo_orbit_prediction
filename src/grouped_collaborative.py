"""Grouped collaborative residual learning for satellite orbit correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GroupedCollaborativeLearning:
    """
    Error-statistics-based grouped collaborative learning system.

    Each satellite group updates a group-specific model. Equal parameter
    aggregation then produces the shared model for the next iteration.
    """

    def __init__(self,
                 model_class,
                 model_kwargs: Dict,
                 num_groups: int,
                 device: torch.device,
                 model_type: str,
                 model_common: Dict[str, Any],
                 model_params: Dict[str, Any]):
        """
        Initialize the grouped collaborative learning system.

        参数：
            model_class: 模型类（如 AttentionLSTM）
            model_kwargs: 模型初始化参数
            num_groups: Satellite-group count
            device: 计算所用设备
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.num_groups = num_groups
        self.device = device
        self.model_type = str(model_type).lower()
        if not self.model_type:
            raise ValueError("model_type must be a non-empty string")
        self.model_common = dict(model_common)
        self.model_params = dict(model_params)

        self.shared_model = model_class(**model_kwargs).to(device)

        self.group_models = [
            model_class(**model_kwargs).to(device)
            for _ in range(num_groups)
        ]

        logger.info("Grouped collaborative learning initialized: %d groups", num_groups)
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Shared model parameters: "
                   f"{sum(p.numel() for p in self.shared_model.parameters()):,}")

    def aggregate_shared_parameters(
        self, group_weights_list: List[Dict], group_sample_counts: List[int]
    ):
        """
        Equally average group-specific parameters into shared parameters.

        参数：
            group_weights_list: Group-model state dictionaries
            group_sample_counts: Per-group sample counts, used only for length validation
        """
        if len(group_weights_list) != len(group_sample_counts):
            raise ValueError(
                "Length mismatch: group_weights_list and group_sample_counts must have the same length."
            )

        shared_dict = self.shared_model.state_dict()

        for key in shared_dict.keys():
            stacked = torch.stack([
                group_weights[key].float() for group_weights in group_weights_list
            ], dim=0)
            shared_dict[key] = stacked.mean(dim=0)

        self.shared_model.load_state_dict(shared_dict)

        logger.debug("Shared parameters updated by equal group averaging")

    def broadcast_shared_model(self):
        """Broadcast shared parameters to all group-specific models."""
        shared_weights = self.shared_model.state_dict()

        for group_model in self.group_models:
            group_model.load_state_dict(deepcopy(shared_weights))

        logger.debug("Shared parameters broadcast to all satellite groups")

    def get_group_model(self, group_id: int) -> nn.Module:
        """Return the model associated with one satellite group.

        Parameters:
            group_id: Satellite-group index
        """
        return self.group_models[group_id]

    def save_shared_model(self, save_path: str, **kwargs):
        """Save the shared-model checkpoint.

        参数：
            save_path: 模型保存路径
            **kwargs: 需要额外保存的内容
        """
        checkpoint = {
            'model_state_dict': self.shared_model.state_dict(),
            'model_type': self.model_type,
            'model_class': self.model_class.__name__,
            'model_kwargs': self.model_kwargs,
            'model_common': self.model_common,
            'model_params': self.model_params,
            'num_groups': self.num_groups,
            **kwargs
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Shared model saved to {save_path}")

    def load_shared_model(self, load_path: str) -> Dict:
        """Load a shared-model checkpoint.

        参数：
            load_path: 模型检查点路径

        返回：
            检查点字典
        """
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        checkpoint_model_type = checkpoint.get('model_type')
        if checkpoint_model_type is None:
            raise ValueError("Checkpoint missing required field: model_type")
        checkpoint_model_type = str(checkpoint_model_type).lower()
        if checkpoint_model_type != self.model_type:
            raise ValueError(
                f"Checkpoint model_type ({checkpoint_model_type}) does not match current model_type ({self.model_type})."
            )

        self.shared_model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Shared model loaded from {load_path}")

        return checkpoint


class GroupedCollaborativeTrainer:
    """
    Coordinates group-specific updates, parameter aggregation, and broadcasting.
    """

    def __init__(self,
                 collaborative_system: GroupedCollaborativeLearning,
                 group_loaders: List,
                 config,
                 device: torch.device,
                 group_val_loaders: List = None,
                 norm_params: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize the grouped collaborative trainer.

        参数：
            collaborative_system: Grouped collaborative learning system
            group_loaders: Group-specific training data loaders
            config: 配置对象
            device: 计算所用设备
            group_val_loaders: Group-specific validation data loaders
            norm_params: Shared normalization parameters for residuals in km
        """
        self.collaborative_system = collaborative_system
        self.group_loaders = group_loaders
        self.group_val_loaders = group_val_loaders
        self.config = config
        self.device = device
        self.norm_params = norm_params

        self._loss_type = str(getattr(config.training, 'loss_type', 'loss2')).lower()
        self._loss_delta = float(getattr(config.training, 'loss_delta', 1.0))
        self._loss3_mu = float(getattr(config.training, 'loss3_mu', 0.0))
        axis_weights = getattr(config.training, 'loss_axis_weights', [0.1, 0.8, 0.1])
        axis_weights = np.asarray(axis_weights, dtype=np.float32)
        axis_weight_sum = float(axis_weights.sum())
        if axis_weight_sum <= 0:
            raise ValueError('training.loss_axis_weights must have a positive sum')
        axis_weights = axis_weights / axis_weight_sum
        self._huber_axis_weights = torch.tensor(
            axis_weights.tolist(), dtype=torch.float32, device=self.device
        )
        self._y_std_tensor = None
        if norm_params is not None and 'y_std' in norm_params:
            y_std = np.asarray(norm_params['y_std'], dtype=np.float32)
            self._y_std_tensor = torch.from_numpy(y_std).to(self.device).view(1, -1)

        self.history = {
            'train_loss': [],
            'val_rounds': [],
            'val_loss': [],
            'val_t_rms_km': [],
            'learning_rates': []
        }

        logger.info("Grouped collaborative trainer initialized")
        logger.info(f"  Number of satellite groups: {len(group_loaders)}")
        if group_val_loaders is not None:
            total_val_samples = sum(len(loader.dataset) for loader in group_val_loaders)
            logger.info(f"  Validation: aggregated across {len(group_val_loaders)} groups")
            logger.info(f"  Total validation samples: {total_val_samples}")
        else:
            logger.info("  Validation: DISABLED (no validation data provided)")
        logger.info("  Loss type: %s", self._loss_type)
        if self._loss_type in {"loss2", "loss3"}:
            logger.info("  Loss delta: %.4f", self._loss_delta)
        if self._loss_type == "loss3":
            logger.info("  Loss 3 proximal mu: %.6f", self._loss3_mu)
        logger.info(
            "  Weighted loss axis weights (R/T/N): %.3f / %.3f / %.3f",
            float(self._huber_axis_weights[0]),
            float(self._huber_axis_weights[1]),
            float(self._huber_axis_weights[2]),
        )

    def _compute_loss3_proximal_term(
        self,
        group_model: nn.Module,
        shared_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss3 consensus regularization term.
        """
        prox = torch.zeros((), dtype=torch.float32, device=self.device)
        for name, param in group_model.named_parameters():
            if not param.requires_grad:
                continue
            prox = prox + torch.sum((param - shared_params[name]) ** 2)
        return 0.5 * self._loss3_mu * prox

    def _weighted_loss_per_sample(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算逐样本加权损失（最后一维视为 R/T/N 三轴）。
        支持 loss1 / loss2 / loss3，其中 loss3 的数据项与 loss2 相同。
        返回形状: (...)，即去掉最后一维后的逐样本损失。
        """
        if self._loss_type in {"loss2", "loss3"}:
            loss_per_axis = F.huber_loss(
                predictions, targets,
                delta=self._loss_delta,
                reduction='none'
            )
        elif self._loss_type == "loss1":
            loss_per_axis = F.mse_loss(
                predictions, targets,
                reduction='none'
            )
        else:
            raise ValueError(
                f"Unsupported loss_type: {self._loss_type}. Supported: loss1, loss2, loss3"
            )

        if loss_per_axis.size(-1) != self._huber_axis_weights.numel():
            raise ValueError(
                f"Expected last dim={self._huber_axis_weights.numel()} for weighted loss, "
                f"got {loss_per_axis.size(-1)}"
            )
        view_shape = [1] * (loss_per_axis.dim() - 1) + [self._huber_axis_weights.numel()]
        weights = self._huber_axis_weights.view(*view_shape)
        return (loss_per_axis * weights).sum(dim=-1)

    def _compute_round_lr(self, round_idx: int, num_rounds: int) -> float:
        """
        计算当前轮次学习率：
        1) 前 warmup_rounds 轮做线性 warmup
        2) warmup 之后再应用配置中的调度策略（cosine/none）
        """
        base_lr = float(self.config.training.optimizer.lr)
        lr_cfg = self.config.training.lr_schedule
        schedule_type = getattr(lr_cfg, "type", "none")
        warmup_rounds = int(getattr(lr_cfg, "warmup_rounds", 0))
        warmup_start_factor = float(getattr(lr_cfg, "warmup_start_factor", 0.1))

        if warmup_rounds > 0 and round_idx < warmup_rounds:
            if warmup_rounds == 1:
                return base_lr
            start_lr = base_lr * warmup_start_factor
            progress = round_idx / float(warmup_rounds - 1)
            return start_lr + (base_lr - start_lr) * progress

        if schedule_type == "cosine":
            min_lr = float(getattr(lr_cfg, "min_lr", 1e-5))
            effective_total = max(1, num_rounds - warmup_rounds)
            effective_idx = max(0, round_idx - warmup_rounds)
            lr = base_lr * (0.5 * (1 + np.cos(np.pi * effective_idx / effective_total)))
            return max(lr, min_lr)

        return base_lr

    def train_round(self,
                    round_idx: int,
                    num_rounds: int,
                    log_every_n_batches: int = 0,
                    collect_grad_stats: bool = False,
                    capture_grad_hist: bool = False) -> Tuple[float, float, Dict[str, Any]]:
        """
        Execute one grouped collaborative iteration.

        参数：
            round_idx: 当前轮次索引（从 0 开始）
            num_rounds: 总轮次数
            log_every_n_batches: 每隔多少个 batch 记录一次 batch 指标，<=0 表示不记录
            collect_grad_stats: 是否记录梯度范数
            capture_grad_hist: 是否抓取梯度快照（用于直方图）

        返回：
            (当前轮次加权平均训练损失, 当前学习率, 诊断信息字典)
        """
        self.collaborative_system.broadcast_shared_model()

        group_weights = []
        group_sample_counts = []
        round_train_losses = []
        logged_batch_losses = []
        logged_grad_norms = []
        grad_hist_snapshot = None
        batch_counter = 0

        lr = self._compute_round_lr(round_idx=round_idx, num_rounds=num_rounds)

        optimizer_name = self.config.training.optimizer.name.lower()

        for group_idx, group_model in enumerate(self.collaborative_system.group_models):
            if optimizer_name == "adam":
                optimizer = torch.optim.Adam(
                    group_model.parameters(),
                    lr=lr,
                    weight_decay=self.config.training.optimizer.weight_decay
                )
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(
                    group_model.parameters(),
                    lr=lr,
                    weight_decay=self.config.training.optimizer.weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer.name}. "
                                 f"Supported: adam, adamw")

            loader = self.group_loaders[group_idx]
            shared_params = None
            if self._loss_type == "loss3":
                shared_params = {
                    name: param.detach().clone()
                    for name, param in group_model.named_parameters()
                    if param.requires_grad
                }

            group_model.train()

            for epoch in range(self.config.training.collaborative.local_epochs):
                epoch_loss_sum = 0.0
                epoch_sample_count = 0

                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    predictions = group_model(batch_x)
                    data_loss = self._weighted_loss_per_sample(predictions, batch_y).mean()
                    loss = data_loss
                    if self._loss_type == "loss3":
                        loss = loss + self._compute_loss3_proximal_term(
                            group_model, shared_params
                        )

                    loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        group_model.parameters(),
                        self.config.training.grad_clip
                    )

                    if log_every_n_batches > 0 and (batch_counter % log_every_n_batches == 0):
                        logged_batch_losses.append(float(loss.item()))
                        if collect_grad_stats:
                            grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                            logged_grad_norms.append(grad_norm_value)

                    if capture_grad_hist and group_idx == 0:
                        grad_hist_snapshot = {
                            name: param.grad.detach().cpu().float().clone()
                            for name, param in group_model.named_parameters()
                            if param.grad is not None
                        }

                    optimizer.step()
                    batch_size = int(batch_y.size(0))
                    epoch_loss_sum += loss.item() * batch_size
                    epoch_sample_count += batch_size
                    batch_counter += 1

                avg_loss = epoch_loss_sum / max(epoch_sample_count, 1)

            round_train_losses.append(avg_loss)
            group_sample_counts.append(len(loader.dataset))

            group_weights.append(group_model.state_dict())

            logger.info(f"  Group {group_idx}: last update loss = {avg_loss:.4f} (LR={lr:.6f})")

        self.collaborative_system.aggregate_shared_parameters(group_weights, group_sample_counts)

        total_samples = sum(group_sample_counts)
        avg_train_loss = sum(
            loss * count for loss, count in zip(round_train_losses, group_sample_counts)
        ) / total_samples

        diagnostics = {
            'group_losses': [float(v) for v in round_train_losses],
            'batch_losses': logged_batch_losses,
            'batch_grad_norms': logged_grad_norms,
            'grad_hist_snapshot': grad_hist_snapshot
        }

        return avg_train_loss, lr, diagnostics

    def validate(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate the shared model across all group validation sets.

        返回：
            验证指标字典（如果没有验证数据则返回 None）
        """
        if self.group_val_loaders is None:
            logger.debug("validate() called but no validation data available")
            return None

        self.collaborative_system.shared_model.eval()
        total_weighted_loss = 0.0
        residual_sq_sum_km = np.zeros(3, dtype=np.float64)
        total_samples = 0

        with torch.no_grad():
            for group_val_loader in self.group_val_loaders:
                for batch_x, batch_y in group_val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    predictions = self.collaborative_system.shared_model(batch_x)
                    weighted_loss_per_sample = self._weighted_loss_per_sample(
                        predictions, batch_y
                    )
                    total_weighted_loss += weighted_loss_per_sample.sum().item()
                    residual = predictions - batch_y
                    if self._y_std_tensor is not None:
                        residual = residual * self._y_std_tensor
                    residual_sq_sum_km += (
                        residual.pow(2).sum(dim=0).detach().cpu().numpy().astype(np.float64)
                    )
                    total_samples += int(batch_y.size(0))

        avg_val_loss = total_weighted_loss / total_samples if total_samples > 0 else 0.0
        if total_samples > 0:
            val_rms_km = np.sqrt(residual_sq_sum_km / total_samples)
            val_t_rms_km = float(val_rms_km[1])
        else:
            val_rms_km = np.zeros(3, dtype=np.float64)
            val_t_rms_km = 0.0

        return {
            'val_loss': float(avg_val_loss),
            'val_t_rms_km': val_t_rms_km,
            'val_rms_km': val_rms_km.astype(float).tolist(),
        }

    def get_history(self) -> Dict:
        """
        获取训练历史

        返回：
            历史记录字典
        """
        return self.history
