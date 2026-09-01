"""Training loop for grouped collaborative residual learning."""

import torch
import logging
import math
from typing import Dict
from pathlib import Path

from grouped_collaborative import GroupedCollaborativeTrainer
from utils import EarlyStopping

logger = logging.getLogger(__name__)


def train_grouped_collaborative_model(
    collaborative_trainer: GroupedCollaborativeTrainer,
    num_rounds: int,
    early_stopping: EarlyStopping = None,
    save_dir: str = None,
) -> Dict:
    """
    Run multiple grouped collaborative iterations.

    参数：
        collaborative_trainer: GroupedCollaborativeTrainer instance
        num_rounds: Number of collaborative iterations
        early_stopping: 提前停止处理器（可选）
        save_dir: 检查点保存目录（可选）

    返回：
        训练历史记录字典
    """
    logger.info("="* 70)
    logger.info("GROUPED COLLABORATIVE TRAINING")
    logger.info("=" * 70)
    logger.info(f"Training configuration:")
    logger.info(f"  Total rounds: {num_rounds}")
    logger.info(
        f"  Group-update epochs per iteration: "
        f"{collaborative_trainer.config.training.collaborative.local_epochs}"
    )
    logger.info(f"  Batch size: {collaborative_trainer.config.training.batch_size}")
    logger.info(f"  Base learning rate: {collaborative_trainer.config.training.optimizer.lr}")
    model_sel_cfg = getattr(collaborative_trainer.config.training, "model_selection", None)
    validation_cfg = collaborative_trainer.config.training.validation
    eval_every_rounds = int(validation_cfg.eval_every_rounds)
    monitor_metric = str(getattr(model_sel_cfg, "metric", "val_loss"))
    model_selection_mode = str(getattr(model_sel_cfg, "mode", "min"))
    model_selection_min_delta = float(getattr(model_sel_cfg, "min_delta", 0.0))
    logger.info(f"  Validation cadence: every {eval_every_rounds} round(s)")
    logger.info(f"  Model selection metric: {monitor_metric}")
    logger.info(
        f"  Best model selection: mode={model_selection_mode}, min_delta={model_selection_min_delta}"
    )
    if early_stopping is not None:
        logger.info(f"  Early stopping: enabled (patience={early_stopping.patience}, min_delta={early_stopping.min_delta})")
    else:
        logger.info(f"  Early stopping: disabled")
    logger.info("=" * 70)

    best_model_path = None
    if save_dir:
        best_model_path = Path(save_dir) / "best_model.pth"
    best_model_monitor_value = None
    best_model_round = None

    for round_idx in range(num_rounds):
        logger.info(f"\n{'='*70}")
        logger.info(f"Collaborative iteration {round_idx + 1}/{num_rounds}")
        logger.info(f"{'='*70}")

        avg_train_loss, current_lr, _ = collaborative_trainer.train_round(
            round_idx,
            num_rounds
        )
        collaborative_trainer.history['train_loss'].append(avg_train_loss)
        collaborative_trainer.history['learning_rates'].append(current_lr)

        run_validation = ((round_idx + 1) % eval_every_rounds == 0)
        avg_val_loss = None
        val_t_rms_km = None

        if run_validation:
            val_metrics = collaborative_trainer.validate()

            if val_metrics is None:
                avg_val_loss = avg_train_loss
                val_t_rms_km = float("nan")
                logger.info(f"  Iteration {round_idx + 1} → Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")
                logger.info(f"  (No validation data - monitoring via train loss)")
            else:
                avg_val_loss = float(val_metrics['val_loss'])
                val_t_rms_km = float(val_metrics['val_t_rms_km'])
                logger.info(
                    f"  Iteration {round_idx + 1} → Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val T-RMS(km): {val_t_rms_km:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            collaborative_trainer.history['val_rounds'].append(round_idx + 1)
            collaborative_trainer.history['val_loss'].append(avg_val_loss)
            collaborative_trainer.history['val_t_rms_km'].append(val_t_rms_km)

            if monitor_metric == "val_t_rms_km":
                monitor_value = val_t_rms_km
                if not math.isfinite(monitor_value):
                    monitor_value = avg_val_loss
            else:
                monitor_value = avg_val_loss

            if best_model_path is not None:
                if best_model_monitor_value is None:
                    improved_for_model_selection = True
                else:
                    if model_selection_mode == "min":
                        improved_for_model_selection = (
                            monitor_value < (best_model_monitor_value - model_selection_min_delta)
                        )
                    else:
                        improved_for_model_selection = (
                            monitor_value > (best_model_monitor_value + model_selection_min_delta)
                        )

                if improved_for_model_selection:
                    best_model_monitor_value = monitor_value
                    best_model_round = round_idx
                    collaborative_trainer.collaborative_system.save_shared_model(
                        str(best_model_path),
                        round=round_idx + 1,
                        train_loss=avg_train_loss,
                        val_loss=avg_val_loss,
                        val_t_rms_km=val_t_rms_km,
                        monitor_metric=monitor_metric,
                        monitor_value=monitor_value,
                        is_best=True
                    )
                    logger.info(f"  ✓ Best model saved to {best_model_path}")

            if early_stopping is not None:
                if early_stopping(monitor_value, round_idx):
                    logger.info(f"\n{'='*70}")
                    logger.info(f"Early stopping triggered at iteration {round_idx + 1}")
                    logger.info(
                        f"Best iteration was {early_stopping.best_epoch + 1} with "
                        f"Best monitor metric ({monitor_metric}) = {early_stopping.best_score:.4f}"
                    )
                    logger.info(f"{'='*70}")
                    break
        else:
            logger.info(
                f"  Iteration {round_idx + 1} → Train Loss: {avg_train_loss:.4f} | "
                f"LR: {current_lr:.6f} (validation skipped)"
            )

        if save_dir and (round_idx + 1) % 10 == 0:
            save_path = Path(save_dir) / f"checkpoint_round_{round_idx + 1}.pth"
            collaborative_trainer.collaborative_system.save_shared_model(
                str(save_path),
                round=round_idx + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                val_t_rms_km=val_t_rms_km
            )
            logger.info(f"  Checkpoint saved to {save_path}")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training completed!")
    if early_stopping is not None:
        if early_stopping.best_score is not None:
            logger.info(f"  Best iteration (for early stopping): {early_stopping.best_epoch + 1}/{num_rounds}")
            logger.info(
                f"  Best monitor metric for early stopping ({monitor_metric}): "
                f"{early_stopping.best_score:.4f}"
            )
        else:
            logger.info("  Early stopping state: no validation rounds were executed")
    if best_model_round is not None and best_model_monitor_value is not None:
        logger.info(
            f"  Best model checkpoint iteration: {best_model_round + 1}/{num_rounds} "
            f"(monitor={best_model_monitor_value:.4f})"
        )
    if best_model_path and best_model_path.exists():
        logger.info(f"  Best model saved at: {best_model_path}")
    logger.info("=" * 70)

    return collaborative_trainer.get_history()
