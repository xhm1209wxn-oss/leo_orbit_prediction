#!/usr/bin/env python3
"""Main experiment entry point for grouped collaborative residual learning."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config, validate_config
from utils import (setup_logging, setup_reproducibility, setup_device,
                   ensure_directory, EarlyStopping)
from data_download import TLEDownloader
from preprocessing import (parse_epoch, compute_orbital_elements, propagate_satrec_to_time,
                          build_tle_error_dataset,
                          save_normalization_params, cross_satellite_iqr_filter)
from sgp4.api import Satrec
from grouped_preprocessing import compute_group_statistics, aggregate_group_statistics
from clustering import run_clustering_pipeline, get_group_satellite_lists
from models import build_model_spec, get_model_metadata
from dataset import (TLEErrorDataset,
                    create_group_datasets,
                    create_group_dataloaders)
from torch.utils.data import DataLoader
from grouped_collaborative import GroupedCollaborativeLearning, GroupedCollaborativeTrainer
from train import train_grouped_collaborative_model
from visualization import plot_training_history, plot_dt_distribution
import torch

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _load_runtime_config(config_path: str):
    loaded_config = load_config(config_path)
    return loaded_config


def _build_timestamped_output_root(base_root: Path) -> Path:
    """
    基于基准输出目录生成唯一的时间戳目录，例如 outputs20260224-123456-123456
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    candidate = base_root.parent / f"{base_root.name}{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_root.parent / f"{base_root.name}{stamp}-{suffix}"
        suffix += 1
    return candidate


def _apply_timestamped_output_paths(config) -> Path:
    """
    将配置中的输出路径重定向到时间戳目录，避免并行实验互相覆盖。
    """
    base_root = Path(config.outputs.paths.models).parent
    run_root = _build_timestamped_output_root(base_root)

    config.outputs.paths.models = str(run_root / "models")
    config.outputs.paths.plots = str(run_root / "plots")
    config.outputs.paths.metrics = str(run_root / "metrics")
    config.outputs.paths.clustering = str(run_root / "clustering")

    return run_root



def main():
    """主要的实验流程"""

    parser = argparse.ArgumentParser(
        description='Grouped Collaborative Residual Correction of TLE-based Orbit Predictions'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    config = _load_runtime_config(args.config)

    run_output_root = _apply_timestamped_output_paths(config)

    setup_logging(
        log_level=config.logging.level,
        console=config.logging.console
    )

    logger.info("="* 70)
    logger.info("GROUPED COLLABORATIVE RESIDUAL CORRECTION OF TLE-BASED ORBIT PREDICTIONS")
    logger.info("=" * 70)
    logger.info(f"Run outputs directory: {run_output_root}")

    validate_config(config)

    filter_cfg = getattr(config.data, 'filter', None)
    filter_enabled = bool(getattr(filter_cfg, 'enabled', False))
    min_sat_samples_before_split = int(getattr(filter_cfg, 'min_sat_samples_before_split', 0))
    min_train_samples = int(getattr(filter_cfg, 'min_train_samples', 0))
    min_val_samples = int(getattr(filter_cfg, 'min_val_samples', 0))

    setup_reproducibility(config.experiment.seed)

    device = setup_device(config.experiment.device)

    for dir_name in ['models', 'plots', 'metrics', 'clustering']:
        ensure_directory(Path(config.outputs.paths.models).parent / dir_name)

    logger.info("\n" + "="*70)
    logger.info("STEP 1: TLE DATA DOWNLOAD")
    logger.info("="*70)

    downloader = TLEDownloader(
        config.data.spacetrack.username,
        config.data.spacetrack.password
    )

    tle_histories, refreshed_satellites = downloader.download_multiple_satellites(
        config.targets,  # 从文件加载的卫星目标
        start_date=config.data.tle.start_date,
        end_date=config.data.tle.end_date,
        skip_cached=config.data.tle.skip_cached,
        cache_dir=config.outputs.paths.data_raw,
        batch_size=config.data.tle.batch_size,
        request_delay_seconds=config.data.tle.request_delay_seconds
    )

    downloader.save_tle_data(
        tle_histories,
        config.outputs.paths.data_raw,
        satellite_names=refreshed_satellites
    )

    total_sats = len(tle_histories)
    empty_sats = [name for name, data in sorted(tle_histories.items()) if len(data) == 0]
    single_sats = [name for name, data in sorted(tle_histories.items()) if len(data) == 1]
    usable_sats = {
        name: data
        for name, data in sorted(tle_histories.items())
        if len(data) >= 2
    }

    logger.info(f"\n--- TLE Download Diagnostics ---")
    logger.info(f"Total satellites: {total_sats}")
    logger.info(f"Empty (0 TLEs): {len(empty_sats)}")
    logger.info(f"Single TLE (unusable for pairing): {len(single_sats)}")
    logger.info(f"Usable (>= 2 TLEs): {len(usable_sats)}")

    if len(usable_sats) == 0:
        logger.error("=" * 70)
        logger.error("FATAL: No satellites have enough TLE data (>= 2 TLEs)!")
        logger.error("=" * 70)
        logger.error("Possible causes:")
        logger.error("  1. Space-Track credentials are invalid (check config.yaml)")
        logger.error("  2. Cached data/raw/*.json files are empty (delete data/raw/ and re-run)")
        logger.error("  3. Network connection issues during download")
        logger.error("  4. Space-Track rate limiting (wait and retry)")
        if empty_sats:
            logger.error(f"  Examples of empty satellites: {empty_sats[:5]}")
        import os
        cache_dir = Path(config.outputs.paths.data_raw)
        if cache_dir.exists():
            json_files = list(cache_dir.glob("*.json"))
            if json_files:
                sample_file = json_files[0]
                size = os.path.getsize(sample_file)
                logger.error(f"  Cache dir has {len(json_files)} JSON files, sample size: {size} bytes")
                if size < 10:
                    logger.error(f"  ⚠ Cache files appear empty! Try: delete {cache_dir} and re-run")
        raise RuntimeError(
            f"No satellites have enough TLE data. "
            f"Check Space-Track credentials in config.yaml, or delete data/raw/ directory to force re-download."
        )

    tle_histories = usable_sats
    logger.info(f"Proceeding with {len(tle_histories)} satellites that have >= 2 TLEs")

    dtc_sats = [name for name in tle_histories if '[DTC]' in name]
    if dtc_sats:
        for name in dtc_sats:
            del tle_histories[name]
        logger.info(f"Filtered {len(dtc_sats)} [DTC] satellites → {len(tle_histories)} remain")

    logger.info("\n" + "="*70)
    logger.info("STEP 2: ORBITAL ELEMENTS COMPUTATION")
    logger.info("="*70)

    orbital_elements = {}

    for sat_name in sorted(tle_histories.keys()):
        tle_data = tle_histories[sat_name]
        if len(tle_data) > 0:
            latest_tle = tle_data[-1]
            tle_line1 = latest_tle['TLE_LINE1']
            tle_line2 = latest_tle['TLE_LINE2']
            epoch = parse_epoch(latest_tle['EPOCH'])

            satellite = Satrec.twoline2rv(tle_line1, tle_line2)
            r, v = propagate_satrec_to_time(satellite, epoch)

            if r is not None and v is not None:
                elem = compute_orbital_elements(r, v)
                orbital_elements[sat_name] = {
                    'a_mean': elem['a'],
                    'e_mean': elem['e'],
                    'i_mean': elem['i'],
                    'Omega_mean': elem['Omega'],  # 添加RAAN（升交点赤经）
                }

    logger.info(f"Orbital elements computed for {len(orbital_elements)} satellites.")

    logger.info("\n" + "="*70)
    logger.info("STEP 3: TLE ERROR DATASET CONSTRUCTION")
    logger.info("="*70)

    error_datasets = {}
    sat_items = tqdm(
        sorted(tle_histories.keys()),
        total=len(tle_histories),
        desc="Building error datasets",
        unit="sat",
        dynamic_ncols=True,
    )
    for sat_name in sat_items:
        tle_data = tle_histories[sat_name]
        X, y, timestamps = build_tle_error_dataset(
            tle_data,
            max_dt=config.data.tle.max_dt_days * 24 * 3600,
            seed=config.experiment.seed,
            orbit_samples_n=config.data.tle.orbit_samples_n,
            delta_n_threshold=config.data.tle.delta_n_threshold,
            iqr_k=config.data.tle.iqr_k
        )
        error_datasets[sat_name] = {'X': X, 'y': y, 'timestamps': timestamps}
    logger.info(f"Error datasets built for {len(error_datasets)} satellites.")

    nonempty_count = sum(1 for ds in error_datasets.values() if ds['X'].shape[0] > 0)
    empty_count = len(error_datasets) - nonempty_count
    logger.info(f"  Non-empty datasets: {nonempty_count}, Empty datasets: {empty_count}")
    if nonempty_count > 0:
        sample_counts = [ds['X'].shape[0] for ds in error_datasets.values() if ds['X'].shape[0] > 0]
        logger.info(f"  Sample counts: min={min(sample_counts)}, max={max(sample_counts)}, "
                    f"mean={sum(sample_counts)/len(sample_counts):.1f}")

    valid_error_datasets = {}
    skipped_invalid = []
    skipped_reasons = {}
    for sat_name in sorted(error_datasets.keys()):
        ds = error_datasets[sat_name]
        X, y = ds['X'], ds['y']
        if X.ndim == 2 and y.ndim == 2 and X.shape[0] == y.shape[0] and X.shape[0] > 0:
            valid_error_datasets[sat_name] = ds
        else:
            skipped_invalid.append(sat_name)
            reason = f"X.shape={X.shape}, y.shape={y.shape}, X.ndim={X.ndim}, y.ndim={y.ndim}"
            skipped_reasons[sat_name] = reason

    if skipped_invalid:
        logger.warning(f"Skipping {len(skipped_invalid)} satellites due to invalid/empty data.")
        if len(skipped_invalid) <= 10:
            for name in skipped_invalid:
                logger.warning(f"  {name}: {skipped_reasons[name]}")
        else:
            for name in skipped_invalid[:5]:
                logger.warning(f"  {name}: {skipped_reasons[name]}")
            logger.warning(f"  ... and {len(skipped_invalid) - 5} more")

    if not valid_error_datasets:
        logger.error("=" * 70)
        logger.error("FATAL: No valid error datasets after shape check!")
        logger.error("=" * 70)
        logger.error(f"Total satellites attempted: {len(error_datasets)}")
        logger.error(f"All {len(skipped_invalid)} failed shape validation.")
        logger.error("This usually means:")
        logger.error("  1. TLE data is too sparse (< 2 TLEs per satellite within max_dt_days)")
        logger.error("  2. delta_n_threshold is too strict (try increasing from 0.003 to 0.01)")
        logger.error("  3. max_dt_days is too small (try increasing from 3 to 7)")
        logger.error("  4. All TLE pairs were filtered by mean motion continuity check")
        logger.error(f"Current config: max_dt_days={config.data.tle.max_dt_days}, "
                    f"delta_n_threshold={config.data.tle.delta_n_threshold}, "
                    f"iqr_k={config.data.tle.iqr_k}")
        raise RuntimeError(
            f"No valid error datasets. Try: increase max_dt_days (current: {config.data.tle.max_dt_days}) "
            f"or delta_n_threshold (current: {config.data.tle.delta_n_threshold}), "
            f"or delete data/raw/ to force fresh TLE download."
        )

    error_datasets = valid_error_datasets

    if filter_enabled and min_sat_samples_before_split > 0:
        dropped_pre_split = [
            sat_name
            for sat_name in sorted(error_datasets.keys())
            if error_datasets[sat_name]['X'].shape[0] < min_sat_samples_before_split
        ]
        if dropped_pre_split:
            for sat_name in dropped_pre_split:
                del error_datasets[sat_name]
                if sat_name in orbital_elements:
                    del orbital_elements[sat_name]
                if sat_name in tle_histories:
                    del tle_histories[sat_name]
            logger.info(
                f"Pre-split small-sample filter: dropped {len(dropped_pre_split)} satellites "
                f"(min_sat_samples_before_split={min_sat_samples_before_split})"
            )
            if len(error_datasets) == 0:
                raise RuntimeError(
                    "No satellites remain after pre-split small-sample filtering. "
                    "Please lower data.filter.min_sat_samples_before_split."
                )

    iqr_k = config.data.tle.iqr_k
    iqr_metric = config.data.tle.iqr_metric

    if iqr_k > 0:
        logger.info("\n" + "-"*50)
        logger.info("STEP 3.1: CROSS-SATELLITE IQR FILTERING")
        logger.info("-"*50)

        error_datasets, removed_sats = cross_satellite_iqr_filter(
            error_datasets,
            k=iqr_k,
            metric=iqr_metric
        )

        if removed_sats:
            for sat_name in removed_sats:
                if sat_name in orbital_elements:
                    del orbital_elements[sat_name]
                if sat_name in tle_histories:
                    del tle_histories[sat_name]

            logger.info(f"After cross-satellite filter: {len(error_datasets)} satellites remain")
    else:
        logger.info("Cross-satellite IQR filter disabled (iqr_k <= 0)")

    all_t_stats = []
    for sat_name in sorted(error_datasets.keys()):
        sat_data = error_datasets[sat_name]
        if sat_data['y'].shape[0] > 0:
            t_vals = sat_data['y'][:, 1]
            t_mean = t_vals.mean()
            t_std = t_vals.std()
            t_error = abs(t_mean)
            t_median = np.median(t_vals)
            t_abs = np.abs(t_vals)
            t_p90 = np.percentile(t_abs, 90)
            t_p95 = np.percentile(t_abs, 95)
            n_tle = len(tle_histories[sat_name])
            all_t_stats.append(
                (
                    sat_name,
                    t_mean,
                    t_error,
                    t_std,
                    t_median,
                    t_p90,
                    t_p95,
                    n_tle,
                    sat_data['y'].shape[0],
                )
            )

    out_path = Path(config.outputs.paths.metrics) / "t_mean_outliers.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(
            "satellite,T_mean_km,T_error_km,T_std_km,T_median_km,T_abs_p90_km,T_abs_p95_km,n_tle,n_samples\n"
        )
        for (
            sat_name,
            t_mean,
            t_error,
            t_std,
            t_median,
            t_p90,
            t_p95,
            n_tle,
            n_samples,
        ) in all_t_stats:
            f.write(
                f"{sat_name},{t_mean:.6f},{t_error:.6f},{t_std:.6f},{t_median:.6f},{t_p90:.6f},{t_p95:.6f},{n_tle},{n_samples}\n"
            )
    logger.info(f"T_error stats saved to {out_path}")

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3.5: TIME-BASED TRAIN/VALIDATION SPLIT")
    logger.info("=" * 70)

    val_days = config.data.split.val_days
    train_error_datasets = {}
    val_error_datasets = {}
    for sat_name in sorted(error_datasets.keys()):
        ds = error_datasets[sat_name]
        X, y, timestamps = ds['X'], ds['y'], ds['timestamps']
        n = len(X)
        if n == 0:
            continue

        max_time = timestamps.max()
        val_threshold = max_time - np.timedelta64(val_days, 'D')

        train_mask = timestamps < val_threshold
        val_mask = timestamps >= val_threshold

        train_n = train_mask.sum()
        val_n = val_mask.sum()

        if val_n == 0:
            train_error_datasets[sat_name] = {'X': X[:-1], 'y': y[:-1]}
            val_error_datasets[sat_name] = {'X': X[-1:], 'y': y[-1:]}
            logger.debug(f"  {sat_name}: No samples in last {val_days} days, using last 1 sample as validation")
        elif train_n == 0:
            train_error_datasets[sat_name] = {'X': X[:1], 'y': y[:1]}
            val_error_datasets[sat_name] = {'X': X[1:], 'y': y[1:]}
            logger.debug(f"  {sat_name}: All samples in last {val_days} days, using first 1 sample as training")
        else:
            train_error_datasets[sat_name] = {'X': X[train_mask], 'y': y[train_mask]}
            val_error_datasets[sat_name] = {'X': X[val_mask], 'y': y[val_mask]}

    dropped_small_train_sats = []
    if filter_enabled and (min_train_samples > 0 or min_val_samples > 0):
        filtered_train = {}
        filtered_val = {}
        for sat_name in sorted(train_error_datasets.keys()):
            train_n = len(train_error_datasets[sat_name]['X'])
            val_n = len(val_error_datasets[sat_name]['X'])
            if train_n < min_train_samples or val_n < min_val_samples:
                dropped_small_train_sats.append(sat_name)
                continue
            filtered_train[sat_name] = train_error_datasets[sat_name]
            filtered_val[sat_name] = val_error_datasets[sat_name]

        train_error_datasets = filtered_train
        val_error_datasets = filtered_val

        for sat_name in dropped_small_train_sats:
            if sat_name in error_datasets:
                del error_datasets[sat_name]
            if sat_name in orbital_elements:
                del orbital_elements[sat_name]
            if sat_name in tle_histories:
                del tle_histories[sat_name]

        logger.info(f"Dropped small-sample train satellites: {len(dropped_small_train_sats)}")
        if len(train_error_datasets) == 0:
            raise RuntimeError(
                "No training satellites remain after train/val small-sample filtering. "
                "Please lower data.filter.min_train_samples or data.filter.min_val_samples."
            )

    total_samples = sum(len(ds['X']) for ds in error_datasets.values())
    train_samples = sum(len(ds['X']) for ds in train_error_datasets.values())
    val_samples = sum(len(ds['X']) for ds in val_error_datasets.values())
    logger.info(f"Total samples (train+val pools): {total_samples}")
    logger.info(f"Time-based split applied: last {val_days} days as validation")
    logger.info(f"  Train samples: {train_samples}, Val samples: {val_samples}")
    logger.info(f"  Actual validation ratio: {val_samples/total_samples*100 if total_samples>0 else 0:.2f}%")

    logger.info("\n" + "="*70)
    logger.info("STEP 4: SATELLITE CLUSTERING")
    logger.info("="*70)
    clustering_results, clustering_matrix = run_clustering_pipeline(
        orbital_elements, config, train_error_datasets=train_error_datasets
    )

    if config.outputs.save.clustering_results:
        save_dir = Path(config.outputs.paths.clustering)
        save_dir.mkdir(parents=True, exist_ok=True)
        clustering_results.to_csv(save_dir / 'clustering_results.csv', index=False)
        logger.info(f"Clustering results saved to {save_dir}")

    group_sat_lists = get_group_satellite_lists(clustering_results)

    logger.info("\n" + "="*70)
    logger.info("STEP 4.5: GROUP-WISE NORMALIZATION")
    logger.info("="*70)

    group_train_data = {}
    for group_id, sat_list in enumerate(group_sat_lists):
        X_list, y_list = [], []
        for sat_name in sat_list:
            if sat_name in train_error_datasets and len(train_error_datasets[sat_name]['X']) > 0:
                X_list.append(train_error_datasets[sat_name]['X'])
                y_list.append(train_error_datasets[sat_name]['y'])
        if X_list:
            group_train_data[group_id] = {
                'X': np.vstack(X_list),
                'y': np.vstack(y_list)
            }

    group_stats_list = []
    for group_id, data in group_train_data.items():
        group_stat = compute_group_statistics(data['X'], data['y'], group_id)
        group_stats_list.append(group_stat)

    norm_params = aggregate_group_statistics(group_stats_list)

    norm_params_path = Path(config.outputs.paths.models) / 'normalization_params.json'
    save_normalization_params(norm_params, str(norm_params_path))

    logger.info("\n" + "="*70)
    logger.info("STEP 5: DATASET PREPARATION")
    logger.info("="*70)

    sequence_length = config.model.common.sequence_length
    logger.info(f"  Sequence length: {sequence_length}")

    group_datasets = create_group_datasets(
        train_error_datasets, group_sat_lists,
        norm_params['X_mean'], norm_params['X_std'],
        norm_params['y_mean'], norm_params['y_std'],
        sequence_length=sequence_length
    )
    group_loaders = create_group_dataloaders(
        group_datasets,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.split.num_workers if hasattr(config.data.split, 'num_workers') else 0
    )

    group_val_datasets_dict = {}
    for group_id, sat_list in enumerate(group_sat_lists):
        val_data = {}
        for sat_name in sat_list:
            if sat_name in val_error_datasets and len(val_error_datasets[sat_name]['X']) > 0:
                val_data[sat_name] = val_error_datasets[sat_name]
        if val_data:
            group_val_datasets_dict[group_id] = val_data

    group_val_info = []
    group_val_loaders = []
    for group_id, val_data_dict in group_val_datasets_dict.items():
        val_dataset = TLEErrorDataset(
            val_data_dict,
            mean=norm_params['X_mean'],
            std=norm_params['X_std'],
            y_mean=norm_params['y_mean'],
            y_std=norm_params['y_std'],
            sequence_length=sequence_length
        )
        group_val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.split.num_workers if hasattr(config.data.split, 'num_workers') else 0,
            pin_memory=torch.cuda.is_available()
        )
        group_val_info.append((group_id, group_val_loader, norm_params))
        group_val_loaders.append(group_val_loader)

    logger.info("\n" + "="*70)
    logger.info("STEP 6: MODEL INITIALIZATION")
    logger.info("="*70)

    model_class, model_kwargs, model_type = build_model_spec(config)
    _, model_common, model_params = get_model_metadata(config)
    logger.info(f"Model type: {model_type}")

    collaborative_system = GroupedCollaborativeLearning(
        model_class=model_class,
        model_kwargs=model_kwargs,
        num_groups=len(group_loaders),
        device=device,
        model_type=model_type,
        model_common=model_common,
        model_params=model_params
    )

    collaborative_trainer = GroupedCollaborativeTrainer(
        collaborative_system, group_loaders, config, device,
        group_val_loaders=group_val_loaders if group_val_loaders else None,
        norm_params=norm_params
    )

    logger.info("\n" + "="*70)
    logger.info("STEP 7: GROUPED COLLABORATIVE TRAINING")
    logger.info("="*70)

    early_stopping = None
    if config.training.early_stopping.enabled:
        model_selection_cfg = getattr(config.training, 'model_selection', None)
        monitor_mode = str(getattr(model_selection_cfg, 'mode', 'min'))
        early_stop_min_delta = float(config.training.early_stopping.min_delta)
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=early_stop_min_delta,
            mode=monitor_mode
        )

    history = train_grouped_collaborative_model(
        collaborative_trainer,
        num_rounds=config.training.collaborative.num_rounds,
        early_stopping=early_stopping,
        save_dir=config.outputs.paths.models
    )

    best_model_path = Path(config.outputs.paths.models) / 'best_model.pth'
    if best_model_path.exists():
        default_monitor_metric = str(getattr(config.training.model_selection, 'metric', 'val_loss'))
        checkpoint = collaborative_system.load_shared_model(str(best_model_path))
        best_round = checkpoint.get('round', 'N/A')
        best_metric_name = checkpoint.get('monitor_metric', default_monitor_metric)
        best_metric_value = checkpoint.get('monitor_value', None)
        logger.info(f"\nLoading best model from iteration {best_round}")
        if best_metric_value is not None:
            logger.info(
                f"✓ Best model loaded (Best monitor metric [{best_metric_name}]: "
                f"{float(best_metric_value):.4f})"
            )
        else:
            logger.info("✓ Best model loaded")

    if config.outputs.save.plots:
        logger.info("\n" + "="*70)
        logger.info("STEP 8: GENERATING TRAINING/VALIDATION VISUALIZATIONS")
        logger.info("="*70)

        dt_by_split = {
            'train': np.concatenate(
                [ds['X'][:, 0] for ds in train_error_datasets.values() if len(ds['X']) > 0],
                axis=0,
            ) if train_error_datasets else np.array([], dtype=np.float64),
            'val': np.concatenate(
                [ds['X'][:, 0] for ds in val_error_datasets.values() if len(ds['X']) > 0],
                axis=0,
            ) if val_error_datasets else np.array([], dtype=np.float64),
        }
        plot_dt_distribution(
            dt_by_split,
            save_path=Path(config.outputs.paths.plots) / 'dt_distribution.png',
            dpi=config.outputs.visualization.dpi,
        )

        best_epoch_idx = early_stopping.best_epoch if early_stopping is not None else None
        plot_training_history(
            history,
            save_path=Path(config.outputs.paths.plots) / 'training_history.png',
            dpi=config.outputs.visualization.dpi,
            best_epoch=best_epoch_idx
        )


    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: {Path(config.outputs.paths.models).parent}")
    logger.info(f"  Models: {config.outputs.paths.models}")
    logger.info(f"  Plots: {config.outputs.paths.plots}")
    logger.info(f"  Metrics: {config.outputs.paths.metrics}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
