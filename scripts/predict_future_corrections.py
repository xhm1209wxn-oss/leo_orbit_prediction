#!/usr/bin/env python3
"""Predict future RTN position-error corrections from the latest TLE.

This script implements the v1 future-correction inference flow:

- select the latest TLE as the anchor,
- build a 19-step normalized historical context with the supplied configuration,
- append one normalized future feature for each horizon,
- predict RTN error and subtract it from the anchor SGP4 future position.
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from models import load_model  # noqa: E402
from preprocessing import (  # noqa: E402
    build_tle_error_dataset,
    compute_argument_of_latitude_sin_cos,
    extract_ballistic_coefficient,
    load_normalization_params,
    parse_epoch,
    propagate_satrec_to_time,
)
from utils import setup_device, setup_logging  # noqa: E402


logger = logging.getLogger(__name__)

_WORKER_ARGS = None
_WORKER_DEVICE = None
_WORKER_MODEL_CACHE = None

CSV_FIELDS = [
    "satellite",
    "norad_id",
    "model_run",
    "target_time_utc",
    "anchor_epoch_utc",
    "horizon_hours",
    "r_sgp4_x",
    "r_sgp4_y",
    "r_sgp4_z",
    "v_sgp4_x",
    "v_sgp4_y",
    "v_sgp4_z",
    "delta_R",
    "delta_T",
    "delta_N",
    "delta_teme_x",
    "delta_teme_y",
    "delta_teme_z",
    "r_corrected_teme_x",
    "r_corrected_teme_y",
    "r_corrected_teme_z",
    "correction_sign",
    "frame_used",
    "sequence_length",
    "history_context_len",
    "padding_len",
    "status",
    "error_message",
]


@dataclass
class PredictorBundle:
    run_dir: Path
    config: object
    model: torch.nn.Module
    norm_params: Dict[str, np.ndarray]
    sequence_length: int
    input_size: int
    output_size: int


def _utc_string(dt) -> str:
    return dt.isoformat(timespec="seconds") + "Z"


def _load_config(config_path: Path):
    return load_config(str(config_path))


def _parse_horizons(value: str) -> List[float]:
    horizons: List[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        horizon = float(part)
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        horizons.append(horizon)
    if not horizons:
        raise ValueError("At least one horizon is required")
    return horizons


def _normalize_satellite_key(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())


def _history_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_tle"):
        stem = stem[:-4]
    return stem


def _load_tle_history(path: Path, satellite_filter: Optional[str] = None) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"TLE history must be a JSON list: {path}")

    records = [rec for rec in data if "TLE_LINE1" in rec and "TLE_LINE2" in rec and "EPOCH" in rec]
    if satellite_filter:
        key = _normalize_satellite_key(satellite_filter)
        records = [
            rec
            for rec in records
            if _normalize_satellite_key(str(rec.get("OBJECT_NAME", ""))) == key
            or str(rec.get("NORAD_CAT_ID", "")).strip() == satellite_filter.strip()
        ]

    records.sort(key=lambda rec: parse_epoch(rec["EPOCH"]))
    return records


def _find_history_paths(raw_dir: Path, satellite: Optional[str]) -> List[Path]:
    all_paths = sorted(raw_dir.glob("*_tle.json"))
    if satellite is None:
        return all_paths

    direct = raw_dir / f"{satellite}_tle.json"
    if direct.exists():
        return [direct]

    target_key = _normalize_satellite_key(satellite)
    name_matches = [
        path
        for path in all_paths
        if _normalize_satellite_key(_history_name_from_path(path)) == target_key
    ]
    if name_matches:
        return name_matches

    matches = []
    for path in all_paths:
        try:
            records = _load_tle_history(path)
        except Exception:
            continue
        if not records:
            continue
        latest = records[-1]
        if (
            _normalize_satellite_key(str(latest.get("OBJECT_NAME", ""))) == target_key
            or str(latest.get("NORAD_CAT_ID", "")).strip() == satellite.strip()
        ):
            matches.append(path)
    return matches


def _select_run_dir(args) -> Path:
    """Return the one explicitly selected shared-model run directory."""
    return Path(args.run_dir).resolve()


def _load_checkpoint_metadata(model_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a metadata dictionary: {model_path}")
    return checkpoint


def _load_predictor(run_dir: Path, config_path: Path, device: torch.device) -> PredictorBundle:
    model_path = run_dir / "models" / "best_model.pth"
    norm_path = run_dir / "models" / "normalization_params.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization_params.json: {norm_path}")

    config = _load_config(config_path)
    checkpoint = _load_checkpoint_metadata(model_path, device)

    common = getattr(config.model, "common")
    input_size = int(common.input_size)
    output_size = int(common.output_size)
    sequence_length = int(common.sequence_length)

    ckpt_common = checkpoint.get("model_common", {})
    if ckpt_common:
        expected = {
            "input_size": input_size,
            "output_size": output_size,
            "sequence_length": sequence_length,
        }
        for key, expected_value in expected.items():
            actual_value = int(ckpt_common.get(key, expected_value))
            if actual_value != expected_value:
                raise ValueError(
                    f"Checkpoint {key}={actual_value} does not match configuration {key}={expected_value}"
                )

    model_kwargs = checkpoint.get("model_kwargs", {})
    if "max_seq_len" in model_kwargs and int(model_kwargs["max_seq_len"]) != sequence_length:
        raise ValueError(
            f"Checkpoint max_seq_len={model_kwargs['max_seq_len']} does not match sequence_length={sequence_length}"
        )

    if input_size != 19 or output_size != 3:
        raise ValueError(f"Expected input_size=19 and output_size=3, got {input_size}/{output_size}")
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")

    model = load_model(str(model_path), config, device)
    norm_params = load_normalization_params(str(norm_path))

    logger.info(
        "Loaded predictor: run=%s seq_len=%d input=%d output=%d",
        run_dir,
        sequence_length,
        input_size,
        output_size,
    )
    return PredictorBundle(
        run_dir=run_dir,
        config=config,
        model=model,
        norm_params=norm_params,
        sequence_length=sequence_length,
        input_size=input_size,
        output_size=output_size,
    )


def _mean_motion_rad_s(rec: dict) -> float:
    try:
        n_rev_day = float(rec["TLE_LINE2"].split()[7][:11])
    except (KeyError, IndexError, ValueError):
        n_rev_day = float(rec.get("MEAN_MOTION", "nan"))
    if not np.isfinite(n_rev_day) or n_rev_day <= 0:
        raise ValueError(f"Invalid mean motion for {rec.get('OBJECT_NAME', 'unknown')}")
    return float(n_rev_day * 2.0 * np.pi / 86400.0)


def _build_future_feature(
    sat: Satrec,
    anchor_rec: dict,
    anchor_epoch,
    r_anchor: np.ndarray,
    v_anchor: np.ndarray,
    n_rad_s: float,
    bstar: float,
    target_time,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_future, v_future = propagate_satrec_to_time(sat, target_time)
    if r_future is None:
        raise RuntimeError("SGP4 failed at target time")

    dt = (target_time - anchor_epoch).total_seconds()
    if dt <= 0:
        raise ValueError(f"Target time must be after anchor epoch, got dt={dt}")

    phase = float(n_rad_s * dt)
    sin_u, cos_u = compute_argument_of_latitude_sin_cos(r_future, v_future)
    height = float(np.linalg.norm(r_future) - 6378.137)

    feature = np.hstack(
        [
            [dt],
            [float(np.sin(phase))],
            [float(np.cos(phase))],
            r_anchor,
            v_anchor,
            r_future,
            v_future,
            [sin_u],
            [cos_u],
            [height],
            [bstar],
        ]
    ).astype(np.float32)
    if feature.shape != (19,):
        raise RuntimeError(f"Future feature has unexpected shape: {feature.shape}")
    if not np.isfinite(feature).all():
        raise RuntimeError("Future feature contains non-finite values")
    return feature, r_future.astype(np.float64), v_future.astype(np.float64)


def _build_context(records: Sequence[dict], anchor_epoch, bundle: PredictorBundle) -> Tuple[np.ndarray, int, int]:
    tle_cfg = bundle.config.data.tle
    history_records = [rec for rec in records if parse_epoch(rec["EPOCH"]) <= anchor_epoch]
    X_hist_raw, _, timestamps = build_tle_error_dataset(
        history_records,
        max_dt=float(tle_cfg.max_dt_days) * 24.0 * 3600.0,
        orbit_samples_n=int(tle_cfg.orbit_samples_n),
        delta_n_threshold=float(tle_cfg.delta_n_threshold),
        iqr_k=float(tle_cfg.iqr_k),
    )

    if len(timestamps) > 1:
        sort_idx = np.argsort(timestamps, kind="mergesort")
        X_hist_raw = X_hist_raw[sort_idx]
        timestamps = timestamps[sort_idx]

    context_len = bundle.sequence_length - 1
    if len(timestamps) > 0:
        mask = timestamps <= np.datetime64(anchor_epoch)
        X_context_raw = X_hist_raw[mask][-context_len:]
    else:
        X_context_raw = np.empty((0, bundle.input_size), dtype=np.float32)

    X_mean = bundle.norm_params["X_mean"]
    X_std = bundle.norm_params["X_std"]
    X_context_norm = ((X_context_raw - X_mean) / X_std).astype(np.float32)

    real_context_len = len(X_context_norm)
    padding_len = max(0, context_len - real_context_len)
    if padding_len > 0:
        X_context_norm = np.vstack(
            [
                np.zeros((padding_len, bundle.input_size), dtype=np.float32),
                X_context_norm,
            ]
        )
    return X_context_norm, real_context_len, padding_len


def _rtn_to_teme(delta_rtn: np.ndarray, r_sgp4: np.ndarray, v_sgp4: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_sgp4)
    if r_norm <= 0:
        raise RuntimeError("Cannot build RTN frame from zero position vector")
    r_hat = r_sgp4 / r_norm

    h_vec = np.cross(r_sgp4, v_sgp4)
    h_norm = np.linalg.norm(h_vec)
    if h_norm <= 0:
        raise RuntimeError("Cannot build RTN frame from degenerate angular momentum")
    n_hat = h_vec / h_norm
    t_hat = np.cross(n_hat, r_hat)

    return (
        float(delta_rtn[0]) * r_hat
        + float(delta_rtn[1]) * t_hat
        + float(delta_rtn[2]) * n_hat
    )


def _empty_row(
    satellite: str,
    norad_id: str,
    model_run: str,
    target_time,
    anchor_epoch,
    horizon: float,
    status: str,
    error_message: str,
    sequence_length: int = "",
    history_context_len: int = "",
    padding_len: int = "",
) -> dict:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "satellite": satellite,
            "norad_id": norad_id,
            "model_run": model_run,
            "target_time_utc": _utc_string(target_time) if target_time else "",
            "anchor_epoch_utc": _utc_string(anchor_epoch) if anchor_epoch else "",
            "horizon_hours": horizon,
            "correction_sign": -1,
            "frame_used": "sgp4_rtn",
            "sequence_length": sequence_length,
            "history_context_len": history_context_len,
            "padding_len": padding_len,
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def _predict_history(
    path: Path,
    records: Sequence[dict],
    bundle: PredictorBundle,
    horizons: Sequence[float],
    device: torch.device,
) -> Iterable[dict]:
    if not records:
        yield _empty_row(
            _history_name_from_path(path),
            "",
            str(bundle.run_dir),
            None,
            None,
            "",
            "failed",
            "No valid TLE records",
            bundle.sequence_length,
        )
        return

    anchor = records[-1]
    anchor_epoch = parse_epoch(anchor["EPOCH"])
    satellite = str(anchor.get("OBJECT_NAME") or _history_name_from_path(path))
    norad_id = str(anchor.get("NORAD_CAT_ID", ""))

    try:
        X_context_norm, real_context_len, padding_len = _build_context(records, anchor_epoch, bundle)
    except Exception as exc:
        for horizon in horizons:
            target_time = anchor_epoch + timedelta(hours=float(horizon))
            yield _empty_row(
                satellite,
                norad_id,
                str(bundle.run_dir),
                target_time,
                anchor_epoch,
                horizon,
                "failed",
                f"Failed to build context: {exc}",
                bundle.sequence_length,
            )
        return

    X_mean = bundle.norm_params["X_mean"]
    X_std = bundle.norm_params["X_std"]
    y_mean = bundle.norm_params["y_mean"]
    y_std = bundle.norm_params["y_std"]
    context_len = bundle.sequence_length - 1

    try:
        sat = Satrec.twoline2rv(anchor["TLE_LINE1"], anchor["TLE_LINE2"])
        r_anchor, v_anchor = propagate_satrec_to_time(sat, anchor_epoch)
        if r_anchor is None:
            raise RuntimeError("SGP4 failed at anchor epoch")
        n_rad_s = _mean_motion_rad_s(anchor)
        bstar = extract_ballistic_coefficient(anchor["TLE_LINE1"])
    except Exception as exc:
        for horizon in horizons:
            target_time = anchor_epoch + timedelta(hours=float(horizon))
            yield _empty_row(
                satellite,
                norad_id,
                str(bundle.run_dir),
                target_time,
                anchor_epoch,
                horizon,
                "failed",
                f"Failed to initialize SGP4 anchor: {exc}",
                bundle.sequence_length,
                real_context_len,
                padding_len,
            )
        return

    result_rows: List[Optional[dict]] = [None] * len(horizons)
    valid_items = []

    for horizon_idx, horizon in enumerate(horizons):
        target_time = anchor_epoch + timedelta(hours=float(horizon))
        try:
            x_future, r_sgp4, v_sgp4 = _build_future_feature(
                sat,
                anchor,
                anchor_epoch,
                r_anchor,
                v_anchor,
                n_rad_s,
                bstar,
                target_time,
            )
            x_future_norm = ((x_future - X_mean) / X_std).astype(np.float32)
            X_window_norm = np.vstack(
                [
                    X_context_norm[-context_len:],
                    x_future_norm.reshape(1, bundle.input_size),
                ]
            ).astype(np.float32)
            if X_window_norm.shape != (bundle.sequence_length, bundle.input_size):
                raise RuntimeError(f"Unexpected input shape: {X_window_norm.shape}")
            if not np.isfinite(X_window_norm).all():
                raise RuntimeError("Input window contains non-finite values")

            valid_items.append((horizon_idx, horizon, target_time, r_sgp4, v_sgp4, X_window_norm))
        except Exception as exc:
            result_rows[horizon_idx] = _empty_row(
                satellite,
                norad_id,
                str(bundle.run_dir),
                target_time,
                anchor_epoch,
                horizon,
                "failed",
                str(exc),
                bundle.sequence_length,
                real_context_len,
                padding_len,
            )

    if valid_items:
        try:
            X_batch_norm = np.stack([item[5] for item in valid_items], axis=0).astype(np.float32)
            x_tensor = torch.from_numpy(X_batch_norm).float().to(device)
            with torch.no_grad():
                pred_norm_batch = bundle.model(x_tensor).detach().cpu().numpy()
            delta_rtn_batch = pred_norm_batch * y_std + y_mean
        except Exception as exc:
            for horizon_idx, horizon, target_time, _r_sgp4, _v_sgp4, _X_window_norm in valid_items:
                result_rows[horizon_idx] = _empty_row(
                    satellite,
                    norad_id,
                    str(bundle.run_dir),
                    target_time,
                    anchor_epoch,
                    horizon,
                    "failed",
                    f"Model batch inference failed: {exc}",
                    bundle.sequence_length,
                    real_context_len,
                    padding_len,
                )
        else:
            for batch_idx, (horizon_idx, horizon, target_time, r_sgp4, v_sgp4, _X_window_norm) in enumerate(valid_items):
                delta_rtn = delta_rtn_batch[batch_idx]
                delta_teme = _rtn_to_teme(delta_rtn, r_sgp4, v_sgp4)
                r_corrected = r_sgp4 - delta_teme

                row = _empty_row(
                    satellite,
                    norad_id,
                    str(bundle.run_dir),
                    target_time,
                    anchor_epoch,
                    horizon,
                    "ok",
                    "",
                    bundle.sequence_length,
                    real_context_len,
                    padding_len,
                )
                row.update(
                    {
                        "r_sgp4_x": r_sgp4[0],
                        "r_sgp4_y": r_sgp4[1],
                        "r_sgp4_z": r_sgp4[2],
                        "v_sgp4_x": v_sgp4[0],
                        "v_sgp4_y": v_sgp4[1],
                        "v_sgp4_z": v_sgp4[2],
                        "delta_R": delta_rtn[0],
                        "delta_T": delta_rtn[1],
                        "delta_N": delta_rtn[2],
                        "delta_teme_x": delta_teme[0],
                        "delta_teme_y": delta_teme[1],
                        "delta_teme_z": delta_teme[2],
                        "r_corrected_teme_x": r_corrected[0],
                        "r_corrected_teme_y": r_corrected[1],
                        "r_corrected_teme_z": r_corrected[2],
                    }
                )
                result_rows[horizon_idx] = row

    for row in result_rows:
        if row is not None:
            yield row
        else:
            yield _empty_row(
                satellite,
                norad_id,
                str(bundle.run_dir),
                None,
                anchor_epoch,
                "",
                "failed",
                "Internal error: missing batch result row",
                bundle.sequence_length,
                real_context_len,
                padding_len,
            )


def _format_value(value):
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.12g}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_rows(output_csv: Path, rows: Iterable[dict]) -> Tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_ok = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            formatted = {key: _format_value(row.get(key, "")) for key in CSV_FIELDS}
            writer.writerow(formatted)
            n_rows += 1
            if row.get("status") == "ok":
                n_ok += 1
    return n_rows, n_ok



def _chunk_paths(paths: Sequence[Path], chunk_size: int) -> List[List[str]]:
    return [
        [str(path) for path in paths[start : start + chunk_size]]
        for start in range(0, len(paths), chunk_size)
    ]


def _init_prediction_worker(args_dict: dict, device_str: str, torch_threads: int) -> None:
    global _WORKER_ARGS, _WORKER_DEVICE, _WORKER_MODEL_CACHE
    torch_threads = max(1, int(torch_threads))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass
    _WORKER_ARGS = argparse.Namespace(**args_dict)
    _WORKER_DEVICE = torch.device(device_str)
    _WORKER_MODEL_CACHE = {}


def _predict_chunk_worker(task: Tuple[int, List[str]]) -> Tuple[int, List[dict]]:
    if _WORKER_ARGS is None or _WORKER_DEVICE is None or _WORKER_MODEL_CACHE is None:
        raise RuntimeError("Prediction worker was not initialized")

    chunk_idx, path_strings = task
    args = _WORKER_ARGS
    device = _WORKER_DEVICE
    model_cache = _WORKER_MODEL_CACHE
    horizons = _parse_horizons(args.horizons)
    rows: List[dict] = []

    for path_string in path_strings:
        path = Path(path_string)
        try:
            records = _load_tle_history(path, args.satellite if args.tle_history else None)
            if not records:
                raise ValueError("No matching TLE records")

            run_dir = _select_run_dir(args)
            if run_dir not in model_cache:
                model_cache[run_dir] = _load_predictor(
                    run_dir, Path(args.config).resolve(), device
                )
            bundle = model_cache[run_dir]
            rows.extend(_predict_history(path, records, bundle, horizons, device))
        except Exception as exc:
            satellite = _history_name_from_path(path)
            for horizon in horizons:
                rows.append(
                    _empty_row(
                        satellite,
                        "",
                        str(args.run_dir),
                        None,
                        None,
                        horizon,
                        "failed",
                        str(exc),
                    )
                )
    return chunk_idx, rows


def _iter_prediction_rows_parallel(paths: Sequence[Path], args) -> Iterable[dict]:
    chunk_size = max(1, int(args.chunk_size))
    chunks = _chunk_paths(paths, chunk_size)
    tasks = list(enumerate(chunks))
    total = len(paths)
    completed = 0

    worker_args = vars(args).copy()
    logger.info(
        "Using %d worker processes, chunk_size=%d, worker_torch_threads=%d",
        args.num_workers,
        chunk_size,
        args.worker_torch_threads,
    )

    with ProcessPoolExecutor(
        max_workers=int(args.num_workers),
        initializer=_init_prediction_worker,
        initargs=(worker_args, "cpu", int(args.worker_torch_threads)),
    ) as executor:
        for chunk_idx, rows in executor.map(_predict_chunk_worker, tasks, chunksize=1):
            completed += len(chunks[chunk_idx])
            logger.info("Processed %d/%d TLE histories", min(completed, total), total)
            yield from rows


def _iter_prediction_rows(paths: Sequence[Path], args, device: torch.device) -> Iterable[dict]:
    horizons = _parse_horizons(args.horizons)
    model_cache: Dict[Path, PredictorBundle] = {}
    total = len(paths)

    for idx, path in enumerate(paths, start=1):
        try:
            records = _load_tle_history(path, args.satellite if args.tle_history else None)
            if not records:
                raise ValueError("No matching TLE records")

            run_dir = _select_run_dir(args)
            if run_dir not in model_cache:
                model_cache[run_dir] = _load_predictor(
                    run_dir, Path(args.config).resolve(), device
                )
            bundle = model_cache[run_dir]

            if idx == 1 or idx % 100 == 0:
                logger.info("Processing %d/%d: %s", idx, total, path.name)
            yield from _predict_history(path, records, bundle, horizons, device)
        except Exception as exc:
            satellite = _history_name_from_path(path)
            logger.warning("Failed to process %s: %s", path, exc)
            for horizon in horizons:
                yield _empty_row(
                    satellite,
                    "",
                    str(args.run_dir),
                    None,
                    None,
                    horizon,
                    "failed",
                    str(exc),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict future RTN correction estimates from latest TLE histories."
    )
    parser.add_argument("--raw-dir", type=str, default=str(ROOT / "data" / "raw"))
    parser.add_argument("--tle-history", type=str, default=None, help="Single TLE history JSON file")
    parser.add_argument("--satellite", type=str, default=None, help="Satellite name or NORAD ID")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Single shared-model run directory used for every input satellite",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config.yaml"),
        help="Configuration file matching the trained model",
    )
    parser.add_argument("--anchor-policy", choices=["latest"], default="latest")
    parser.add_argument("--horizons", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--output-csv", type=str, default=str(ROOT / "future_corrections.csv"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of CPU worker processes for satellite-level parallel prediction")
    parser.add_argument("--chunk-size", type=int, default=64, help="Number of TLE histories sent to each worker task")
    parser.add_argument("--worker-torch-threads", type=int, default=1, help="PyTorch intra/inter-op threads per worker process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of TLE histories for smoke tests")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    setup_logging(args.log_level, console=True)
    if args.num_workers > 1:
        if args.device.lower() != "cpu":
            raise ValueError("--num-workers > 1 currently supports --device cpu only")
        device = torch.device("cpu")
        logger.info("Using CPU with multiprocessing")
    else:
        device = setup_device(args.device)

    if args.tle_history:
        paths = [Path(args.tle_history).resolve()]
    else:
        raw_dir = Path(args.raw_dir).resolve()
        paths = _find_history_paths(raw_dir, args.satellite)

    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError("No TLE history files matched the request")

    output_csv = Path(args.output_csv).resolve()
    logger.info("Predicting %d TLE histories to %s", len(paths), output_csv)
    if args.num_workers > 1:
        row_iter = _iter_prediction_rows_parallel(paths, args)
    else:
        row_iter = _iter_prediction_rows(paths, args, device)
    n_rows, n_ok = _write_rows(output_csv, row_iter)
    elapsed_seconds = time.perf_counter() - started_at
    rows_per_second = n_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
    logger.info(
        "Finished: rows=%d ok=%d failed=%d elapsed_seconds=%.3f rows_per_second=%.2f output=%s",
        n_rows,
        n_ok,
        n_rows - n_ok,
        elapsed_seconds,
        rows_per_second,
        output_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
