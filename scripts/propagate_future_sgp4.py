#!/usr/bin/env python3
"""Propagate latest raw TLE histories with plain SGP4.

This script mirrors the latest-anchor selection used by
scripts/predict_future_corrections.py, but does not load the correction model.
It propagates each satellite's latest TLE to future horizon(s), writes the
plain SGP4 state, and logs total runtime.
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import parse_epoch, propagate_satrec_to_time  # noqa: E402
from utils import setup_logging  # noqa: E402


logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "satellite",
    "norad_id",
    "source_json",
    "target_time_utc",
    "anchor_epoch_utc",
    "horizon_hours",
    "r_sgp4_x",
    "r_sgp4_y",
    "r_sgp4_z",
    "v_sgp4_x",
    "v_sgp4_y",
    "v_sgp4_z",
    "status",
    "error_message",
]


def _utc_string(dt) -> str:
    return dt.isoformat(timespec="seconds") + "Z"


def _normalize_satellite_key(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())


def _history_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_tle"):
        stem = stem[:-4]
    return stem


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


def _empty_row(
    satellite: str,
    norad_id: str,
    source_json: str,
    target_time,
    anchor_epoch,
    horizon: float,
    status: str,
    error_message: str,
) -> dict:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "satellite": satellite,
            "norad_id": norad_id,
            "source_json": source_json,
            "target_time_utc": _utc_string(target_time) if target_time else "",
            "anchor_epoch_utc": _utc_string(anchor_epoch) if anchor_epoch else "",
            "horizon_hours": horizon,
            "status": status,
            "error_message": error_message,
        }
    )
    return row


def _propagate_history(path: Path, records: Sequence[dict], horizons: Sequence[float]) -> Iterable[dict]:
    if not records:
        for horizon in horizons:
            yield _empty_row(
                _history_name_from_path(path),
                "",
                str(path),
                None,
                None,
                horizon,
                "failed",
                "No valid TLE records",
            )
        return

    anchor = records[-1]
    anchor_epoch = parse_epoch(anchor["EPOCH"])
    satellite = str(anchor.get("OBJECT_NAME") or _history_name_from_path(path))
    norad_id = str(anchor.get("NORAD_CAT_ID", ""))

    try:
        sat = Satrec.twoline2rv(anchor["TLE_LINE1"], anchor["TLE_LINE2"])
    except Exception as exc:
        for horizon in horizons:
            target_time = anchor_epoch + timedelta(hours=float(horizon))
            yield _empty_row(
                satellite,
                norad_id,
                str(path),
                target_time,
                anchor_epoch,
                horizon,
                "failed",
                f"Failed to create Satrec: {exc}",
            )
        return

    for horizon in horizons:
        target_time = anchor_epoch + timedelta(hours=float(horizon))
        try:
            r_sgp4, v_sgp4 = propagate_satrec_to_time(sat, target_time)
            if r_sgp4 is None or v_sgp4 is None:
                raise RuntimeError("SGP4 propagation failed")

            row = _empty_row(
                satellite,
                norad_id,
                str(path),
                target_time,
                anchor_epoch,
                horizon,
                "ok",
                "",
            )
            row.update(
                {
                    "r_sgp4_x": r_sgp4[0],
                    "r_sgp4_y": r_sgp4[1],
                    "r_sgp4_z": r_sgp4[2],
                    "v_sgp4_x": v_sgp4[0],
                    "v_sgp4_y": v_sgp4[1],
                    "v_sgp4_z": v_sgp4[2],
                }
            )
            yield row
        except Exception as exc:
            yield _empty_row(
                satellite,
                norad_id,
                str(path),
                target_time,
                anchor_epoch,
                horizon,
                "failed",
                str(exc),
            )



def _chunk_paths(paths: Sequence[Path], chunk_size: int) -> List[List[str]]:
    return [
        [str(path) for path in paths[start : start + chunk_size]]
        for start in range(0, len(paths), chunk_size)
    ]


def _propagate_chunk_worker(task: Tuple[int, List[str], dict]) -> Tuple[int, List[dict]]:
    chunk_idx, path_strings, args_dict = task
    args = argparse.Namespace(**args_dict)
    horizons = _parse_horizons(args.horizons)
    rows: List[dict] = []

    for path_string in path_strings:
        path = Path(path_string)
        try:
            records = _load_tle_history(path, args.satellite if args.tle_history else None)
            if not records:
                raise ValueError("No matching TLE records")
            rows.extend(_propagate_history(path, records, horizons))
        except Exception as exc:
            satellite = _history_name_from_path(path)
            for horizon in horizons:
                rows.append(_empty_row(satellite, "", str(path), None, None, horizon, "failed", str(exc)))
    return chunk_idx, rows


def _iter_sgp4_rows_parallel(paths: Sequence[Path], args) -> Iterable[dict]:
    chunk_size = max(1, int(args.chunk_size))
    chunks = _chunk_paths(paths, chunk_size)
    tasks = [(idx, chunk, vars(args).copy()) for idx, chunk in enumerate(chunks)]
    total = len(paths)
    completed = 0

    logger.info("Using %d worker processes, chunk_size=%d", args.num_workers, chunk_size)
    with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
        for chunk_idx, rows in executor.map(_propagate_chunk_worker, tasks, chunksize=1):
            completed += len(chunks[chunk_idx])
            logger.info("Processed %d/%d TLE histories", min(completed, total), total)
            yield from rows


def _iter_sgp4_rows(paths: Sequence[Path], args) -> Iterable[dict]:
    horizons = _parse_horizons(args.horizons)
    total = len(paths)

    for idx, path in enumerate(paths, start=1):
        try:
            records = _load_tle_history(path, args.satellite if args.tle_history else None)
            if not records:
                raise ValueError("No matching TLE records")
            if idx == 1 or idx % 1000 == 0:
                logger.info("Processing %d/%d: %s", idx, total, path.name)
            yield from _propagate_history(path, records, horizons)
        except Exception as exc:
            satellite = _history_name_from_path(path)
            logger.warning("Failed to process %s: %s", path, exc)
            for horizon in horizons:
                yield _empty_row(satellite, "", str(path), None, None, horizon, "failed", str(exc))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propagate latest raw TLE histories with plain SGP4.")
    parser.add_argument("--raw-dir", type=str, default=str(ROOT / "data" / "raw"))
    parser.add_argument("--tle-history", type=str, default=None, help="Single TLE history JSON file")
    parser.add_argument("--satellite", type=str, default=None, help="Satellite name or NORAD ID")
    parser.add_argument("--anchor-policy", choices=["latest"], default="latest")
    parser.add_argument("--horizons", type=str, default="12")
    parser.add_argument("--output-csv", type=str, default=str(ROOT / "future_sgp4_12h.csv"))
    parser.add_argument("--num-workers", type=int, default=1, help="Number of CPU worker processes for satellite-level parallel propagation")
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of TLE histories sent to each worker task")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of TLE histories for smoke tests")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    setup_logging(args.log_level, console=True)

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
    logger.info("Propagating %d TLE histories to %s", len(paths), output_csv)
    if args.num_workers > 1:
        row_iter = _iter_sgp4_rows_parallel(paths, args)
    else:
        row_iter = _iter_sgp4_rows(paths, args)
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
