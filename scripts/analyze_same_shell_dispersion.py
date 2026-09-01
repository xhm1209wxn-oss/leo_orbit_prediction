#!/usr/bin/env python3
"""
Analyze cross-satellite dispersion of along-track error statistics inside a
near-identical orbital shell from a plain two-line TLE text file.

The script:
1. Parses a `.txt` file containing repeated TLE line1/line2 pairs.
2. Groups records by NORAD catalog ID and rebuilds each satellite's T-error
   samples using the repository's existing preprocessing logic.
3. Computes per-satellite T-error statistics and mean orbital shell parameters.
4. Searches for same-shell satellite groups with both sufficient population and
   strong divergence in T-error statistics.
5. Writes machine-readable CSV summaries and a paper-friendly figure.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing import build_tle_error_dataset, extract_ballistic_coefficient  # noqa: E402


MU_EARTH_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze same-shell dispersion from a two-line TLE txt file."
    )
    parser.add_argument(
        "--tle-file",
        type=Path,
        default=REPO_ROOT / "data" / "tle10011231.txt",
        help="Input TLE txt file containing repeated line1/line2 pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "same_shell_dispersion",
        help="Directory for CSV, figure, and summary outputs.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-10-01",
        help="Inclusive UTC start date filter in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="Inclusive UTC end date filter in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--max-dt-days",
        type=float,
        default=3.0,
        help="Maximum TLE pair spacing in days for sample construction.",
    )
    parser.add_argument(
        "--orbit-samples-n",
        type=int,
        default=2,
        help="Uniform orbit-window segments (n -> n+1 samples).",
    )
    parser.add_argument(
        "--delta-n-threshold",
        type=float,
        default=0.003,
        help="Mean-motion continuity threshold in rev/day.",
    )
    parser.add_argument(
        "--iqr-k",
        type=float,
        default=2.5,
        help="Per-satellite IQR outlier filter coefficient.",
    )
    parser.add_argument(
        "--min-tles",
        type=int,
        default=30,
        help="Minimum number of TLE records per satellite to analyze.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=120,
        help="Minimum number of rebuilt T-error samples per satellite.",
    )
    parser.add_argument(
        "--alt-window-km",
        type=float,
        default=5.0,
        help="Same-shell altitude half-width in km.",
    )
    parser.add_argument(
        "--inc-window-deg",
        type=float,
        default=0.05,
        help="Same-shell inclination half-width in deg.",
    )
    parser.add_argument(
        "--ecc-window",
        type=float,
        default=5.0e-4,
        help="Same-shell eccentricity half-width.",
    )
    parser.add_argument(
        "--min-shell-size",
        type=int,
        default=8,
        help="Minimum satellite count for a valid shell.",
    )
    parser.add_argument(
        "--top-shells",
        type=int,
        default=20,
        help="Number of ranked shell candidates to save.",
    )
    parser.add_argument(
        "--center-norad",
        type=str,
        default=None,
        help="Optional NORAD catalog ID to force shell selection around one satellite.",
    )
    parser.add_argument(
        "--max-satellites",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Random seed passed into the existing preprocessing function.",
    )
    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def parse_tle_epoch(line1: str) -> datetime:
    raw = line1[18:32].strip()
    if len(raw) < 5:
        raise ValueError(f"Invalid TLE epoch field: {raw!r}")

    year_2d = int(raw[:2])
    day_of_year = float(raw[2:])
    year = 1900 + year_2d if year_2d >= 57 else 2000 + year_2d
    whole_days = int(math.floor(day_of_year)) - 1
    frac_day = day_of_year - math.floor(day_of_year)
    epoch = datetime(year, 1, 1) + timedelta(days=whole_days, seconds=frac_day * 86400.0)
    return epoch


def line2_mean_elements(line2: str) -> Dict[str, float]:
    tokens = line2.split()
    if len(tokens) < 8:
        raise ValueError(f"Invalid TLE line2: {line2!r}")

    inc_deg = float(tokens[2])
    raan_deg = float(tokens[3])
    ecc = float(f"0.{tokens[4].strip()}")
    mean_motion_rev_day = float(tokens[7])

    n_rad_s = mean_motion_rev_day * 2.0 * math.pi / 86400.0
    a_km = (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)

    return {
        "altitude_km": a_km - EARTH_RADIUS_KM,
        "inclination_deg": inc_deg,
        "eccentricity": ecc,
        "raan_deg": raan_deg,
    }


def iter_tle_pairs(lines: Sequence[str]) -> Iterable[Tuple[str, str]]:
    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx]
        if line.startswith("0 "):
            idx += 1
            if idx + 1 >= n:
                break
            line1 = lines[idx]
            line2 = lines[idx + 1]
            idx += 2
        else:
            if idx + 1 >= n:
                break
            line1 = lines[idx]
            line2 = lines[idx + 1]
            idx += 2

        if not line1.startswith("1 ") or not line2.startswith("2 "):
            continue
        yield line1, line2


def load_tle_histories(
    tle_file: Path,
    start_dt: datetime,
    end_dt_inclusive: datetime,
) -> Dict[str, List[Dict[str, str]]]:
    with tle_file.open("r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    end_exclusive = end_dt_inclusive + timedelta(days=1)
    histories: Dict[str, List[Dict[str, str]]] = {}
    n_pairs = 0
    n_kept = 0

    for line1, line2 in iter_tle_pairs(raw_lines):
        n_pairs += 1
        satnum_1 = line1[2:7].strip()
        satnum_2 = line2[2:7].strip()
        if not satnum_1 or satnum_1 != satnum_2:
            continue

        epoch = parse_tle_epoch(line1)
        if epoch < start_dt or epoch >= end_exclusive:
            continue

        sat_label = f"NORAD-{satnum_1}"
        record = {
            "OBJECT_NAME": sat_label,
            "NORAD_CAT_ID": satnum_1,
            "EPOCH": epoch.isoformat(timespec="microseconds"),
            "TLE_LINE0": f"0 {sat_label}",
            "TLE_LINE1": line1,
            "TLE_LINE2": line2,
        }
        histories.setdefault(satnum_1, []).append(record)
        n_kept += 1

    for satnum in histories:
        histories[satnum].sort(key=lambda rec: rec["EPOCH"])

    logging.info("Parsed %d TLE pairs, kept %d records across %d satellites", n_pairs, n_kept, len(histories))
    return histories


def compute_shell_means(tle_records: Sequence[Dict[str, str]]) -> Dict[str, float]:
    altitudes: List[float] = []
    inclinations: List[float] = []
    eccentricities: List[float] = []
    raans: List[float] = []
    bstars: List[float] = []

    for rec in tle_records:
        elems = line2_mean_elements(rec["TLE_LINE2"])
        altitudes.append(elems["altitude_km"])
        inclinations.append(elems["inclination_deg"])
        eccentricities.append(elems["eccentricity"])
        raans.append(elems["raan_deg"])
        bstars.append(extract_ballistic_coefficient(rec["TLE_LINE1"]))

    return {
        "Altitude_km": float(np.mean(altitudes)),
        "Inclination_deg": float(np.mean(inclinations)),
        "Eccentricity": float(np.mean(eccentricities)),
        "RAAN_deg": float(np.mean(raans)),
        "BSTAR_median": float(np.median(bstars)),
    }


def build_per_satellite_stats(
    histories: Dict[str, List[Dict[str, str]]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    satnums = sorted(histories.keys())
    if args.max_satellites is not None:
        satnums = satnums[: args.max_satellites]

    total = len(satnums)
    for idx, satnum in enumerate(satnums, start=1):
        tle_records = histories[satnum]
        if len(tle_records) < args.min_tles:
            continue

        try:
            X, y, _timestamps = build_tle_error_dataset(
                tle_records,
                max_dt=float(args.max_dt_days) * 86400.0,
                seed=int(args.seed),
                orbit_samples_n=int(args.orbit_samples_n),
                delta_n_threshold=float(args.delta_n_threshold),
                iqr_k=float(args.iqr_k),
            )
        except Exception as exc:
            logging.warning("Skip NORAD-%s due to preprocessing error: %s", satnum, exc)
            continue

        if len(y) < args.min_samples:
            continue

        t_vals = np.asarray(y)[:, 1].astype(np.float64)
        t_abs = np.abs(t_vals)
        shell_means = compute_shell_means(tle_records)

        row: Dict[str, float] = {
            "Satellite": f"NORAD-{satnum}",
            "NORAD_CAT_ID": satnum,
            "n_tle": len(tle_records),
            "n_samples": int(len(y)),
            "T_mean": float(np.mean(t_vals)),
            "T_std": float(np.std(t_vals)),
            "T_rms": float(np.sqrt(np.mean(t_vals ** 2))),
            "T_abs_mean": float(np.mean(t_abs)),
            "T_median": float(np.median(t_vals)),
            "T_abs_median": float(np.median(t_abs)),
            "T_p90_abs": float(np.percentile(t_abs, 90)),
            "T_p95_abs": float(np.percentile(t_abs, 95)),
        }
        row.update(shell_means)
        rows.append(row)

        if idx % 50 == 0 or idx == total:
            logging.info("Processed %d/%d satellites, retained %d", idx, total, len(rows))

    if not rows:
        raise RuntimeError("No satellites satisfied the TLE/sample thresholds.")

    return pd.DataFrame(rows).sort_values("Satellite").reset_index(drop=True)


def quantile_iqr(values: np.ndarray) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def build_shell_mask(
    stats_df: pd.DataFrame,
    center_idx: int,
    alt_window_km: float,
    inc_window_deg: float,
    ecc_window: float,
) -> np.ndarray:
    alt = stats_df["Altitude_km"].to_numpy(dtype=float)
    inc = stats_df["Inclination_deg"].to_numpy(dtype=float)
    ecc = stats_df["Eccentricity"].to_numpy(dtype=float)

    mask = (
        (np.abs(alt - alt[center_idx]) <= alt_window_km)
        & (np.abs(inc - inc[center_idx]) <= inc_window_deg)
        & (np.abs(ecc - ecc[center_idx]) <= ecc_window)
    )
    return mask


def summarize_shell(
    stats_df: pd.DataFrame,
    member_mask: np.ndarray,
    center_idx: int,
) -> Dict[str, float]:
    shell_df = stats_df.loc[member_mask].copy()
    t_p90_abs = shell_df["T_p90_abs"].to_numpy(dtype=float)
    t_std = shell_df["T_std"].to_numpy(dtype=float)
    t_med_abs = np.abs(shell_df["T_median"].to_numpy(dtype=float))
    p10 = max(float(np.percentile(t_p90_abs, 10)), 1.0e-9)
    p90 = float(np.percentile(t_p90_abs, 90))
    max_min_ratio = float(np.max(t_p90_abs) / max(np.min(t_p90_abs), 1.0e-9))
    p90_p10_ratio = float(p90 / p10)
    member_count = int(len(shell_df))
    score = float(member_count * math.log1p(max(p90_p10_ratio, 1.0)))

    return {
        "center_satellite": str(stats_df.iloc[center_idx]["Satellite"]),
        "center_norad": str(stats_df.iloc[center_idx]["NORAD_CAT_ID"]),
        "member_count": member_count,
        "score": score,
        "Altitude_center_km": float(stats_df.iloc[center_idx]["Altitude_km"]),
        "Inclination_center_deg": float(stats_df.iloc[center_idx]["Inclination_deg"]),
        "Eccentricity_center": float(stats_df.iloc[center_idx]["Eccentricity"]),
        "Altitude_span_km": float(shell_df["Altitude_km"].max() - shell_df["Altitude_km"].min()),
        "Inclination_span_deg": float(shell_df["Inclination_deg"].max() - shell_df["Inclination_deg"].min()),
        "Eccentricity_span": float(shell_df["Eccentricity"].max() - shell_df["Eccentricity"].min()),
        "T_p90_abs_median_km": float(np.median(t_p90_abs)),
        "T_p90_abs_iqr_km": quantile_iqr(t_p90_abs),
        "T_p90_abs_p90_p10_ratio": p90_p10_ratio,
        "T_p90_abs_max_min_ratio": max_min_ratio,
        "T_std_median_km": float(np.median(t_std)),
        "T_std_iqr_km": quantile_iqr(t_std),
        "abs_T_median_median_km": float(np.median(t_med_abs)),
        "abs_T_median_iqr_km": quantile_iqr(t_med_abs),
        "members": ",".join(shell_df["Satellite"].tolist()),
    }


def rank_shell_candidates(stats_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.center_norad is not None:
        forced = f"NORAD-{args.center_norad}" if not args.center_norad.startswith("NORAD-") else args.center_norad
        matches = np.where(stats_df["Satellite"].to_numpy() == forced)[0]
        if len(matches) == 0:
            raise ValueError(f"Requested center satellite not found: {forced}")
        candidate_indices = [int(matches[0])]
    else:
        candidate_indices = list(range(len(stats_df)))

    seen_member_sets = set()
    candidate_rows: List[Dict[str, float]] = []

    for center_idx in candidate_indices:
        member_mask = build_shell_mask(
            stats_df,
            center_idx,
            alt_window_km=float(args.alt_window_km),
            inc_window_deg=float(args.inc_window_deg),
            ecc_window=float(args.ecc_window),
        )
        member_indices = tuple(np.flatnonzero(member_mask).tolist())
        if len(member_indices) < args.min_shell_size:
            continue
        if member_indices in seen_member_sets:
            continue
        seen_member_sets.add(member_indices)
        candidate_rows.append(summarize_shell(stats_df, member_mask, center_idx))

    if not candidate_rows:
        raise RuntimeError("No shell candidates satisfied the requested thresholds.")

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df = candidate_df.sort_values(
        ["score", "member_count", "T_p90_abs_p90_p10_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return candidate_df


def save_selected_shell_outputs(
    stats_df: pd.DataFrame,
    selected_shell: pd.Series,
    output_dir: Path,
) -> None:
    member_names = selected_shell["members"].split(",")
    shell_df = stats_df.loc[stats_df["Satellite"].isin(member_names)].copy()
    shell_df = shell_df.sort_values("T_p90_abs").reset_index(drop=True)
    shell_df.to_csv(output_dir / "selected_shell_members.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax_left, ax_right = axes

    x = np.arange(len(shell_df))
    scatter = ax_left.scatter(
        x,
        shell_df["T_p90_abs"].to_numpy(dtype=float),
        c=shell_df["T_std"].to_numpy(dtype=float),
        cmap="viridis",
        s=48,
        edgecolors="white",
        linewidths=0.7,
    )
    ax_left.plot(x, shell_df["T_p90_abs"].to_numpy(dtype=float), color="#6c6c6c", alpha=0.45, linewidth=1.0)
    ax_left.set_xlabel("Satellites in selected shell (sorted by $T_{p90,abs}$)")
    ax_left.set_ylabel("$T_{p90,abs}$ (km)")
    ax_left.set_title("Same-shell cross-satellite dispersion")
    ax_left.grid(alpha=0.25, linewidth=0.6)

    label_indices = list(range(min(3, len(shell_df)))) + list(range(max(len(shell_df) - 3, 0), len(shell_df)))
    for idx in sorted(set(label_indices)):
        row = shell_df.iloc[idx]
        ax_left.annotate(
            row["Satellite"],
            (x[idx], row["T_p90_abs"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    cbar = fig.colorbar(scatter, ax=ax_left)
    cbar.set_label("$T_{std}$ (km)")

    metric_values = [
        shell_df["T_std"].to_numpy(dtype=float),
        shell_df["T_p90_abs"].to_numpy(dtype=float),
        np.abs(shell_df["T_median"].to_numpy(dtype=float)),
    ]
    metric_labels = ["$T_{std}$", "$T_{p90,abs}$", "$|T_{median}|$"]
    ax_right.boxplot(metric_values, labels=metric_labels, showmeans=True)
    ax_right.set_ylabel("Satellite-level statistic (km)")
    ax_right.set_title("Distribution of shell-level satellite statistics")
    ax_right.grid(axis="y", alpha=0.25, linewidth=0.6)

    shell_title = (
        f"N={int(selected_shell['member_count'])}, "
        f"h={selected_shell['Altitude_center_km']:.1f}±{selected_shell['Altitude_span_km'] / 2.0:.1f} km, "
        f"i={selected_shell['Inclination_center_deg']:.3f}±{selected_shell['Inclination_span_deg'] / 2.0:.3f} deg"
    )
    fig.suptitle(shell_title, fontsize=12)
    fig.savefig(output_dir / "selected_shell_dispersion.png", dpi=220)
    plt.close(fig)


def write_summary(
    args: argparse.Namespace,
    selected_shell: pd.Series,
    output_dir: Path,
) -> None:
    summary_lines = [
        "Selected same-shell summary",
        f"Input TLE file: {args.tle_file}",
        f"Date window: {args.start_date} to {args.end_date}",
        "",
        "Shell criteria:",
        f"  |Delta h| <= {args.alt_window_km:.3f} km",
        f"  |Delta i| <= {args.inc_window_deg:.5f} deg",
        f"  |Delta e| <= {args.ecc_window:.6g}",
        "",
        "Selected shell:",
        f"  Center satellite: {selected_shell['center_satellite']}",
        f"  Satellite count: {int(selected_shell['member_count'])}",
        f"  Altitude center/span: {selected_shell['Altitude_center_km']:.3f} km / {selected_shell['Altitude_span_km']:.3f} km",
        f"  Inclination center/span: {selected_shell['Inclination_center_deg']:.5f} deg / {selected_shell['Inclination_span_deg']:.5f} deg",
        f"  Eccentricity center/span: {selected_shell['Eccentricity_center']:.7f} / {selected_shell['Eccentricity_span']:.7f}",
        "",
        "Dispersion metrics:",
        f"  Median T_p90_abs: {selected_shell['T_p90_abs_median_km']:.3f} km",
        f"  IQR T_p90_abs: {selected_shell['T_p90_abs_iqr_km']:.3f} km",
        f"  P90/P10 of T_p90_abs: {selected_shell['T_p90_abs_p90_p10_ratio']:.3f}",
        f"  Max/Min of T_p90_abs: {selected_shell['T_p90_abs_max_min_ratio']:.3f}",
        f"  Median T_std: {selected_shell['T_std_median_km']:.3f} km",
        f"  IQR T_std: {selected_shell['T_std_iqr_km']:.3f} km",
        f"  Median |T_median|: {selected_shell['abs_T_median_median_km']:.3f} km",
        "",
        "Suggested paper sentence:",
        (
            "Within a narrow orbital shell defined by nearly identical altitude, inclination, and "
            "eccentricity, satellites still exhibit substantial cross-satellite dispersion in along-track "
            "error statistics, with the selected shell showing "
            f"a T_p90_abs P90/P10 ratio of {selected_shell['T_p90_abs_p90_p10_ratio']:.2f} "
            f"across {int(selected_shell['member_count'])} satellites."
        ),
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)

    histories = load_tle_histories(args.tle_file, start_dt, end_dt)
    stats_df = build_per_satellite_stats(histories, args)
    stats_df.to_csv(args.output_dir / "per_satellite_stats.csv", index=False)
    logging.info("Saved per-satellite statistics: %s", args.output_dir / "per_satellite_stats.csv")

    candidate_df = rank_shell_candidates(stats_df, args)
    candidate_df.head(int(args.top_shells)).to_csv(args.output_dir / "candidate_shells.csv", index=False)
    logging.info("Saved shell ranking: %s", args.output_dir / "candidate_shells.csv")

    selected_shell = candidate_df.iloc[0]
    save_selected_shell_outputs(stats_df, selected_shell, args.output_dir)
    write_summary(args, selected_shell, args.output_dir)

    logging.info("Selected shell center: %s", selected_shell["center_satellite"])
    logging.info("Selected shell size: %d", int(selected_shell["member_count"]))
    logging.info(
        "Selected shell T_p90_abs P90/P10 ratio: %.3f",
        float(selected_shell["T_p90_abs_p90_p10_ratio"]),
    )
    logging.info("Outputs saved under: %s", args.output_dir)


if __name__ == "__main__":
    main()
