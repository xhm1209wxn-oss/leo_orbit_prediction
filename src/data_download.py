"""
TLE 数据下载模块
负责从 Space-Track.org 下载 TLE 历史数据
"""

import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple
from pathlib import Path
import json

from spacetrack import SpaceTrackClient

logger = logging.getLogger(__name__)


class TLEDownloader:
    """来自 Space-Track.org 的 TLE 数据下载器"""

    @staticmethod
    def _cache_file_path(base_dir: Path, sat_name: str) -> Path:
        """构建单颗卫星的缓存文件路径。"""
        safe_name = sat_name.replace('/', '_')
        return base_dir / f"{safe_name}_tle.json"

    def __init__(self, username: str, password: str):
        """
        初始化 TLE 下载器

        参数：
            username: Space-Track 用户名
            password: Space-Track 密码
        """
        self.spacetrack_client = SpaceTrackClient(username, password)
        logger.info("Connected to Space-Track.org")

    def _save_single_satellite_cache(self, save_path: Path, sat_name: str, tle_data: List[Dict]) -> None:
        """立即写回单颗卫星缓存，便于中断后复用已下载结果。"""
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = self._cache_file_path(save_path, sat_name)
        with open(file_path, 'w') as f:
            json.dump(tle_data, f, indent=2)

    def download_tle_history(self,
                             norad_id: int,
                             start_date: str,
                             end_date: str) -> List[Dict]:
        """
        下载单颗卫星的 TLE 历史

        参数：
            norad_id: NORAD 编号
            start_date: 起始日期（UTC，YYYY-MM-DD）
            end_date: 结束日期（UTC，YYYY-MM-DD）

        返回：
            TLE 记录列表
        """
        start_dt, end_dt, date_range = self._build_date_range(start_date, end_date)

        logger.info(
            f"Requesting TLEs for NORAD {norad_id} from {start_dt.strftime('%Y-%m-%d')} "
            f"to {end_dt.strftime('%Y-%m-%d')}"
        )

        try:
            tle_data = self.spacetrack_client.gp_history(
                norad_cat_id=norad_id,
                epoch=date_range,
                orderby='epoch asc',
                format='json'
            )

            if isinstance(tle_data, str):
                tle_data = json.loads(tle_data)

            logger.info(f"Retrieved {len(tle_data)} TLE sets for NORAD {norad_id}")

            return tle_data
        except Exception as e:
            logger.error(f"Error in download_tle_history for NORAD {norad_id}: {type(e).__name__}: {str(e)}")
            logger.error(f"Exception repr: {repr(e)}")
            raise

    def download_multiple_satellites(self,
                                      satellite_dict: Dict[str, int],
                                      start_date: str,
                                      end_date: str,
                                      skip_cached: bool = False,
                                      cache_dir: str = None,
                                      batch_size: int = 200,
                                      request_delay_seconds: float = 2.0) -> Tuple[Dict[str, List[Dict]], Set[str]]:
        """
        通过批量查询下载多颗卫星的 TLE 历史

        参数：
            satellite_dict: 卫星名称与 NORAD ID 的映射
            start_date: 起始日期（UTC，YYYY-MM-DD）
            end_date: 结束日期（UTC，YYYY-MM-DD）
            skip_cached: 若为 True，则从 cache_dir 读取已有 JSON 并跳过下载
            cache_dir: 缓存 TLE JSON 的目录
            batch_size: 每次请求的卫星数量
            request_delay_seconds: 回退模式下单次请求间的延时

        返回：
            二元组：
                - 卫星名称到 TLE 数据的映射
                - 本轮需要写回缓存的卫星名称集合
        """
        logger.info("=" * 70)
        logger.info("Downloading TLE histories from Space-Track.org (BATCH MODE)")
        logger.info("=" * 70)
        logger.info(f"Total satellites to download: {len(satellite_dict)}")

        tle_histories = {}
        missing_satellites = {}
        cache_path = Path(cache_dir) if cache_dir else None

        if skip_cached and cache_path and cache_path.exists():
            for sat_name in sorted(satellite_dict.keys()):
                norad_id = satellite_dict[sat_name]
                file_path = self._cache_file_path(cache_path, sat_name)
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            tle_data = json.load(f)
                        tle_histories[sat_name] = tle_data
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load cache for {sat_name}: {e}")
                missing_satellites[sat_name] = norad_id
        else:
            missing_satellites = {
                sat_name: satellite_dict[sat_name]
                for sat_name in sorted(satellite_dict.keys())
            }

        satellites_to_refresh = set(missing_satellites.keys())

        if not missing_satellites:
            logger.info("All satellites loaded from cache, skipping download.")
            return tle_histories, satellites_to_refresh

        start_dt, end_dt, date_range = self._build_date_range(start_date, end_date)

        norad_items = sorted(missing_satellites.items(), key=lambda x: x[0])
        total_missing = len(norad_items)

        logger.info(
            f"Requesting TLEs for {total_missing} satellites from {start_dt.strftime('%Y-%m-%d')} "
            f"to {end_dt.strftime('%Y-%m-%d')}"
        )
        logger.info(f"Batch size: {batch_size}, request delay (fallback): {request_delay_seconds}s")

        for sat_name in sorted(missing_satellites.keys()):
            tle_histories.setdefault(sat_name, [])

        for batch_start in range(0, total_missing, batch_size):
            batch = norad_items[batch_start:batch_start + batch_size]
            batch_ids = [str(nid) for _, nid in batch]
            norad_id_str = ','.join(batch_ids)
            logger.info(f"Batch {batch_start//batch_size + 1}: requesting {len(batch)} satellites")

            try:
                all_tle_data = self.spacetrack_client.gp_history(
                    norad_cat_id=norad_id_str,
                    epoch=date_range,
                    orderby='epoch asc',
                    format='json'
                )

                if isinstance(all_tle_data, str):
                    all_tle_data = json.loads(all_tle_data)

                logger.info(f"✓ Batch retrieved {len(all_tle_data)} TLE sets")
            except Exception as e:
                logger.error(f"Batch download failed: {type(e).__name__}: {str(e)}")
                logger.error(f"Exception details: {repr(e)}")
                import traceback
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                logger.warning("Falling back to individual downloads with rate limiting for this batch...")
                batch_dict = {sat: nid for sat, nid in batch}
                fallback_data = self._download_individual_with_delay(
                    batch_dict,
                    start_date,
                    end_date,
                    request_delay_seconds,
                    cache_dir=cache_path,
                )
                tle_histories.update(fallback_data)
                continue

            id_to_name = {str(nid): sat for sat, nid in batch}
            for tle in all_tle_data:
                norad_cat_id = str(tle.get('NORAD_CAT_ID', ''))
                if norad_cat_id in id_to_name:
                    sat_name = id_to_name[norad_cat_id]
                    tle_histories[sat_name].append(tle)

            if cache_path is not None:
                saved_in_batch = 0
                for sat_name, _norad_id in batch:
                    self._save_single_satellite_cache(cache_path, sat_name, tle_histories[sat_name])
                    saved_in_batch += 1
                logger.info(
                    f"Batch {batch_start//batch_size + 1}: incremental cache saved for "
                    f"{saved_in_batch} satellites to {cache_path}"
                )

        logger.info("\n" + "=" * 70)
        logger.info("Download Summary:")
        logger.info("=" * 70)

        total_tles = 0
        no_data_count = 0
        low_data_count = 0

        for sat_name in sorted(tle_histories.keys()):
            tle_data = tle_histories[sat_name]
            count = len(tle_data)
            total_tles += count

            if count == 0:
                no_data_count += 1
            elif count < 5:
                low_data_count += 1

        logger.info("\n" + "=" * 70)
        logger.info(f"TLE download complete! Used date range {start_dt.strftime('%Y-%m-%d')}--{end_dt.strftime('%Y-%m-%d')}")
        logger.info(f"Total TLE sets: {total_tles}")
        logger.info(f"Satellites with no data: {no_data_count}")
        logger.info(f"Satellites with low data (<5 TLEs): {low_data_count}")
        logger.info(f"Usable satellites (>=5 TLEs): {len(satellite_dict) - no_data_count - low_data_count}")
        logger.info("=" * 70)

        return tle_histories, satellites_to_refresh

    def _download_individual_with_delay(self,
                                         satellite_dict: Dict[str, int],
                                         start_date: str,
                                         end_date: str,
                                         delay_seconds: float = 2.0,
                                         cache_dir: Path = None) -> Dict[str, List[Dict]]:
        """
        回退方法：逐个卫星下载并加入延时

        参数：
            satellite_dict: 卫星名称与 NORAD ID 的映射
            start_date: 起始日期（UTC，YYYY-MM-DD）
            end_date: 结束日期（UTC，YYYY-MM-DD）

        返回：
            卫星名称到 TLE 数据的映射
        """
        import time

        tle_histories = {}
        total_tles = 0
        total = len(satellite_dict)

        ordered_sat_items = sorted(satellite_dict.items(), key=lambda x: x[0])
        for idx, (sat_name, norad_id) in enumerate(ordered_sat_items, 1):
            logger.info(f"\n[{idx}/{total}] {sat_name} (NORAD ID: {norad_id}):")

            try:
                tle_data = self.download_tle_history(norad_id, start_date, end_date)
                tle_histories[sat_name] = tle_data
                total_tles += len(tle_data)
                if cache_dir is not None:
                    self._save_single_satellite_cache(cache_dir, sat_name, tle_data)
                    logger.info(f"Incremental cache saved: {sat_name}")

                if idx < total:  # 最后一个请求后不再等待
                    time.sleep(delay_seconds)

            except Exception as e:
                logger.error(f"Failed to download {sat_name}: {type(e).__name__}: {str(e)}")
                logger.error(f"Exception details: {repr(e)}")
                import traceback
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                tle_histories[sat_name] = []

        logger.info("\n" + "=" * 70)
        logger.info(f"TLE download complete! Used date range {start_date}--{end_date}")
        logger.info(f"Total TLE sets: {total_tles}")
        logger.info("=" * 70)

        return tle_histories

    @staticmethod
    def _build_date_range(start_date: str, end_date: str):
        """构建并校验 Space-Track 需要的日期范围字符串。"""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid start_date '{start_date}', expected YYYY-MM-DD") from exc
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid end_date '{end_date}', expected YYYY-MM-DD") from exc
        if start_dt > end_dt:
            raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")
        date_range = f"{start_dt.strftime('%Y-%m-%d')}--{end_dt.strftime('%Y-%m-%d')}"
        return start_dt, end_dt, date_range

    def save_tle_data(self,
                      tle_histories: Dict[str, List[Dict]],
                      save_dir: str,
                      satellite_names: Set[str] = None):
        """
        将 TLE 数据保存为 JSON 文件

        参数：
            tle_histories: TLE 历史数据字典
            save_dir: TLE 数据保存目录
            satellite_names: 仅保存这些卫星；为 None 时保存全部
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        saved = 0
        names_to_save = satellite_names if satellite_names is not None else set(tle_histories.keys())

        if not names_to_save:
            logger.info(f"No TLE cache files needed updating in {save_path}")
            return

        for sat_name in sorted(names_to_save):
            if sat_name not in tle_histories:
                logger.warning(f"Skipping cache save for {sat_name}: no TLE data present in memory")
                continue
            tle_data = tle_histories[sat_name]
            file_path = self._cache_file_path(save_path, sat_name)

            with open(file_path, 'w') as f:
                json.dump(tle_data, f, indent=2)

            saved += 1

        logger.info(f"TLE JSON saved for {saved} satellites to {save_path}")
