#!/usr/bin/env python3
"""Probe Starlink public ephemeris endpoints and attempt real downloads."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


README_URL = "https://api.starlink.com/public-files/ephemerides/README.md"
DIRECTORY_URL = "https://api.starlink.com/public-files/ephemerides/"
MANIFEST_CANDIDATES = [
    "https://api.starlink.com/public-files/ephemerides/MANIFEST.txt",
    "https://api.starlink.com/public-files/ephemerides/manifest.txt",
    "https://starlink.com/public-files/ephemerides/MANIFEST.txt",
]
USER_AGENT = "Mozilla/5.0 (compatible; ephemeris-probe/1.0)"


def http_get(url: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", resp.getcode())
            headers = {k: v for k, v in resp.headers.items()}
            body = resp.read()
            return int(status), headers, body
    except HTTPError as exc:
        body = exc.read() if exc.fp is not None else b""
        headers = {k: v for k, v in exc.headers.items()} if exc.headers else {}
        return int(exc.code), headers, body
    except URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc


def preview_text(data: bytes, limit: int = 200) -> str:
    txt = data.decode("utf-8", errors="replace").replace("\n", "\\n")
    return txt[:limit]


def parse_manifest_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            lines.append(line)
            continue
        if re.search(r"\.(txt|itc|zip|gz)$", line, flags=re.IGNORECASE):
            lines.append(line)
    return lines


def to_url(entry: str) -> str:
    if entry.startswith("http://") or entry.startswith("https://"):
        return entry
    return urljoin(DIRECTORY_URL, entry)


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def probe_urls(urls: Iterable[str], timeout: float) -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    for url in urls:
        try:
            status, _headers, body = http_get(url, timeout)
            rows.append((url, status, len(body), preview_text(body)))
        except RuntimeError as exc:
            rows.append((url, -1, 0, str(exc)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether Starlink ephemeris files can be downloaded now."
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/starlink_ephemeris_probe",
        help="Directory for downloaded test files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=2,
        help="If manifest is available, try downloading this many entries.",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="HTTP timeout in seconds."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Endpoint probe ===")
    rows = probe_urls([README_URL, DIRECTORY_URL, *MANIFEST_CANDIDATES], args.timeout)
    for url, status, size, preview in rows:
        print(f"[{status:>3}] {url}")
        print(f"      bytes={size} preview={preview}")

    manifest_text = None
    manifest_url = None
    for candidate in MANIFEST_CANDIDATES:
        try:
            status, _headers, body = http_get(candidate, args.timeout)
        except RuntimeError:
            continue
        if status == 200 and body.strip():
            manifest_text = body.decode("utf-8", errors="replace")
            manifest_url = candidate
            break

    if manifest_text is None:
        print("\nResult: manifest is not downloadable (all candidates failed).")
        print(
            "Conclusion: automatic public download path is currently unavailable from these URLs."
        )
        return 2

    manifest_path = output_dir / "MANIFEST.txt"
    write_file(manifest_path, manifest_text.encode("utf-8"))
    print(f"\nManifest source: {manifest_url}")
    print(f"Saved: {manifest_path}")

    entries = parse_manifest_lines(manifest_text)
    if not entries:
        print("Manifest downloaded but no file entries were parsed.")
        return 3

    print(f"Parsed entries: {len(entries)}")
    download_count = min(max(args.max_files, 1), len(entries))
    print(f"Trying first {download_count} entries...")

    success = 0
    for idx, entry in enumerate(entries[:download_count], start=1):
        url = to_url(entry)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", entry.strip("/")) or f"file_{idx}"
        local_path = output_dir / safe_name
        try:
            status, _headers, body = http_get(url, args.timeout)
        except RuntimeError as exc:
            print(f"[ERR] {url} -> {exc}")
            continue
        if status == 200 and body:
            write_file(local_path, body)
            print(f"[ OK] {url} -> {local_path} ({len(body)} bytes)")
            success += 1
        else:
            print(f"[{status:>3}] {url} -> not downloaded")

    if success == 0:
        print("Result: manifest exists but test file downloads failed.")
        return 4

    print(f"Result: download succeeded for {success}/{download_count} test files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
