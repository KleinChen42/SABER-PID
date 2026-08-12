"""Inspect and selectively materialize members from a large remote ZIP.

The downloader uses HTTP byte ranges and a sparse local file.  It first writes
only the ZIP central directory and end records, which is enough for Python's
``zipfile`` module to enumerate members.  Requested members are then fetched
individually at their original offsets, so a multi-gigabyte archive need not be
downloaded in full.

This is intended for immutable public research archives whose total size and
published checksum are recorded separately.  The sparse ZIP is a transport
cache, not a claim that the complete archive was downloaded or verified.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"


def fetch_range(url: str, start: int, end: int, retries: int = 8) -> tuple[bytes, int]:
    """Fetch an inclusive byte range and return payload plus archive size."""

    if start < 0 or end < start:
        raise ValueError(f"Invalid range {start}-{end}")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "SABER-PID-selective-archive-client/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = int(getattr(response, "status", response.getcode()))
                content_range = str(response.headers.get("Content-Range", ""))
                payload = response.read()
            if status != 206:
                raise RuntimeError(f"Server did not honor Range request: HTTP {status}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match:
                raise RuntimeError(f"Invalid Content-Range: {content_range!r}")
            actual_start, actual_end, total = map(int, match.groups())
            if actual_start != start or actual_end != end:
                raise RuntimeError(
                    f"Unexpected Content-Range {actual_start}-{actual_end}; expected {start}-{end}"
                )
            if len(payload) != end - start + 1:
                raise RuntimeError(
                    f"Short range read: {len(payload)} bytes; expected {end - start + 1}"
                )
            return payload, total
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.IncompleteRead,
            TimeoutError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Range request failed after {retries} attempts: {last_error}")


def fetch_range_chunked(
    url: str, start: int, end: int, *, chunk_bytes: int = 512 * 1024
) -> tuple[bytes, int]:
    """Fetch a large inclusive range as independently retryable chunks."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    chunks = []
    total_size: int | None = None
    position = start
    while position <= end:
        chunk_end = min(end, position + chunk_bytes - 1)
        payload, observed_total = fetch_range(url, position, chunk_end)
        if total_size is not None and observed_total != total_size:
            raise RuntimeError("Archive size changed between range chunks")
        total_size = observed_total
        chunks.append(payload)
        position = chunk_end + 1
    if total_size is None:
        raise RuntimeError("No range chunks were fetched")
    return b"".join(chunks), total_size


def locate_central_directory(url: str, total_size: int, tail_bytes: int) -> dict[str, Any]:
    tail_start = max(0, total_size - tail_bytes)
    tail, observed_total = fetch_range(url, tail_start, total_size - 1)
    if observed_total != total_size:
        raise RuntimeError(f"Archive size changed: expected {total_size}, observed {observed_total}")
    eocd_at = tail.rfind(EOCD_SIGNATURE)
    if eocd_at < 0 or len(tail) - eocd_at < 22:
        raise RuntimeError("EOCD record not found in requested archive tail")
    (
        _signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size_32,
        central_offset_32,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    if eocd_at + 22 + comment_length > len(tail):
        raise RuntimeError("EOCD comment extends beyond fetched tail")

    central_size = int(central_size_32)
    central_offset = int(central_offset_32)
    zip64 = central_size_32 == 0xFFFFFFFF or central_offset_32 == 0xFFFFFFFF or entry_count == 0xFFFF
    zip64_record: bytes | None = None
    zip64_offset: int | None = None
    if zip64:
        locator_at = eocd_at - 20
        if locator_at < 0 or tail[locator_at : locator_at + 4] != ZIP64_LOCATOR_SIGNATURE:
            raise RuntimeError("ZIP64 locator not found immediately before EOCD")
        _sig, _zip64_disk, zip64_offset, _disk_count = struct.unpack_from(
            "<4sLQL", tail, locator_at
        )
        zip64_record, observed_total = fetch_range(url, int(zip64_offset), int(zip64_offset) + 55)
        if observed_total != total_size or zip64_record[:4] != ZIP64_EOCD_SIGNATURE:
            raise RuntimeError("Invalid ZIP64 EOCD record")
        (
            _sig,
            _record_size,
            _version_made,
            _version_needed,
            disk_number,
            central_disk,
            entries_on_disk,
            entry_count,
            central_size,
            central_offset,
        ) = struct.unpack_from("<4sQ2H2L4Q", zip64_record, 0)

    if disk_number != 0 or central_disk != 0:
        raise RuntimeError("Multi-disk ZIP archives are not supported")
    if central_offset < 0 or central_size <= 0 or central_offset + central_size > total_size:
        raise RuntimeError("Central-directory bounds are invalid")
    return {
        "tail": tail,
        "tail_start": tail_start,
        "central_offset": int(central_offset),
        "central_size": int(central_size),
        "entry_count": int(entry_count),
        "entries_on_disk": int(entries_on_disk),
        "zip64": bool(zip64),
        "zip64_offset": zip64_offset,
        "zip64_record": zip64_record,
    }


def write_sparse_catalog(
    *, url: str, total_size: int, sparse_path: Path, tail_bytes: int
) -> dict[str, Any]:
    layout = locate_central_directory(url, total_size, tail_bytes)
    central, observed_total = fetch_range_chunked(
        url,
        layout["central_offset"],
        layout["central_offset"] + layout["central_size"] - 1,
    )
    if observed_total != total_size:
        raise RuntimeError("Archive size changed while reading central directory")
    sparse_path.parent.mkdir(parents=True, exist_ok=True)
    with sparse_path.open("wb") as handle:
        handle.truncate(total_size)
        handle.seek(layout["central_offset"])
        handle.write(central)
        handle.seek(layout["tail_start"])
        handle.write(layout["tail"])
        if layout["zip64_record"] is not None:
            handle.seek(int(layout["zip64_offset"]))
            handle.write(layout["zip64_record"])
    return {key: value for key, value in layout.items() if key not in {"tail", "zip64_record"}}


def info_dict(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "filename": info.filename,
        "file_size": int(info.file_size),
        "compress_size": int(info.compress_size),
        "compress_type": int(info.compress_type),
        "crc32": f"{info.CRC:08x}",
        "header_offset": int(info.header_offset),
        "is_dir": info.is_dir(),
    }


def materialize_member(url: str, total_size: int, sparse_path: Path, info: zipfile.ZipInfo) -> int:
    """Fetch one local header plus compressed payload into the sparse archive."""

    # One HTTP request per member.  The 256-KiB allowance covers the local
    # filename and extra fields without a separate header request.
    allowance = 256 * 1024
    start = int(info.header_offset)
    end = min(total_size - 1, start + int(info.compress_size) + allowance - 1)
    payload, observed_total = fetch_range_chunked(url, start, end)
    if observed_total != total_size or payload[:4] != LOCAL_HEADER_SIGNATURE:
        raise RuntimeError(f"Invalid local header for {info.filename}")
    if len(payload) < 30:
        raise RuntimeError(f"Truncated local header for {info.filename}")
    local = struct.unpack_from("<4s5H3L2H", payload, 0)
    filename_length, extra_length = int(local[-2]), int(local[-1])
    required = 30 + filename_length + extra_length + int(info.compress_size)
    if required > len(payload):
        raise RuntimeError(
            f"Range allowance insufficient for {info.filename}: need {required}, received {len(payload)}"
        )
    with sparse_path.open("r+b") as handle:
        handle.seek(start)
        handle.write(payload[:required])
    return required


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--total-size", type=int, required=True)
    parser.add_argument("--sparse-zip", required=True)
    parser.add_argument("--catalog-json", required=True)
    parser.add_argument("--member-regex", default="OPEN100")
    parser.add_argument("--tail-bytes", type=int, default=256 * 1024)
    parser.add_argument(
        "--reuse-sparse-catalog",
        action="store_true",
        help=(
            "Reuse an existing logical-size sparse ZIP whose central directory "
            "was populated by an earlier catalog pass"
        ),
    )
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--max-members", type=int, default=0)
    parser.add_argument("--max-matched-bytes", type=int, default=0)
    parser.add_argument("--extract-root")
    args = parser.parse_args()

    sparse_path = Path(args.sparse_zip).resolve()
    if args.reuse_sparse_catalog:
        if not sparse_path.is_file() or sparse_path.stat().st_size != args.total_size:
            raise RuntimeError(
                "--reuse-sparse-catalog requires an existing sparse ZIP with the frozen logical size"
            )
        layout = {
            "reused_sparse_catalog": True,
            "logical_size": sparse_path.stat().st_size,
        }
    else:
        layout = write_sparse_catalog(
            url=args.url,
            total_size=args.total_size,
            sparse_path=sparse_path,
            tail_bytes=args.tail_bytes,
        )
    pattern = re.compile(args.member_regex, re.IGNORECASE)
    with zipfile.ZipFile(sparse_path) as archive:
        all_infos = archive.infolist()
        selected = [info for info in all_infos if pattern.search(info.filename) and not info.is_dir()]
    if args.max_members > 0:
        selected = selected[: args.max_members]
    if args.max_matched_bytes > 0:
        selected = [info for info in selected if int(info.file_size) <= args.max_matched_bytes]

    fetched: list[dict[str, Any]] = []
    if args.materialize:
        for number, info in enumerate(selected, start=1):
            fetched_bytes = materialize_member(args.url, args.total_size, sparse_path, info)
            fetched.append({**info_dict(info), "fetched_bytes": fetched_bytes})
            print(
                json.dumps(
                    {"member": info.filename, "progress": f"{number}/{len(selected)}", "status": "fetched"},
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.extract_root:
            extract_root = Path(args.extract_root).resolve()
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(sparse_path) as archive:
                for info in selected:
                    archive.extract(info, extract_root)

    catalog = {
        "status": "pass",
        "transport": "HTTP byte-range sparse ZIP",
        "complete_archive_downloaded": False,
        "url": args.url,
        "total_size": args.total_size,
        "sparse_path": str(sparse_path),
        "sparse_allocated_bytes_note": "filesystem allocation is intentionally much smaller than logical size",
        "layout": layout,
        "archive_member_count": len(all_infos),
        "member_regex": args.member_regex,
        "matched_member_count": len(selected),
        "matched_members": [info_dict(info) for info in selected],
        "materialized_member_count": len(fetched),
        "materialized_members": fetched,
    }
    catalog_path = Path(args.catalog_json).resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "archive_members": len(all_infos),
                "matched_members": len(selected),
                "catalog": str(catalog_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
