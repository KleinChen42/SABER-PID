"""Parallel, best-effort PIDQA image acquisition for a declared sheet list.

Unlike the original strict pilot downloader, one slow or missing mirror object
does not abort the whole source set.  The summary preserves failures so an
incomplete image set cannot be mistaken for a complete experiment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from pidbench.io import read_jsonl, write_json


MIRROR_ROOT = "https://hf-mirror.com/datasets/kunalnchemtech/digitize-pid-symbols/resolve/main/DigitizePID_Dataset"


def download_sheet(sheet: str, output_root: Path, timeout: int) -> tuple[str, str, str | None]:
    if not sheet.isdigit():
        return sheet, "failed", "non_numeric_sheet"
    last_error: str | None = None
    for partition in ("train", "val"):
        target = output_root / partition / f"{sheet}.jpg"
        if target.exists() and target.stat().st_size > 0:
            return sheet, "existing", partition
        url = f"{MIRROR_ROOT}/{partition}/{sheet}.jpg"
        request = urllib.request.Request(url, headers={"User-Agent": "pid-reliability-benchmark/0.1"})
        temporary = target.with_suffix(".jpg.part")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if temporary.stat().st_size == 0:
                temporary.unlink(missing_ok=True)
                last_error = "zero_byte_response"
                continue
            temporary.replace(target)
            return sheet, "downloaded", partition
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            last_error = f"HTTP_{error.code}"
            if error.code == 404:
                continue
        except Exception as error:  # keep the source list moving
            temporary.unlink(missing_ok=True)
            last_error = f"{type(error).__name__}: {error}"
    return sheet, "failed", last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sheets = sorted({str(row["source_sheet"]) for row in read_jsonl(args.records)}, key=int)
    output_root = Path(args.output_root)
    results: list[tuple[str, str, str | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_sheet, sheet, output_root, args.timeout) for sheet in sheets]
        for future in futures:
            results.append(future.result())
    results.sort(key=lambda row: int(row[0]))
    summary = {
        "mirror_root": MIRROR_ROOT,
        "requested_sheet_count": len(sheets),
        "image_count": sum(status in {"existing", "downloaded"} for _, status, _ in results),
        "downloaded_count": sum(status == "downloaded" for _, status, _ in results),
        "existing_count": sum(status == "existing" for _, status, _ in results),
        "failed_count": sum(status == "failed" for _, status, _ in results),
        "failed_sheets": [
            {"sheet": sheet, "error": detail}
            for sheet, status, detail in results
            if status == "failed"
        ],
        "partition_counts": {
            partition: sum(
                status in {"existing", "downloaded"} and detail == partition
                for _, status, detail in results
            )
            for partition in ("train", "val")
        },
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
