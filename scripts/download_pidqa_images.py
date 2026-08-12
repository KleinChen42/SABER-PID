"""Fetch only the public PIDQA sheet images needed for a selected pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from pidbench.io import read_jsonl, write_json


MIRROR_ROOT = "https://hf-mirror.com/datasets/kunalnchemtech/digitize-pid-symbols/resolve/main/DigitizePID_Dataset"


def download_sheet(sheet: str, output_root: Path, timeout: int) -> str:
    if not sheet.isdigit():
        raise ValueError(f"Unexpected PIDQA sheet identifier: {sheet!r}")
    for partition in ("train", "val"):
        target = output_root / partition / f"{sheet}.jpg"
        if target.exists() and target.stat().st_size > 0:
            return partition
        url = f"{MIRROR_ROOT}/{partition}/{sheet}.jpg"
        request = urllib.request.Request(url, headers={"User-Agent": "pid-reliability-benchmark/0.1"})
        temporary = target.with_suffix(".jpg.part")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if temporary.stat().st_size == 0:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Downloaded zero-byte image for sheet {sheet}")
            temporary.replace(target)
            return partition
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code == 404:
                continue
            raise RuntimeError(f"Failed fetching {url}: HTTP {error.code}") from error
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    raise FileNotFoundError(f"No train/val image found for PIDQA sheet {sheet}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="Answer-isolated selected records")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    sheets = sorted({str(row["source_sheet"]) for row in read_jsonl(args.records)}, key=int)
    output_root = Path(args.output_root)
    location_by_sheet = {
        sheet: download_sheet(sheet, output_root, args.timeout)
        for sheet in sheets
    }
    summary = {
        "image_count": len(location_by_sheet),
        "sheets": sheets,
        "partition_counts": {
            partition: sum(location == partition for location in location_by_sheet.values())
            for partition in ("train", "val")
        },
        "mirror_root": MIRROR_ROOT,
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
