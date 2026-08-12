"""Audit same-drawing semantic-query exposure for a PIDQA split."""

from __future__ import annotations

import argparse
import json

from pidbench.exposure import build_same_source_cache_audit
from pidbench.io import read_jsonl, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    predictions, summary = build_same_source_cache_audit(
        read_jsonl(args.records), read_jsonl(args.split)
    )
    write_jsonl(args.predictions, predictions)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
