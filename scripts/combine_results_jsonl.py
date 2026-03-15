from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[tuple[str, str], dict] = {}
    for raw_path in args.input_jsonl:
        path = Path(raw_path)
        for row in load_rows(path):
            merged[(str(row.get("instance_id", "")), str(row.get("mode", "")))] = row

    rows = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("instance_id", "")),
            str(row.get("mode", "")),
            str(row.get("run_id", "")),
        ),
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {output_path.resolve()}")
