from __future__ import annotations

import argparse
from pathlib import Path

from plan5.manifest import build_manifest_from_testset, write_manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True)
    parser.add_argument("--instances-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root-id", default="D01")
    parser.add_argument("--split", default="benchmark")
    parser.add_argument("--family", default="unclassified")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    entries = build_manifest_from_testset(
        testset_path=args.testset,
        instances_root=args.instances_root,
        source_root_id=args.source_root_id,
        split=args.split,
        family=args.family,
    )
    output = write_manifest_csv(entries, args.output)
    print(f"wrote {len(entries)} rows to {Path(output).resolve()}")
