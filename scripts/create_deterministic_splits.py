from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--dev-size", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest.sort_values("instance_id").reset_index(drop=True)
    pilot = manifest.iloc[: args.pilot_size].copy()
    dev = manifest.iloc[args.pilot_size : args.pilot_size + args.dev_size].copy()
    test = manifest.iloc[args.pilot_size + args.dev_size :].copy()

    pilot["split"] = "pilot"
    dev["split"] = "dev"
    test["split"] = "test"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(outdir / "pilot_manifest.csv", index=False)
    dev.to_csv(outdir / "dev_manifest.csv", index=False)
    test.to_csv(outdir / "test_manifest.csv", index=False)
    print(
        f"pilot={len(pilot)} dev={len(dev)} test={len(test)} "
        f"written_to={outdir.resolve()}"
    )
