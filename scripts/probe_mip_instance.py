from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import gurobipy as gp  # type: ignore

    model = gp.read(args.instance)
    payload = {
        "instance": str(Path(args.instance).resolve()),
        "model_name": model.ModelName,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
        "num_bin_vars": model.NumBinVars,
        "num_int_vars": model.NumIntVars,
        "num_qconstrs": model.NumQConstrs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
