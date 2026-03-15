from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


RUN_REGISTRY_FIELDS = [
    "run_id",
    "timestamp_start",
    "timestamp_end",
    "stage",
    "selector_name",
    "backend_name",
    "candidate_pool_size_m",
    "budget_k",
    "depth_p",
    "dataset_manifest_path",
    "config_path",
    "command",
    "solver_name",
    "solver_version",
    "python_version",
    "machine_id",
    "git_commit_or_snapshot",
    "stdout_log_path",
    "stderr_log_path",
    "raw_result_path",
    "status",
]


def append_jsonl(path: str | Path, record: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def append_run_registry(path: str | Path, record: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_REGISTRY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in RUN_REGISTRY_FIELDS})
    return output


def make_run_id(stage: str, selector: str, split: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"P5_{stage}_{selector}_{split}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
