from __future__ import annotations

import csv
import json
from pathlib import PurePosixPath
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def redact_path(value: str | None, prefix: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    name = PurePosixPath(text.replace("\\", "/")).name
    return f"{prefix}/{name}" if name else prefix


def sanitize_manifest_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        row["abs_path"] = redact_path(row.get("abs_path"), "INSTANCE_ROOT")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_results_jsonl(path: Path) -> None:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "manifest_path" in row:
                row["manifest_path"] = redact_path(row.get("manifest_path"), "MANIFEST_ROOT")
            if "instance_path" in row:
                row["instance_path"] = redact_path(row.get("instance_path"), "INSTANCE_ROOT")
            rows.append(row)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_registry_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    for row in rows:
        row["dataset_manifest_path"] = redact_path(row.get("dataset_manifest_path"), "MANIFEST_ROOT")
        row["raw_result_path"] = redact_path(row.get("raw_result_path"), "RESULT_ROOT")
        row["stdout_log_path"] = redact_path(row.get("stdout_log_path"), "LOG_ROOT")
        row["stderr_log_path"] = redact_path(row.get("stderr_log_path"), "LOG_ROOT")
        row["config_path"] = redact_path(row.get("config_path"), "CONFIG_ROOT")
        row["machine_id"] = "REDACTED"
        row["command"] = str(row.get("command", "")).replace("experiments/", "scripts/")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_shard_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for shard in payload.get("shards", []):
        if "manifest_path" in shard:
            shard["manifest_path"] = redact_path(shard.get("manifest_path"), "MANIFEST_ROOT")
        if "output_jsonl" in shard:
            shard["output_jsonl"] = redact_path(shard.get("output_jsonl"), "RESULT_ROOT")
        if "run_registry" in shard:
            shard["run_registry"] = redact_path(shard.get("run_registry"), "RESULT_ROOT")
        if "cutsel_log" in shard:
            shard["cutsel_log"] = redact_path(shard.get("cutsel_log"), "RESULT_ROOT")
        if "stdout_log" in shard:
            shard["stdout_log"] = redact_path(shard.get("stdout_log"), "LOG_ROOT")
        if "stderr_log" in shard:
            shard["stderr_log"] = redact_path(shard.get("stderr_log"), "LOG_ROOT")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for manifest_csv in sorted((DATA_ROOT / "manifests").glob("*.csv")):
        sanitize_manifest_csv(manifest_csv)

    for result_dir in sorted((DATA_ROOT / "results").iterdir()):
        if not result_dir.is_dir():
            continue
        for jsonl_path in result_dir.glob("*_results_merged.jsonl"):
            sanitize_results_jsonl(jsonl_path)
        for registry_csv in result_dir.glob("*_run_registry_merged.csv"):
            sanitize_registry_csv(registry_csv)
        for shard_manifest in result_dir.glob("*_shard_manifest.json"):
            sanitize_shard_manifest(shard_manifest)

    print(f"sanitized public release artifacts under {DATA_ROOT}")
