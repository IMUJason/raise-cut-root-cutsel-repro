from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def detect_resources() -> dict:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    cpu_logical = os.cpu_count()
    cpu_physical = psutil.cpu_count(logical=False) if psutil else None
    memory_total_gb = round(psutil.virtual_memory().total / 1024**3, 2) if psutil else None
    memory_available_gb = round(psutil.virtual_memory().available / 1024**3, 2) if psutil else None
    disk = shutil.disk_usage(Path.cwd())
    resource = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": cpu_logical,
        "cpu_count_physical": cpu_physical,
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "chip": _run("sysctl -n machdep.cpu.brand_string"),
        "recommendations": _recommend(cpu_logical or 1, memory_available_gb),
    }
    return resource


def _run(command: str) -> str | None:
    try:
        return subprocess.check_output(
            ["bash", "-lc", command],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _recommend(cpu_count: int, memory_available_gb: float | None) -> dict:
    workers = max(1, cpu_count - 2)
    memory_strategy = "moderate_memory"
    if memory_available_gb is not None and memory_available_gb < 8:
        memory_strategy = "chunked_processing"
    elif memory_available_gb is not None and memory_available_gb >= 16:
        memory_strategy = "in_memory_ok_for_pilot"
    return {
        "suggested_workers": workers,
        "memory_strategy": memory_strategy,
        "qaoa_statevector_limit_qubits": 16,
        "solver_recommendation": "Use exact Gurobi QUBO for small candidate pools, greedy/local search otherwise.",
    }


def write_resource_snapshot(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(detect_resources(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
