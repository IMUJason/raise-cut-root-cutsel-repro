from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from .schemas import ManifestEntry


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_format(path: Path) -> str:
    suffixes = "".join(path.suffixes)
    return suffixes.lstrip(".") or "unknown"


def parse_testset_lines(testset_path: str | Path) -> list[str]:
    lines = []
    for raw in Path(testset_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_manifest_from_testset(
    testset_path: str | Path,
    instances_root: str | Path,
    source_root_id: str = "D01",
    split: str = "benchmark",
    family: str = "unclassified",
) -> list[ManifestEntry]:
    root = Path(instances_root)
    entries: list[ManifestEntry] = []
    for line in parse_testset_lines(testset_path):
        basename = Path(line).name
        abs_path = root / basename
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing instance referenced by testset: {abs_path}")
        entries.append(
            ManifestEntry(
                instance_id=abs_path.stem.replace(".mps", ""),
                abs_path=abs_path.resolve(),
                source_root_id=source_root_id,
                family=family,
                split=split,
                file_sha256=sha256_file(abs_path),
                size_bytes=abs_path.stat().st_size,
                format=infer_format(abs_path),
                notes=f"derived_from={Path(testset_path).name}",
            )
        )
    return entries


def write_manifest_csv(entries: Iterable[ManifestEntry], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id",
        "abs_path",
        "source_root_id",
        "family",
        "split",
        "file_sha256",
        "size_bytes",
        "format",
        "notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "instance_id": entry.instance_id,
                    "abs_path": str(entry.abs_path),
                    "source_root_id": entry.source_root_id,
                    "family": entry.family,
                    "split": entry.split,
                    "file_sha256": entry.file_sha256,
                    "size_bytes": entry.size_bytes,
                    "format": entry.format,
                    "notes": entry.notes,
                }
            )
    return output
