from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plan5.manifest import infer_format, parse_testset_lines, sha256_file, write_manifest_csv
from plan5.schemas import ManifestEntry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--exclude-testset", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--solu-file", required=True)
    parser.add_argument("--output-full", required=True)
    parser.add_argument("--output-sample", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--sample-size", type=int, default=140)
    parser.add_argument("--max-size-bytes", type=int, default=0)
    parser.add_argument("--source-root-id", default="D02")
    parser.add_argument("--family", default="unclassified")
    return parser.parse_args()


def parse_solu_instances(path: Path) -> set[str]:
    instances: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        tag = parts[0].lower()
        if tag not in {"=opt=", "=best=", "=inf=", "=unkn=", "=feas=", "=unbd="}:
            continue
        instances.add(parts[1])
    return instances


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    safe = frame.copy()
    for column in safe.columns:
        if pd.api.types.is_float_dtype(safe[column]):
            safe[column] = safe[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = [str(column) for column in safe.columns]
    rows = [headers]
    rows.extend([[str(value) for value in row] for row in safe.astype(object).fillna("").values.tolist()])
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    header_line = "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |" for row in rows[1:]]
    return "\n".join([header_line, divider, *body])


def build_entries(
    collection_root: Path,
    benchmark_root: Path,
    exclude_testset: Path,
    solu_instances: set[str],
    solu_filename: str,
    source_root_id: str,
    family: str,
) -> tuple[list[ManifestEntry], pd.DataFrame]:
    excluded_names = {Path(line).name for line in parse_testset_lines(exclude_testset)}
    benchmark_names = {path.name for path in benchmark_root.glob("*.mps.gz")}
    collection_paths = sorted(collection_root.glob("*.mps.gz"))

    rows: list[dict[str, object]] = []
    entries: list[ManifestEntry] = []
    for path in collection_paths:
        instance_id = path.stem.replace(".mps", "")
        in_benchmark = path.name in benchmark_names
        in_exclude_testset = path.name in excluded_names
        in_solu = instance_id in solu_instances
        rows.append(
            {
                "instance_id": instance_id,
                "basename": path.name,
                "abs_path": str(path.resolve()),
                "in_benchmark_v2": in_benchmark,
                "in_exclude_testset": in_exclude_testset,
                "in_solu_file": in_solu,
                "size_bytes": path.stat().st_size,
            }
        )
        if in_exclude_testset:
            continue
        entries.append(
            ManifestEntry(
                instance_id=instance_id,
                abs_path=path.resolve(),
                source_root_id=source_root_id,
                family=family,
                split="external_unseen_pool",
                file_sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                format=infer_format(path),
                notes=f"derived_from={collection_root.name};excluded={exclude_testset.name};solu={solu_filename}",
            )
        )
    audit = pd.DataFrame(rows).sort_values(["in_exclude_testset", "instance_id"], ascending=[False, True])
    return entries, audit


if __name__ == "__main__":
    args = parse_args()
    collection_root = Path(args.collection_root)
    benchmark_root = Path(args.benchmark_root)
    exclude_testset = Path(args.exclude_testset)
    solu_file = Path(args.solu_file)

    solu_instances = parse_solu_instances(solu_file)
    entries, audit = build_entries(
        collection_root=collection_root,
        benchmark_root=benchmark_root,
        exclude_testset=exclude_testset,
        solu_instances=solu_instances,
        solu_filename=solu_file.name,
        source_root_id=args.source_root_id,
        family=args.family,
    )
    output_full = Path(args.output_full)
    output_full.parent.mkdir(parents=True, exist_ok=True)
    full_frame = pd.DataFrame(
        [
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
            for entry in entries
        ]
    )
    if args.max_size_bytes > 0:
        full_frame = full_frame[full_frame["size_bytes"] <= args.max_size_bytes].copy()
    full_frame = full_frame.sort_values(["file_sha256", "instance_id"]).reset_index(drop=True)
    if len(full_frame) < args.sample_size:
        raise SystemExit(
            f"Only {len(full_frame)} unseen instances available after filtering, "
            f"fewer than requested sample size {args.sample_size}"
        )
    full_frame.to_csv(output_full, index=False)
    sample_frame = full_frame.head(args.sample_size).copy()
    sample_frame["split"] = "external_unseen_test"
    output_sample = Path(args.output_sample)
    output_sample.parent.mkdir(parents=True, exist_ok=True)
    sample_frame.to_csv(output_sample, index=False)

    audit_path = Path(args.output_audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "collection_root": str(collection_root.resolve()),
        "benchmark_root": str(benchmark_root.resolve()),
        "exclude_testset": str(exclude_testset.resolve()),
        "solu_file": str(solu_file.resolve()),
        "n_collection_instances": int(len(audit)),
        "n_excluded_benchmark_v2": int(audit["in_exclude_testset"].sum()),
        "n_external_unseen_pool": int(len(full_frame)),
        "n_external_unseen_sample": int(len(sample_frame)),
        "n_external_unseen_pool_with_solu": int(
            audit.loc[
                (~audit["in_exclude_testset"])
                & ((audit["size_bytes"] <= args.max_size_bytes) if args.max_size_bytes > 0 else True),
                "in_solu_file",
            ].sum()
        ),
        "n_external_unseen_sample_with_solu": int(sample_frame["instance_id"].isin(solu_instances).sum()),
        "max_size_bytes_filter": int(args.max_size_bytes),
    }
    lines = [
        "# External Unseen MIPLIB Manifest Audit",
        "",
        *(f"- {key}: `{value}`" for key, value in summary.items()),
        "",
        "## Sample Instances",
        "",
        dataframe_to_markdown(sample_frame[["instance_id", "size_bytes", "file_sha256"]]),
        "",
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"full={Path(output_full).resolve()} sample={output_sample.resolve()} "
        f"audit={audit_path.resolve()} n_pool={len(full_frame)} n_sample={len(sample_frame)}"
    )
