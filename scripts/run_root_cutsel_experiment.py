from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from pyscipopt import Model
except ModuleNotFoundError as exc:
    Model = None
    _PYSCIPOPT_IMPORT_ERROR = exc
else:
    _PYSCIPOPT_IMPORT_ERROR = None

from plan5.logging_utils import append_jsonl, append_run_registry, make_run_id
from plan5.scip_cutsel import (
    MaxEfficacyCutsel,
    Plan5AdaptiveCutsel,
    Plan5ContextCutsel,
    Plan5InteractionCutsel,
    ProbeRootCutsel,
    RAISECutSRCutsel,
    RAISECutsel,
    route_probe_to_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--run-registry", required=True)
    parser.add_argument("--cutsel-log", required=True)
    parser.add_argument("--modes", nargs="+", default=["default", "efficacy", "plan5"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--node-limit", type=int, default=1)
    return parser.parse_args()


def build_model(instance_path: str, time_limit: float, node_limit: int) -> Model:
    if Model is None:
        raise RuntimeError(
            "PySCIPOpt is required for solver reruns. Install the optional dependency set with "
            "`python -m pip install -e .[solver]` after configuring SCIP."
        ) from _PYSCIPOPT_IMPORT_ERROR
    model = Model()
    model.hideOutput(True)
    model.readProblem(instance_path)
    model.setParam("limits/time", time_limit)
    model.setParam("limits/nodes", node_limit)
    model.setParam("display/verblevel", 0)
    model.setParam("separating/maxroundsroot", 8)
    model.setParam("separating/maxcutsroot", 200)
    model.setParam("randomization/randomseedshift", 0)
    return model


def activate_builtin_cutsel(model: Model, target: str) -> None:
    for name in ("hybrid", "dynamic", "ensemble"):
        model.setParam(f"cutselection/{name}/priority", 9_000 if name == target else 0)


def attach_cutsel(model: Model, mode: str, cutsel_log: Path, instance_id: str, run_id: str):
    if mode in {"scip_hybrid", "scip_dynamic", "scip_ensemble"}:
        activate_builtin_cutsel(model, mode.replace("scip_", ""))
        return None
    if mode == "efficacy":
        cutsel = MaxEfficacyCutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
        model.includeCutsel(cutsel, "efficacycutsel", "efficacy cut selector baseline", 5_000_000)
        return cutsel
    if mode == "plan5":
        cutsel = Plan5InteractionCutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
        model.includeCutsel(cutsel, "plan5cutsel", "plan5 interaction-aware cut selector", 5_000_000)
        return cutsel
    if mode == "plan5_adaptive":
        cutsel = Plan5AdaptiveCutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
        model.includeCutsel(cutsel, "plan5adaptivecutsel", "plan5 adaptive interaction-aware cut selector", 5_000_000)
        return cutsel
    if mode == "raise_cut":
        cutsel = RAISECutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
        model.includeCutsel(cutsel, "raisecutsel", "regime-adaptive interaction-aware cut selector", 5_000_000)
        return cutsel
    if mode == "raise_cut_sr":
        cutsel = RAISECutSRCutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
        model.includeCutsel(cutsel, "raisecutsrcutsel", "stateful regime-adaptive interaction-aware cut selector", 5_000_000)
        return cutsel
    if mode == "plan5_context":
        cutsel = Plan5ContextCutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
        model.includeCutsel(cutsel, "plan5contextcutsel", "plan5 context-aware cut selector", 5_000_000)
        return cutsel
    return None


def run_portfolio(
    instance_path: str,
    time_limit: float,
    node_limit: int,
    cutsel_log: Path,
    instance_id: str,
    run_id: str,
    route_policy: str = "raise_portfolio",
):
    probe_model = build_model(instance_path, time_limit, node_limit)
    probe_cutsel = ProbeRootCutsel(log_path=cutsel_log, instance_id=instance_id, run_id=run_id)
    probe_model.includeCutsel(probe_cutsel, "raiseprobecutsel", "portfolio probe cut selector", 5_000_001)
    probe_model.optimize()
    probe_summary = probe_cutsel.probe_summary or {}
    routed_mode, route_reason = route_probe_to_decision(probe_summary, policy_name=route_policy)
    model = build_model(instance_path, time_limit, node_limit)
    cutsel = attach_cutsel(model, routed_mode, cutsel_log, instance_id, run_id)
    model.optimize()
    return model, cutsel, routed_mode, route_reason, probe_summary


def safe_metric(model: Model, getter, default):
    try:
        stage_name = str(model.getStageName()).upper()
    except Exception:
        stage_name = "UNKNOWN"
    if getter.__name__ == "getNCutsApplied" and "PRESOLV" in stage_name:
        return default
    try:
        return getter()
    except Exception:
        return default


if __name__ == "__main__":
    args = parse_args()
    manifest = pd.read_csv(args.manifest).sort_values("size_bytes").head(args.limit)
    output_jsonl = Path(args.output_jsonl)
    run_registry = Path(args.run_registry)
    cutsel_log = Path(args.cutsel_log)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cutsel_log.parent.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        instance_id = row["instance_id"]
        instance_path = row["abs_path"]
        for mode in args.modes:
            started_at = datetime.now()
            run_id = make_run_id("root", f"scip_{mode}", row["split"], started_at)
            portfolio_routed_mode = ""
            portfolio_probe = {}
            portfolio_route_reason = ""
            if mode.startswith("raise_portfolio"):
                model, cutsel, portfolio_routed_mode, portfolio_route_reason, portfolio_probe = run_portfolio(
                    instance_path=instance_path,
                    time_limit=args.time_limit,
                    node_limit=args.node_limit,
                    cutsel_log=cutsel_log,
                    instance_id=instance_id,
                    run_id=run_id,
                    route_policy=mode,
                )
            else:
                model = build_model(instance_path, args.time_limit, args.node_limit)
                cutsel = attach_cutsel(model, mode, cutsel_log, instance_id, run_id)
                model.optimize()
            status = str(safe_metric(model, model.getStatus, "unknown"))
            result_record = {
                "run_id": run_id,
                "instance_id": instance_id,
                "mode": mode,
                "stage": "root",
                "status": status,
                "nnodes": int(safe_metric(model, model.getNNodes, 0)),
                "dualbound": float(safe_metric(model, model.getDualbound, 0.0)),
                "primalbound": float(safe_metric(model, model.getPrimalbound, 1e20)),
                "gap": float(safe_metric(model, model.getGap, 1e20)) if status != "unknown" else None,
                "n_cuts_applied": int(safe_metric(model, model.getNCutsApplied, 0)),
                "n_lp_iterations": int(safe_metric(model, model.getNLPIterations, 0)),
                "solving_time": float(safe_metric(model, model.getSolvingTime, 0.0)),
                "root_rounds_logged": 0 if cutsel is None else int(cutsel.root_rounds),
                "root_candidates_total": 0 if cutsel is None else int(cutsel.total_root_candidates),
                "root_selected_total": 0 if cutsel is None else int(cutsel.total_root_selected),
                "last_root_selected_names": [] if cutsel is None else list(cutsel.last_root_selected_names),
                "portfolio_routed_mode": portfolio_routed_mode,
                "portfolio_route_policy": mode if mode.startswith("raise_portfolio") else "",
                "portfolio_route_reason": portfolio_route_reason,
                "portfolio_probe_candidate_count": int(portfolio_probe.get("candidate_count", 0) or 0),
                "portfolio_probe_mean_obj_parallelism": float(portfolio_probe.get("mean_obj_parallelism", 0.0) or 0.0),
                "portfolio_probe_regime": str(portfolio_probe.get("regime_name", "")),
                "portfolio_probe_dominant_family": str(portfolio_probe.get("dominant_family", "")),
                "portfolio_probe_dominant_share": float(portfolio_probe.get("dominant_share", 0.0) or 0.0),
                "portfolio_probe_family_count": int(portfolio_probe.get("family_count", 0) or 0),
                "manifest_path": str(Path(args.manifest).resolve()),
                "instance_path": instance_path,
            }
            append_jsonl(output_jsonl, result_record)
            append_run_registry(
                run_registry,
                {
                    "run_id": run_id,
                    "timestamp_start": started_at.isoformat(),
                    "timestamp_end": datetime.now().isoformat(),
                    "stage": "root",
                    "selector_name": f"scip-{mode}",
                    "backend_name": "scip-cutsel",
                    "candidate_pool_size_m": "",
                    "budget_k": "",
                    "depth_p": "",
                    "dataset_manifest_path": str(Path(args.manifest).resolve()),
                    "config_path": "",
                    "command": "python experiments/run_root_cutsel_experiment.py",
                    "solver_name": "SCIP",
                    "solver_version": "",
                    "python_version": platform.python_version(),
                    "machine_id": platform.node(),
                    "git_commit_or_snapshot": "",
                    "stdout_log_path": "",
                    "stderr_log_path": "",
                    "raw_result_path": str(output_jsonl.resolve()),
                    "status": "ok",
                },
            )
            print(json.dumps(result_record, ensure_ascii=False))
