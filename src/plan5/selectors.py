from __future__ import annotations

import time

import numpy as np

from .qubo import evaluate_qubo
from .schemas import QUBOModel, SelectionResult


def select_topk_linear(qubo: QUBOModel) -> SelectionResult:
    start = time.perf_counter()
    ranking = np.argsort(qubo.linear)[::-1][: qubo.budget_k]
    x = np.zeros(qubo.size, dtype=int)
    x[ranking] = 1
    latency_ms = (time.perf_counter() - start) * 1000
    return SelectionResult(
        selector_name="P5-L",
        backend_name="linear",
        selected_indices=tuple(sorted(int(i) for i in ranking)),
        selected_cut_ids=tuple(qubo.cut_ids[int(i)] for i in sorted(ranking)),
        objective_value=evaluate_qubo(qubo, x),
        selector_latency_ms=latency_ms,
    )


def select_qubo_classical(
    qubo: QUBOModel,
    strategy: str = "auto",
    exact_threshold: int = 32,
) -> SelectionResult:
    if strategy == "auto":
        strategy = "exact" if qubo.size <= exact_threshold else "greedy_local"
    if strategy == "exact":
        exact = _select_qubo_exact_gurobi(qubo)
        if exact is not None:
            return exact
        strategy = "greedy_local"
    if strategy == "greedy":
        return _greedy_selection(qubo)
    if strategy == "greedy_local":
        return _local_refine(qubo, _greedy_selection(qubo))
    raise ValueError(f"Unsupported strategy: {strategy}")


def _greedy_selection(qubo: QUBOModel) -> SelectionResult:
    start = time.perf_counter()
    selected: list[int] = []
    remaining = set(range(qubo.size))
    while len(selected) < qubo.budget_k:
        best_idx = None
        best_value = float("-inf")
        for idx in remaining:
            trial = selected + [idx]
            x = np.zeros(qubo.size, dtype=int)
            x[trial] = 1
            value = evaluate_qubo(qubo, x)
            if value > best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    x = np.zeros(qubo.size, dtype=int)
    x[selected] = 1
    latency_ms = (time.perf_counter() - start) * 1000
    return SelectionResult(
        selector_name="P5-QUBO",
        backend_name="greedy",
        selected_indices=tuple(sorted(selected)),
        selected_cut_ids=tuple(qubo.cut_ids[i] for i in sorted(selected)),
        objective_value=evaluate_qubo(qubo, x),
        selector_latency_ms=latency_ms,
    )


def _local_refine(qubo: QUBOModel, initial: SelectionResult) -> SelectionResult:
    start = time.perf_counter()
    selected = set(initial.selected_indices)
    current_x = np.zeros(qubo.size, dtype=int)
    current_x[list(selected)] = 1
    current_value = evaluate_qubo(qubo, current_x)
    improved = True
    while improved:
        improved = False
        for out_idx in list(selected):
            for in_idx in range(qubo.size):
                if in_idx in selected:
                    continue
                trial = set(selected)
                trial.remove(out_idx)
                trial.add(in_idx)
                x = np.zeros(qubo.size, dtype=int)
                x[list(trial)] = 1
                value = evaluate_qubo(qubo, x)
                if value > current_value + 1e-9:
                    selected = trial
                    current_value = value
                    improved = True
                    break
            if improved:
                break
    latency_ms = initial.selector_latency_ms + (time.perf_counter() - start) * 1000
    return SelectionResult(
        selector_name="P5-QUBO",
        backend_name="greedy_local",
        selected_indices=tuple(sorted(selected)),
        selected_cut_ids=tuple(qubo.cut_ids[i] for i in sorted(selected)),
        objective_value=current_value,
        selector_latency_ms=latency_ms,
    )


def _select_qubo_exact_gurobi(qubo: QUBOModel) -> SelectionResult | None:
    try:
        import gurobipy as gp  # type: ignore
    except Exception:
        return None

    start = time.perf_counter()
    model = gp.Model("plan5_qubo")
    model.Params.OutputFlag = 0
    x = model.addVars(qubo.size, vtype=gp.GRB.BINARY, name="x")
    model.addConstr(gp.quicksum(x[i] for i in range(qubo.size)) == qubo.budget_k)
    objective = gp.quicksum(qubo.linear[i] * x[i] for i in range(qubo.size))
    objective += gp.quicksum(
        0.5 * qubo.quadratic[i, j] * x[i] * x[j]
        for i in range(qubo.size)
        for j in range(qubo.size)
    )
    model.setObjective(objective, gp.GRB.MAXIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        return None
    selected = tuple(sorted(i for i in range(qubo.size) if x[i].X > 0.5))
    latency_ms = (time.perf_counter() - start) * 1000
    return SelectionResult(
        selector_name="P5-QUBO",
        backend_name="exact_gurobi",
        selected_indices=selected,
        selected_cut_ids=tuple(qubo.cut_ids[i] for i in selected),
        objective_value=float(model.ObjVal),
        selector_latency_ms=latency_ms,
    )
