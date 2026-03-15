from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .qubo import evaluate_qubo
from .schemas import QUBOModel, SelectionResult


def select_qaoa_inspired(
    qubo: QUBOModel,
    depth_p: int = 1,
    gamma_values: list[float] | None = None,
    beta_values: list[float] | None = None,
    max_qubits: int = 20,
    max_basis_states: int = 5000,
    warm_start_mode: str = "linear_softmax",
    warm_start_tau: float = 1.0,
    parameter_search: bool = True,
    local_refine: bool = True,
) -> SelectionResult:
    if qubo.size > max_qubits:
        raise ValueError(f"Constrained QAOA backend supports at most {max_qubits} qubits, got {qubo.size}.")

    start = time.perf_counter()
    basis = _enumerate_fixed_weight_basis(qubo.size, qubo.budget_k)
    basis_size = basis.shape[0]
    if basis_size > max_basis_states:
        raise ValueError(
            f"Feasible basis too large for current constrained simulator: "
            f"{basis_size} states > max_basis_states={max_basis_states}."
        )

    objectives = np.array([evaluate_qubo(qubo, x) for x in basis], dtype=float)
    mixer = _build_johnson_mixer(basis)
    initial_state = _build_initial_state(basis, qubo, warm_start_mode=warm_start_mode, warm_start_tau=warm_start_tau)

    if parameter_search and depth_p == 1 and (gamma_values is None or beta_values is None):
        gamma_values, beta_values, search_score = _grid_search_p1(initial_state, objectives, mixer)
    else:
        gamma_values = gamma_values or [0.35] * depth_p
        beta_values = beta_values or [0.25] * depth_p
        search_score = None

    if len(gamma_values) != depth_p or len(beta_values) != depth_p:
        raise ValueError("Gamma and beta lengths must match depth_p.")

    state = initial_state.copy()
    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        state = np.exp(-1j * gamma * objectives) * state
        state = expm_multiply((-1j * beta) * mixer, state)

    probabilities = np.abs(state) ** 2
    best_index = int(np.argmax(probabilities))
    sampled_x = basis[best_index]
    sampled_indices = tuple(int(i) for i in np.where(sampled_x == 1)[0])
    sampled_objective = float(objectives[best_index])

    if local_refine:
        refined_indices, refined_objective = _local_refine_subset(qubo, sampled_indices)
        selected_indices = refined_indices
        objective_value = refined_objective
        selected_cut_ids = tuple(qubo.cut_ids[i] for i in refined_indices)
        selector_name = "P5-QAOA-C"
        backend_name = "johnson_subspace_sparse+local"
    else:
        selected_indices = sampled_indices
        objective_value = sampled_objective
        selected_cut_ids = tuple(qubo.cut_ids[i] for i in sampled_indices)
        selector_name = "P5-QAOA-C"
        backend_name = "johnson_subspace_sparse"

    latency_ms = (time.perf_counter() - start) * 1000
    return SelectionResult(
        selector_name=selector_name,
        backend_name=backend_name,
        selected_indices=tuple(selected_indices),
        selected_cut_ids=selected_cut_ids,
        objective_value=float(objective_value),
        selector_latency_ms=latency_ms,
        metadata={
            "depth_p": depth_p,
            "gamma_values": list(map(float, gamma_values)),
            "beta_values": list(map(float, beta_values)),
            "basis_size": int(basis_size),
            "feasible_probability": 1.0,
            "best_probability": float(probabilities[best_index]),
            "expected_objective": float(np.dot(probabilities, objectives)),
            "sampled_objective": float(sampled_objective),
            "search_score": None if search_score is None else float(search_score),
            "warm_start_mode": warm_start_mode,
            "warm_start_tau": float(warm_start_tau),
            "local_refine": bool(local_refine),
        },
    )


def select_qaoa_unconstrained_baseline(
    qubo: QUBOModel,
    depth_p: int = 1,
    gamma_values: list[float] | None = None,
    beta_values: list[float] | None = None,
    max_qubits: int = 16,
) -> SelectionResult:
    if qubo.size > max_qubits:
        raise ValueError(f"Unconstrained QAOA baseline supports at most {max_qubits} qubits, got {qubo.size}.")

    start = time.perf_counter()
    gamma_values = gamma_values or [0.35] * depth_p
    beta_values = beta_values or [0.25] * depth_p
    if len(gamma_values) != depth_p or len(beta_values) != depth_p:
        raise ValueError("Gamma and beta lengths must match depth_p.")

    bitstrings = _enumerate_all_bitstrings(qubo.size)
    objectives = np.array([evaluate_qubo(qubo, x) for x in bitstrings], dtype=float)
    state = np.ones(2**qubo.size, dtype=np.complex128) / math.sqrt(2**qubo.size)
    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        state *= np.exp(-1j * gamma * objectives)
        for qubit in range(qubo.size):
            state = _apply_rx_layer(state, qubit, beta)

    probabilities = np.abs(state) ** 2
    feasible_indices = np.where(bitstrings.sum(axis=1) == qubo.budget_k)[0]
    if feasible_indices.size == 0:
        raise RuntimeError("No feasible exact-K states found.")
    ranked = sorted(
        feasible_indices.tolist(),
        key=lambda idx: (probabilities[idx], objectives[idx]),
        reverse=True,
    )
    best_index = ranked[0]
    best_x = bitstrings[best_index]
    selected = tuple(int(i) for i in np.where(best_x == 1)[0])
    feasible_probability = float(np.sum(probabilities[feasible_indices]))
    latency_ms = (time.perf_counter() - start) * 1000
    return SelectionResult(
        selector_name="P5-QAOA-U",
        backend_name="fullspace_rx_baseline",
        selected_indices=selected,
        selected_cut_ids=tuple(qubo.cut_ids[i] for i in selected),
        objective_value=float(objectives[best_index]),
        selector_latency_ms=latency_ms,
        metadata={
            "depth_p": depth_p,
            "gamma_values": list(map(float, gamma_values)),
            "beta_values": list(map(float, beta_values)),
            "best_probability": float(probabilities[best_index]),
            "expected_objective": float(np.dot(probabilities, objectives)),
            "feasible_probability": feasible_probability,
        },
    )


def _enumerate_fixed_weight_basis(n_qubits: int, weight_k: int) -> np.ndarray:
    rows = np.zeros((math.comb(n_qubits, weight_k), n_qubits), dtype=np.int8)
    for idx, combo in enumerate(combinations(range(n_qubits), weight_k)):
        rows[idx, list(combo)] = 1
    return rows


def _build_johnson_mixer(basis: np.ndarray) -> csr_matrix:
    index = {tuple(row.tolist()): idx for idx, row in enumerate(basis)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for idx, state in enumerate(basis):
        ones = np.where(state == 1)[0]
        zeros = np.where(state == 0)[0]
        for out_idx in ones:
            for in_idx in zeros:
                neighbor = state.copy()
                neighbor[out_idx] = 0
                neighbor[in_idx] = 1
                jdx = index.get(tuple(neighbor.tolist()))
                if jdx is not None:
                    rows.append(idx)
                    cols.append(jdx)
                    data.append(1.0)
    size = basis.shape[0]
    return csr_matrix((data, (rows, cols)), shape=(size, size), dtype=float)


def _build_initial_state(
    basis: np.ndarray,
    qubo: QUBOModel,
    warm_start_mode: str,
    warm_start_tau: float,
) -> np.ndarray:
    dimension = basis.shape[0]
    if warm_start_mode == "uniform":
        return np.ones(dimension, dtype=np.complex128) / math.sqrt(dimension)
    if warm_start_mode == "linear_softmax":
        linear_scores = basis @ qubo.linear
        scaled = warm_start_tau * (linear_scores - np.max(linear_scores))
        weights = np.exp(scaled)
        probabilities = weights / np.sum(weights)
        return np.sqrt(probabilities).astype(np.complex128)
    raise ValueError(f"Unsupported warm_start_mode: {warm_start_mode}")


def _grid_search_p1(
    initial_state: np.ndarray,
    objectives: np.ndarray,
    mixer: csr_matrix,
) -> tuple[list[float], list[float], float]:
    gamma_grid = np.linspace(-1.2, 1.2, 13)
    beta_grid = np.linspace(-0.8, 0.8, 13)
    best_pair = (0.35, 0.25)
    best_score = float("-inf")
    for gamma in gamma_grid:
        phase_state = np.exp(-1j * gamma * objectives) * initial_state
        for beta in beta_grid:
            mixed_state = expm_multiply((-1j * beta) * mixer, phase_state)
            probabilities = np.abs(mixed_state) ** 2
            score = float(np.dot(probabilities, objectives))
            if score > best_score:
                best_score = score
                best_pair = (float(gamma), float(beta))
    return [best_pair[0]], [best_pair[1]], best_score


def _local_refine_subset(qubo: QUBOModel, selected_indices: tuple[int, ...]) -> tuple[tuple[int, ...], float]:
    selected = set(selected_indices)
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
    return tuple(sorted(selected)), float(current_value)


def _enumerate_all_bitstrings(n_qubits: int) -> np.ndarray:
    states = np.arange(2**n_qubits, dtype=np.uint32)
    return ((states[:, None] >> np.arange(n_qubits, dtype=np.uint32)) & 1).astype(np.int8)


def _apply_rx_layer(state: np.ndarray, qubit: int, beta: float) -> np.ndarray:
    updated = state.copy()
    stride = 1 << qubit
    block = stride << 1
    c = math.cos(beta)
    s = -1j * math.sin(beta)
    for start in range(0, state.shape[0], block):
        for offset in range(stride):
            i0 = start + offset
            i1 = i0 + stride
            a0 = state[i0]
            a1 = state[i1]
            updated[i0] = c * a0 + s * a1
            updated[i1] = s * a0 + c * a1
    return updated
