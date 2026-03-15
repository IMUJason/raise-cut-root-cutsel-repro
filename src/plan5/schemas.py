from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CutCandidate:
    cut_id: str
    coefficients: np.ndarray
    rhs: float
    family: str = "unknown"
    generator_name: str = "unknown"
    source_round: int = 0
    violation_raw: float = 0.0
    efficacy_raw: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", np.asarray(self.coefficients, dtype=float))


@dataclass(frozen=True)
class NodeContext:
    lp_cols: int
    lp_rows: int
    candidate_pool_size: int
    node_depth: int = 0
    objective_vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.objective_vector is not None:
            object.__setattr__(self, "objective_vector", np.asarray(self.objective_vector, dtype=float))


@dataclass(frozen=True)
class QUBOModel:
    linear: np.ndarray
    quadratic: np.ndarray
    budget_k: int
    penalty_rho: float
    cut_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear", np.asarray(self.linear, dtype=float))
        object.__setattr__(self, "quadratic", np.asarray(self.quadratic, dtype=float))

    @property
    def size(self) -> int:
        return int(self.linear.shape[0])


@dataclass(frozen=True)
class SelectionResult:
    selector_name: str
    backend_name: str
    selected_indices: tuple[int, ...]
    selected_cut_ids: tuple[str, ...]
    objective_value: float
    selector_latency_ms: float
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestEntry:
    instance_id: str
    abs_path: Path
    source_root_id: str
    family: str
    split: str
    file_sha256: str
    size_bytes: int
    format: str
    notes: str = ""
