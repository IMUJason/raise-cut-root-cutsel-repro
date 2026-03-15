from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    from pyscipopt import SCIP_RESULT
    from pyscipopt.scip import Cutsel, Row
except ModuleNotFoundError:
    class _SCIPResult:
        SUCCESS = "SUCCESS"
        DIDNOTFIND = "DIDNOTFIND"

    SCIP_RESULT = _SCIPResult()

    class Cutsel:
        pass

    class Row:
        pass

from .logging_utils import append_jsonl


FAMILY_PATTERN = re.compile(r"[A-Za-z]+")


def normalize_cut_family_name(row_name: str) -> str:
    token = str(row_name).split("_", 1)[0]
    match = FAMILY_PATTERN.match(token)
    if match is None:
        return token.lower() if token else "unknown"
    return match.group(0).lower()


def compute_family_rarity(families: list[str]) -> dict[str, float]:
    counts = Counter(families)
    total = max(1, len(families))
    return {family: 1.0 - (count / total) for family, count in counts.items()}


@dataclass(frozen=True)
class RegimeSignal:
    regime_name: str
    dominant_family: str
    dominant_share: float
    probe_count: int
    family_count: int


@dataclass(frozen=True)
class RegimeConfig:
    candidate_cap: int
    min_selected_per_round: int
    budget_floor_ratio: float
    relative_marginal_threshold: float
    absolute_marginal_threshold: float
    max_nonpositive_streak: int
    dominant_family_bonus: float
    family_rarity_weight: float
    same_family_bonus: float
    cross_family_bonus: float
    dominant_pair_bonus: float
    dominant_quota_ratio: float
    dominant_quota_min: int
    quota_bonus: float


@dataclass
class CutRoundRecord:
    round_index: int
    root: bool
    candidate_count: int
    forcedcut_count: int
    maxnselectedcuts: int
    selected_count: int
    selector_name: str
    selected_cut_names: list[str]
    mean_efficacy: float
    mean_obj_parallelism: float
    objective_value: float | None = None
    regime_name: str | None = None
    dominant_family: str | None = None
    dominant_family_share: float | None = None
    candidate_cap_active: int | None = None
    budget_floor: int | None = None
    dominant_quota: int | None = None


class LoggingCutsel(Cutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        max_selected_cap: int = 50,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.instance_id = instance_id
        self.run_id = run_id
        self.max_selected_cap = max_selected_cap
        self.root_rounds = 0
        self.total_root_candidates = 0
        self.total_root_selected = 0
        self.last_root_selected_names: list[str] = []

    def _log_round(self, record: CutRoundRecord) -> None:
        if self.log_path is None:
            return
        append_jsonl(
            self.log_path,
            {
                "run_id": self.run_id,
                "instance_id": self.instance_id,
                "stage": "root" if record.root else "nonroot",
                "round_index": record.round_index,
                "selector_name": record.selector_name,
                "candidate_count": record.candidate_count,
                "forcedcut_count": record.forcedcut_count,
                "maxnselectedcuts": record.maxnselectedcuts,
                "selected_count": record.selected_count,
                "selected_cut_names": record.selected_cut_names,
                "mean_efficacy": record.mean_efficacy,
                "mean_obj_parallelism": record.mean_obj_parallelism,
                "objective_value": record.objective_value,
                "regime_name": record.regime_name,
                "dominant_family": record.dominant_family,
                "dominant_family_share": record.dominant_family_share,
                "candidate_cap_active": record.candidate_cap_active,
                "budget_floor": record.budget_floor,
                "dominant_quota": record.dominant_quota,
            },
        )

    def _row_name(self, row: Row, fallback: str) -> str:
        name = getattr(row, "name", None)
        if name is None or str(name).strip() == "":
            return fallback
        return str(name)


class MaxEfficacyCutsel(LoggingCutsel):
    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        scip = self.model
        scores = [float(scip.getCutEfficacy(cut)) for cut in cuts]
        rankings = sorted(range(len(cuts)), key=lambda idx: scores[idx], reverse=True)
        sorted_cuts = [cuts[idx] for idx in rankings]
        nselected = min(maxnselectedcuts, len(cuts), self.max_selected_cap)
        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += nselected
            self.last_root_selected_names = [self._row_name(sorted_cuts[i], f"cut_{i}") for i in range(nselected)]
            objpars = [float(scip.getRowObjParallelism(cut)) for cut in cuts] if cuts else []
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=nselected,
                    selector_name="B3-efficacy-cutsel",
                    selected_cut_names=self.last_root_selected_names,
                    mean_efficacy=_mean_or_zero(scores),
                    mean_obj_parallelism=_mean_or_zero(objpars),
                    objective_value=float(sum(scores[idx] for idx in rankings[:nselected])) if scores else 0.0,
                )
            )
        return {"cuts": sorted_cuts, "nselectedcuts": nselected, "result": SCIP_RESULT.SUCCESS}


class Plan5InteractionCutsel(LoggingCutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        max_selected_cap: int = 50,
        linear_weights: dict[str, float] | None = None,
        pair_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(log_path=log_path, instance_id=instance_id, run_id=run_id, max_selected_cap=max_selected_cap)
        self.linear_weights = linear_weights or {
            "efficacy": 0.65,
            "obj_parallelism": 0.25,
            "density": -0.10,
        }
        self.pair_weights = pair_weights or {
            "parallelism": -0.30,
            "overlap": -0.15,
            "diversity": 0.05,
        }

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if len(cuts) == 0:
            return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

        feature_records = [self._single_row_features(cut) for cut in cuts]
        linear_scores = self._linear_scores(feature_records)
        pair_matrix = self._pair_matrix(cuts, feature_records)
        nselected = min(maxnselectedcuts, len(cuts), self.max_selected_cap)
        selected = self._greedy_pairwise_select(linear_scores, pair_matrix, nselected)
        selected = self._local_refine(selected, linear_scores, pair_matrix)
        selected_set = set(selected)
        remaining = [idx for idx in np.argsort(linear_scores)[::-1].tolist() if idx not in selected_set]
        order = list(selected) + remaining
        sorted_cuts = [cuts[idx] for idx in order]
        objective_value = self._evaluate_subset(selected, linear_scores, pair_matrix)

        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += nselected
            self.last_root_selected_names = [self._row_name(cuts[i], f"cut_{i}") for i in selected]
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=nselected,
                    selector_name="P5-root-cutsel",
                    selected_cut_names=self.last_root_selected_names,
                    mean_efficacy=_mean_or_zero([record["efficacy"] for record in feature_records]),
                    mean_obj_parallelism=_mean_or_zero([record["obj_parallelism"] for record in feature_records]),
                    objective_value=float(objective_value),
                )
            )
        return {"cuts": sorted_cuts, "nselectedcuts": nselected, "result": SCIP_RESULT.SUCCESS}

    def _single_row_features(self, row: Row) -> dict[str, Any]:
        scip = self.model
        cols = row.getCols()
        vals = row.getVals()
        support = tuple(sorted(int(col.getLPPos()) for col in cols if col.getLPPos() >= 0))
        support_size = int(row.getNNonz())
        n_lp_cols = max(1, scip.getNLPCols())
        support_density = support_size / n_lp_cols
        row_name = self._row_name(row, "cut")
        nonzero_vals = [abs(float(v)) for v in vals if abs(float(v)) > 1e-12]
        dynamism = (max(nonzero_vals) / min(nonzero_vals)) if nonzero_vals else 1.0
        try:
            best_sol = scip.getBestSol()
            cutoff_distance = float(scip.getCutLPSolCutoffDistance(row, best_sol)) if best_sol is not None else 0.0
        except Exception:
            cutoff_distance = 0.0
        cons_origin = str(row.getOrigintype())
        int_support_ratio = float(scip.getRowNumIntCols(row)) / support_size if support_size else 0.0
        efficacy = float(scip.getCutEfficacy(row))
        obj_parallelism = float(scip.getRowObjParallelism(row))
        return {
            "row": row,
            "row_name": row_name,
            "cut_family": normalize_cut_family_name(row_name),
            "support": support,
            "support_size": support_size,
            "support_density": support_density,
            "efficacy": efficacy,
            "obj_parallelism": obj_parallelism,
            "expected_improvement": efficacy * max(obj_parallelism, 0.0),
            "norm": float(row.getNorm()),
            "lhs": float(row.getLhs()),
            "rhs": float(row.getRhs()),
            "origin_type": str(row.getOrigintype()),
            "cons_origin": cons_origin,
            "values": tuple(float(v) for v in vals),
            "dynamism": float(dynamism),
            "cutoff_distance": max(cutoff_distance, 0.0),
            "int_support_ratio": float(int_support_ratio),
            "is_global_cutpool": bool(row.isInGlobalCutpool()),
            "is_local": bool(row.isLocal()),
            "is_removable": bool(row.isRemovable()),
        }

    def _linear_scores(self, feature_records: list[dict[str, Any]]) -> np.ndarray:
        efficacy = _robust_scale(np.array([record["efficacy"] for record in feature_records], dtype=float))
        objpar = _robust_scale(np.array([record["obj_parallelism"] for record in feature_records], dtype=float))
        density = np.array([record["support_density"] for record in feature_records], dtype=float)
        return (
            self.linear_weights["efficacy"] * efficacy
            + self.linear_weights["obj_parallelism"] * objpar
            + self.linear_weights["density"] * density
        )

    def _pair_matrix(self, cuts, feature_records: list[dict[str, Any]]) -> np.ndarray:
        scip = self.model
        ncuts = len(cuts)
        matrix = np.zeros((ncuts, ncuts), dtype=float)
        for i in range(ncuts):
            support_i = set(feature_records[i]["support"])
            for j in range(i + 1, ncuts):
                support_j = set(feature_records[j]["support"])
                overlap = _support_overlap_ratio(support_i, support_j)
                try:
                    parallelism = float(scip.getRowParallelism(cuts[i], cuts[j]))
                except Exception:
                    parallelism = 0.0
                diversity = 1.0 if feature_records[i]["origin_type"] != feature_records[j]["origin_type"] else 0.0
                score = (
                    self.pair_weights["parallelism"] * parallelism
                    + self.pair_weights["overlap"] * overlap
                    + self.pair_weights["diversity"] * diversity
                )
                matrix[i, j] = score
                matrix[j, i] = score
        return matrix

    def _greedy_pairwise_select(self, linear_scores: np.ndarray, pair_matrix: np.ndarray, nselected: int) -> tuple[int, ...]:
        selected: list[int] = []
        remaining = set(range(linear_scores.shape[0]))
        while len(selected) < nselected and remaining:
            best_idx = None
            best_value = float("-inf")
            for idx in remaining:
                marginal = float(linear_scores[idx])
                if selected:
                    marginal += float(np.sum(pair_matrix[idx, selected]))
                if marginal > best_value:
                    best_value = marginal
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(int(best_idx))
            remaining.remove(int(best_idx))
        return tuple(selected)

    def _local_refine(self, selected: tuple[int, ...], linear_scores: np.ndarray, pair_matrix: np.ndarray) -> tuple[int, ...]:
        selected_set = set(selected)
        if len(selected_set) <= 1:
            return tuple(sorted(selected_set))
        current_value = self._evaluate_subset(selected, linear_scores, pair_matrix)
        improved = True
        while improved:
            improved = False
            for out_idx in list(selected_set):
                for in_idx in range(linear_scores.shape[0]):
                    if in_idx in selected_set:
                        continue
                    trial = set(selected_set)
                    trial.remove(out_idx)
                    trial.add(in_idx)
                    value = self._evaluate_subset(tuple(sorted(trial)), linear_scores, pair_matrix)
                    if value > current_value + 1e-9:
                        selected_set = trial
                        current_value = value
                        improved = True
                        break
                if improved:
                    break
        return tuple(sorted(selected_set))

    def _evaluate_subset(self, selected: tuple[int, ...] | list[int], linear_scores: np.ndarray, pair_matrix: np.ndarray) -> float:
        selected_list = list(selected)
        if not selected_list:
            return 0.0
        value = float(np.sum(linear_scores[selected_list]))
        if len(selected_list) >= 2:
            subm = pair_matrix[np.ix_(selected_list, selected_list)]
            value += 0.5 * float(np.sum(subm))
        return value


class Plan5AdaptiveCutsel(Plan5InteractionCutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        max_selected_cap: int = 50,
        candidate_cap: int = 140,
        min_selected_per_round: int = 4,
        relative_marginal_threshold: float = 0.30,
        absolute_marginal_threshold: float = 0.03,
        max_nonpositive_streak: int = 0,
        linear_weights: dict[str, float] | None = None,
        pair_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            log_path=log_path,
            instance_id=instance_id,
            run_id=run_id,
            max_selected_cap=max_selected_cap,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
        )
        self.candidate_cap = candidate_cap
        self.min_selected_per_round = min_selected_per_round
        self.relative_marginal_threshold = relative_marginal_threshold
        self.absolute_marginal_threshold = absolute_marginal_threshold
        self.max_nonpositive_streak = max_nonpositive_streak

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if len(cuts) == 0:
            return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

        feature_records = [self._single_row_features(cut) for cut in cuts]
        linear_scores = self._linear_scores(feature_records)
        top_indices = np.argsort(linear_scores)[::-1][: min(self.candidate_cap, len(cuts))].tolist()
        working_cuts = [cuts[idx] for idx in top_indices]
        working_features = [feature_records[idx] for idx in top_indices]
        working_linear_scores = linear_scores[top_indices]
        pair_matrix = self._pair_matrix(working_cuts, working_features)
        nselected_cap = min(maxnselectedcuts, len(working_cuts), self.max_selected_cap)
        selected_local = greedy_adaptive_select(
            linear_scores=working_linear_scores,
            pair_matrix=pair_matrix,
            nselected_cap=nselected_cap,
            min_selected=self.min_selected_per_round,
            relative_marginal_threshold=self.relative_marginal_threshold,
            absolute_marginal_threshold=self.absolute_marginal_threshold,
            max_nonpositive_streak=self.max_nonpositive_streak,
        )
        selected_local = self._local_refine(selected_local, working_linear_scores, pair_matrix)
        selected_global = [top_indices[idx] for idx in selected_local]
        nselected = len(selected_global)
        selected_set = set(selected_global)
        remaining = [idx for idx in np.argsort(linear_scores)[::-1].tolist() if idx not in selected_set]
        order = selected_global + remaining
        sorted_cuts = [cuts[idx] for idx in order]
        objective_value = self._evaluate_subset(selected_local, working_linear_scores, pair_matrix)

        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += nselected
            self.last_root_selected_names = [self._row_name(cuts[i], f"cut_{i}") for i in selected_global]
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=nselected,
                    selector_name="P5-adaptive-cutsel",
                    selected_cut_names=self.last_root_selected_names,
                    mean_efficacy=_mean_or_zero([record["efficacy"] for record in feature_records]),
                    mean_obj_parallelism=_mean_or_zero([record["obj_parallelism"] for record in feature_records]),
                    objective_value=float(objective_value),
                )
            )
        return {"cuts": sorted_cuts, "nselectedcuts": nselected, "result": SCIP_RESULT.SUCCESS}


class ProbeRootCutsel(Plan5InteractionCutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        regime_probe_cap: int = 60,
        dominance_threshold: float = 0.55,
    ) -> None:
        super().__init__(log_path=log_path, instance_id=instance_id, run_id=run_id, max_selected_cap=0)
        self.regime_probe_cap = regime_probe_cap
        self.dominance_threshold = dominance_threshold
        self.probe_summary: dict[str, Any] | None = None
        self._interrupted = False

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if len(cuts) == 0:
            return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}
        feature_records = [self._single_row_features(cut) for cut in cuts]
        regime_signal = infer_candidate_regime(
            feature_records,
            probe_cap=self.regime_probe_cap,
            dominance_threshold=self.dominance_threshold,
        )
        mean_obj_parallelism = _mean_or_zero([float(record["obj_parallelism"]) for record in feature_records])
        mean_efficacy = _mean_or_zero([float(record["efficacy"]) for record in feature_records])
        self.probe_summary = {
            "candidate_count": len(cuts),
            "forcedcut_count": len(forcedcuts),
            "mean_efficacy": mean_efficacy,
            "mean_obj_parallelism": mean_obj_parallelism,
            "regime_name": regime_signal.regime_name,
            "dominant_family": regime_signal.dominant_family,
            "dominant_share": float(regime_signal.dominant_share),
            "family_count": int(regime_signal.family_count),
            "probe_count": int(regime_signal.probe_count),
        }
        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += 0
            self.last_root_selected_names = []
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=0,
                    selector_name="RAISE-probe",
                    selected_cut_names=[],
                    mean_efficacy=mean_efficacy,
                    mean_obj_parallelism=mean_obj_parallelism,
                    objective_value=0.0,
                    regime_name=f"{regime_signal.regime_name}:probe",
                    dominant_family=regime_signal.dominant_family,
                    dominant_family_share=float(regime_signal.dominant_share),
                    candidate_cap_active=len(cuts),
                    budget_floor=0,
                    dominant_quota=0,
                )
            )
            if not self._interrupted:
                self._interrupted = True
                self.model.interruptSolve()
        return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.SUCCESS}


class RAISECutsel(Plan5InteractionCutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        max_selected_cap: int = 50,
        regime_probe_cap: int = 60,
        dominance_threshold: float = 0.55,
        linear_weights: dict[str, float] | None = None,
        pair_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            log_path=log_path,
            instance_id=instance_id,
            run_id=run_id,
            max_selected_cap=max_selected_cap,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
        )
        self.regime_probe_cap = regime_probe_cap
        self.dominance_threshold = dominance_threshold
        self.interaction_candidate_cap = 140
        self.flowcover_candidate_cap = 160
        self.interaction_min_selected = 4
        self.flowcover_min_selected = 5
        self.relative_marginal_threshold = 0.30
        self.absolute_marginal_threshold = 0.03
        self.max_nonpositive_streak = 0

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if len(cuts) == 0:
            return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

        feature_records = [self._single_row_features(cut) for cut in cuts]
        regime_signal = infer_candidate_regime(
            feature_records,
            probe_cap=self.regime_probe_cap,
            dominance_threshold=self.dominance_threshold,
        )
        if self._use_dense_policy(regime_signal):
            policy_name = "dense"
            efficacy_scores = [float(record["efficacy"]) for record in feature_records]
            dense_order = sorted(range(len(cuts)), key=lambda idx: efficacy_scores[idx], reverse=True)
            nselected = min(maxnselectedcuts, len(cuts), self.max_selected_cap)
            selected_global = dense_order[:nselected]
            remaining = dense_order[nselected:]
            order = selected_global + remaining
            sorted_cuts = [cuts[idx] for idx in order]
            objective_value = float(sum(efficacy_scores[idx] for idx in selected_global)) if selected_global else 0.0
            active_candidate_cap = len(cuts)
            budget_floor = nselected
        else:
            policy_name = "interaction"
            linear_scores = self._linear_scores(feature_records)
            active_candidate_cap = min(
                self.flowcover_candidate_cap if regime_signal.regime_name == "flowcover" else self.interaction_candidate_cap,
                len(cuts),
            )
            top_indices = np.argsort(linear_scores)[::-1][:active_candidate_cap].tolist()
            working_cuts = [cuts[idx] for idx in top_indices]
            working_features = [feature_records[idx] for idx in top_indices]
            working_linear_scores = linear_scores[top_indices]
            pair_matrix = self._pair_matrix(working_cuts, working_features)
            nselected_cap = min(maxnselectedcuts, len(working_cuts), self.max_selected_cap)
            budget_floor = min(
                nselected_cap,
                self.flowcover_min_selected if regime_signal.regime_name == "flowcover" else self.interaction_min_selected,
            )
            selected_local = greedy_adaptive_select(
                linear_scores=working_linear_scores,
                pair_matrix=pair_matrix,
                nselected_cap=nselected_cap,
                min_selected=budget_floor,
                relative_marginal_threshold=self.relative_marginal_threshold,
                absolute_marginal_threshold=self.absolute_marginal_threshold,
                max_nonpositive_streak=self.max_nonpositive_streak,
            )
            selected_local = self._local_refine(selected_local, working_linear_scores, pair_matrix)
            selected_global = [top_indices[idx] for idx in selected_local]
            nselected = len(selected_global)
            selected_set = set(selected_global)
            remaining = [idx for idx in np.argsort(linear_scores)[::-1].tolist() if idx not in selected_set]
            order = selected_global + remaining
            sorted_cuts = [cuts[idx] for idx in order]
            objective_value = self._evaluate_subset(selected_local, working_linear_scores, pair_matrix)

        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += nselected
            self.last_root_selected_names = [self._row_name(cuts[i], f"cut_{i}") for i in selected_global]
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=nselected,
                    selector_name="RAISE-Cut",
                    selected_cut_names=self.last_root_selected_names,
                    mean_efficacy=_mean_or_zero([record["efficacy"] for record in feature_records]),
                    mean_obj_parallelism=_mean_or_zero([record["obj_parallelism"] for record in feature_records]),
                    objective_value=float(objective_value),
                    regime_name=f"{regime_signal.regime_name}:{policy_name}",
                    dominant_family=regime_signal.dominant_family,
                    dominant_family_share=float(regime_signal.dominant_share),
                    candidate_cap_active=active_candidate_cap,
                    budget_floor=budget_floor,
                    dominant_quota=0,
                )
            )
        return {"cuts": sorted_cuts, "nselectedcuts": nselected, "result": SCIP_RESULT.SUCCESS}

    def _use_dense_policy(self, regime_signal: RegimeSignal) -> bool:
        return regime_signal.regime_name not in {"flowcover", "other"}


class RAISECutSRCutsel(RAISECutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        max_selected_cap: int = 50,
        regime_probe_cap: int = 60,
        dominance_threshold: float = 0.55,
        strong_dominance_threshold: float = 0.75,
        linear_weights: dict[str, float] | None = None,
        pair_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            log_path=log_path,
            instance_id=instance_id,
            run_id=run_id,
            max_selected_cap=max_selected_cap,
            regime_probe_cap=regime_probe_cap,
            dominance_threshold=dominance_threshold,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
        )
        self.strong_dominance_threshold = strong_dominance_threshold
        self.decay_start_round = 5
        self.per_round_decay = 3
        self.history_penalty_weight = 0.03
        self.max_history_penalty = 8
        self.streak_penalty = 2
        self.base_dense_caps = {
            "mixed": 50,
            "gom": 48,
            "cmir": 46,
            "implbd": 48,
            "clique": 24,
            "default": 46,
        }
        self.floor_dense_caps = {
            "mixed": 24,
            "gom": 20,
            "cmir": 18,
            "implbd": 20,
            "clique": 8,
            "default": 18,
        }
        self.dominant_quota_ratio = 0.80
        self.dominant_quota_floor = 6
        self.dominant_quota_history_weight = 0.02
        self.max_quota_history_penalty = 4
        self.min_alt_candidates = 3
        self.context_linear_weights = {
            "efficacy": 0.55,
            "obj_parallelism": 0.20,
            "density": -0.08,
            "cutoff_distance": 0.10,
            "expected_improvement": 0.05,
            "integer_support": 0.04,
            "numerics": 0.04,
            "family_strength": 0.06,
            "local_penalty": -0.02,
        }
        self.context_pair_weights = {
            "parallelism": -0.30,
            "overlap": -0.15,
            "same_family": -0.06,
            "family_diversity": 0.02,
        }
        self.context_candidate_cap = 180
        self.context_max_selected_cap = 36
        self.context_min_selected = 8
        self.context_min_score_factor = 0.25
        self.context_good_score_factor = 0.92
        self.context_max_parallelism = 0.975
        self.context_good_max_parallelism = 0.92
        self.context_min_orthogonality_gain = 1e-4
        self.family_exposure_counts: Counter[str] = Counter()
        self.last_dense_dominant_family: str | None = None
        self.dense_dominant_streak = 0

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if len(cuts) == 0:
            return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

        feature_records = [self._single_row_features(cut) for cut in cuts]
        regime_signal = infer_candidate_regime(
            feature_records,
            probe_cap=self.regime_probe_cap,
            dominance_threshold=self.dominance_threshold,
        )
        round_index = self.root_rounds + 1 if root else 1
        dominant_history = int(self.family_exposure_counts.get(regime_signal.dominant_family, 0))
        mean_obj_parallelism = _mean_or_zero([float(record["obj_parallelism"]) for record in feature_records])
        if self._use_dense_policy(regime_signal):
            if self._use_conservative_dense(regime_signal, round_index, mean_obj_parallelism, len(cuts)):
                policy_name = "dense_context"
                (
                    selected_global,
                    order,
                    nselected,
                    objective_value,
                    active_candidate_cap,
                    budget_floor,
                ) = self._select_context_dense(cuts, feature_records, maxnselectedcuts, mean_obj_parallelism)
                sorted_cuts = [cuts[idx] for idx in order]
                dominant_quota = 0
            else:
                policy_name = "dense"
                efficacy_scores = [float(record["efficacy"]) for record in feature_records]
                dense_order = sorted(range(len(cuts)), key=lambda idx: efficacy_scores[idx], reverse=True)
                nselected = min(maxnselectedcuts, len(cuts), self.max_selected_cap)
                selected_global = dense_order[:nselected]
                remaining = dense_order[nselected:]
                order = selected_global + remaining
                sorted_cuts = [cuts[idx] for idx in order]
                objective_value = float(sum(efficacy_scores[idx] for idx in selected_global)) if selected_global else 0.0
                active_candidate_cap = len(cuts)
                budget_floor = nselected
                dominant_quota = 0
        else:
            policy_name = "interaction"
            linear_scores = self._linear_scores(feature_records)
            active_candidate_cap = min(
                self.flowcover_candidate_cap if regime_signal.regime_name == "flowcover" else self.interaction_candidate_cap,
                len(cuts),
            )
            top_indices = np.argsort(linear_scores)[::-1][:active_candidate_cap].tolist()
            working_cuts = [cuts[idx] for idx in top_indices]
            working_features = [feature_records[idx] for idx in top_indices]
            working_linear_scores = linear_scores[top_indices]
            pair_matrix = self._pair_matrix(working_cuts, working_features)
            nselected_cap = min(maxnselectedcuts, len(working_cuts), self.max_selected_cap)
            budget_floor = min(
                nselected_cap,
                self.flowcover_min_selected if regime_signal.regime_name == "flowcover" else self.interaction_min_selected,
            )
            selected_local = greedy_adaptive_select(
                linear_scores=working_linear_scores,
                pair_matrix=pair_matrix,
                nselected_cap=nselected_cap,
                min_selected=budget_floor,
                relative_marginal_threshold=self.relative_marginal_threshold,
                absolute_marginal_threshold=self.absolute_marginal_threshold,
                max_nonpositive_streak=self.max_nonpositive_streak,
            )
            selected_local = self._local_refine(selected_local, working_linear_scores, pair_matrix)
            selected_global = [top_indices[idx] for idx in selected_local]
            nselected = len(selected_global)
            selected_set = set(selected_global)
            remaining = [idx for idx in np.argsort(linear_scores)[::-1].tolist() if idx not in selected_set]
            order = selected_global + remaining
            sorted_cuts = [cuts[idx] for idx in order]
            objective_value = self._evaluate_subset(selected_local, working_linear_scores, pair_matrix)
            dominant_quota = 0

        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += nselected
            self.last_root_selected_names = [self._row_name(cuts[i], f"cut_{i}") for i in selected_global]
            self._update_root_history(feature_records, selected_global, regime_signal, policy_name)
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=nselected,
                    selector_name="RAISE-Cut-SR",
                    selected_cut_names=self.last_root_selected_names,
                    mean_efficacy=_mean_or_zero([record["efficacy"] for record in feature_records]),
                    mean_obj_parallelism=_mean_or_zero([record["obj_parallelism"] for record in feature_records]),
                    objective_value=float(objective_value),
                    regime_name=f"{regime_signal.regime_name}:{policy_name}",
                    dominant_family=regime_signal.dominant_family,
                    dominant_family_share=float(regime_signal.dominant_share),
                    candidate_cap_active=active_candidate_cap,
                    budget_floor=budget_floor,
                    dominant_quota=dominant_quota,
                )
            )
        return {"cuts": sorted_cuts, "nselectedcuts": nselected, "result": SCIP_RESULT.SUCCESS}

    def _use_conservative_dense(
        self,
        regime_signal: RegimeSignal,
        round_index: int,
        mean_obj_parallelism: float,
        candidate_count: int,
    ) -> bool:
        if regime_signal.regime_name == "clique":
            return True
        if regime_signal.dominant_share < self.strong_dominance_threshold:
            return False
        if mean_obj_parallelism <= 0.01 and candidate_count >= 400:
            return True
        if regime_signal.regime_name == "cmir":
            return mean_obj_parallelism <= 0.05 or round_index >= 4
        if regime_signal.regime_name in {"gom", "implbd"}:
            if mean_obj_parallelism <= 0.05:
                return True
            return candidate_count >= 250 and mean_obj_parallelism <= 0.08 and (round_index >= 4 or self.dense_dominant_streak >= 2)
        return candidate_count >= 250 and mean_obj_parallelism <= 0.08 and (round_index >= 5 or self.dense_dominant_streak >= 2)

    def _select_context_dense(
        self,
        cuts,
        feature_records: list[dict[str, Any]],
        maxnselectedcuts: int,
        mean_obj_parallelism: float,
    ) -> tuple[list[int], list[int], int, float, int, int]:
        original_linear_scores = self._contextual_linear_scores_sr(feature_records)
        active_candidate_cap = min(self.context_candidate_cap, len(cuts))
        top_indices = np.argsort(original_linear_scores)[::-1][:active_candidate_cap].tolist()
        working_cuts = [cuts[idx] for idx in top_indices]
        working_features = [feature_records[idx] for idx in top_indices]
        working_linear_scores = original_linear_scores[top_indices]
        pair_matrix = self._context_pair_matrix_sr(working_cuts, working_features)
        if mean_obj_parallelism <= 0.01 and active_candidate_cap >= 400:
            context_cap = 16
        elif mean_obj_parallelism <= 0.05 and active_candidate_cap >= 250:
            context_cap = 24
        else:
            context_cap = self.context_max_selected_cap
        selected_local = self._adaptive_pairwise_select_sr(
            working_cuts,
            working_features,
            working_linear_scores,
            pair_matrix,
            maxnselectedcuts,
            context_cap,
        )
        if len(selected_local) > 1:
            selected_local = self._local_refine(selected_local, working_linear_scores, pair_matrix)
        if not selected_local:
            fallback_count = min(1, len(top_indices), maxnselectedcuts, context_cap)
            selected_local = tuple(range(fallback_count))
        selected_global = [top_indices[idx] for idx in selected_local]
        nselected = len(selected_global)
        objective_value = self._evaluate_subset(selected_local, working_linear_scores, pair_matrix)
        selected_set = set(selected_global)
        remaining = [idx for idx in np.argsort(original_linear_scores)[::-1].tolist() if idx not in selected_set]
        order = selected_global + remaining
        budget_floor = min(self.context_min_selected, maxnselectedcuts, context_cap)
        return selected_global, order, nselected, objective_value, active_candidate_cap, budget_floor

    def _contextual_linear_scores_sr(self, feature_records: list[dict[str, Any]]) -> np.ndarray:
        family_strength = _family_strength_scores(feature_records)
        efficacy = _robust_scale(np.array([record["efficacy"] for record in feature_records], dtype=float))
        cutoff_distance = _robust_scale(
            np.log1p(np.array([max(record["cutoff_distance"], 0.0) for record in feature_records], dtype=float))
        )
        obj_parallelism = _robust_scale(np.array([record["obj_parallelism"] for record in feature_records], dtype=float))
        density = np.array([record["support_density"] for record in feature_records], dtype=float)
        expected_improvement = _robust_scale(
            np.array([record["expected_improvement"] for record in feature_records], dtype=float)
        )
        int_support = np.array([record["int_support_ratio"] for record in feature_records], dtype=float)
        numerics = _robust_scale(
            np.array([1.0 / np.log1p(max(1.0, record["dynamism"])) for record in feature_records], dtype=float)
        )
        local_cut = np.array([1.0 if record["is_local"] else 0.0 for record in feature_records], dtype=float)
        return (
            self.context_linear_weights["efficacy"] * efficacy
            + self.context_linear_weights["obj_parallelism"] * obj_parallelism
            + self.context_linear_weights["density"] * density
            + self.context_linear_weights["cutoff_distance"] * cutoff_distance
            + self.context_linear_weights["expected_improvement"] * expected_improvement
            + self.context_linear_weights["integer_support"] * int_support
            + self.context_linear_weights["numerics"] * numerics
            + self.context_linear_weights["family_strength"] * family_strength
            + self.context_linear_weights["local_penalty"] * local_cut
        )

    def _context_pair_matrix_sr(self, cuts, feature_records: list[dict[str, Any]]) -> np.ndarray:
        scip = self.model
        ncuts = len(cuts)
        matrix = np.zeros((ncuts, ncuts), dtype=float)
        for i in range(ncuts):
            support_i = set(feature_records[i]["support"])
            for j in range(i + 1, ncuts):
                support_j = set(feature_records[j]["support"])
                overlap = _support_overlap_ratio(support_i, support_j)
                try:
                    parallelism = float(scip.getRowParallelism(cuts[i], cuts[j]))
                except Exception:
                    parallelism = 0.0
                same_family = 1.0 if feature_records[i]["cut_family"] == feature_records[j]["cut_family"] else 0.0
                family_diversity = 1.0 - same_family
                score = (
                    self.context_pair_weights["parallelism"] * parallelism
                    + self.context_pair_weights["overlap"] * overlap
                    + self.context_pair_weights["same_family"] * same_family
                    + self.context_pair_weights["family_diversity"] * family_diversity
                )
                matrix[i, j] = score
                matrix[j, i] = score
        return matrix

    def _adaptive_pairwise_select_sr(
        self,
        cuts,
        feature_records: list[dict[str, Any]],
        linear_scores: np.ndarray,
        pair_matrix: np.ndarray,
        maxnselectedcuts: int,
        context_cap: int,
    ) -> tuple[int, ...]:
        nselected_cap = min(maxnselectedcuts, len(cuts), context_cap)
        if nselected_cap <= 0:
            return tuple()
        best_linear = float(np.max(linear_scores))
        selected: list[int] = []
        remaining = set(range(linear_scores.shape[0]))
        while remaining and len(selected) < nselected_cap:
            best_idx = None
            best_value = float("-inf")
            for idx in remaining:
                marginal = float(linear_scores[idx])
                if selected:
                    marginal += float(np.sum(pair_matrix[idx, selected]))
                if marginal > best_value:
                    best_value = marginal
                    best_idx = idx
            if best_idx is None:
                break
            if selected and len(selected) >= self.context_min_selected and best_value < self.context_min_score_factor * best_linear:
                break
            selected.append(int(best_idx))
            remaining.remove(int(best_idx))
            remaining = self._filter_parallel_cuts_sr(
                selected_idx=int(best_idx),
                remaining=remaining,
                cuts=cuts,
                feature_records=feature_records,
                linear_scores=linear_scores,
                best_linear=best_linear,
            )
        return tuple(selected)

    def _filter_parallel_cuts_sr(
        self,
        selected_idx: int,
        remaining: set[int],
        cuts,
        feature_records: list[dict[str, Any]],
        linear_scores: np.ndarray,
        best_linear: float,
    ) -> set[int]:
        scip = self.model
        selected_record = feature_records[selected_idx]
        selected_eff = max(selected_record["efficacy"], 1e-9)
        filtered_remaining: set[int] = set()
        for idx in remaining:
            record = feature_records[idx]
            try:
                parallelism = float(scip.getRowParallelism(cuts[selected_idx], cuts[idx]))
            except Exception:
                parallelism = 0.0
            candidate_eff = max(record["efficacy"], 1e-9)
            weighted_parallelism = parallelism * candidate_eff / selected_eff
            max_allowed = (
                self.context_good_max_parallelism
                if linear_scores[idx] >= self.context_good_score_factor * best_linear
                else self.context_max_parallelism
            )
            adaptive_limit = self._adaptive_parallelism_limit_sr(selected_eff, candidate_eff, parallelism, max_allowed)
            if weighted_parallelism >= 1.0 or parallelism > adaptive_limit:
                continue
            filtered_remaining.add(idx)
        return filtered_remaining

    def _adaptive_parallelism_limit_sr(
        self,
        selected_eff: float,
        candidate_eff: float,
        parallelism: float,
        fallback_limit: float,
    ) -> float:
        denominator = 2.0 * selected_eff * candidate_eff
        if denominator <= 1e-12:
            return fallback_limit
        numerator = (
            selected_eff**2
            + candidate_eff**2
            - (1.0 + self.context_min_orthogonality_gain) ** 2 * selected_eff**2 * max(0.0, 1.0 - parallelism**2)
        )
        adaptive = numerator / denominator
        return float(np.clip(max(fallback_limit, adaptive), -1.0, 1.0))

    def _resolve_dense_quota(
        self,
        dense_cap: int,
        families: list[str],
        regime_signal: RegimeSignal,
        dominant_history: int,
    ) -> int:
        if dense_cap <= 0 or regime_signal.dominant_share < self.strong_dominance_threshold:
            return 0
        dominant_count = sum(1 for family in families if family == regime_signal.dominant_family)
        alt_count = len(families) - dominant_count
        if alt_count < self.min_alt_candidates:
            return 0
        quota = int(round(self.dominant_quota_ratio * dense_cap))
        quota -= min(self.max_quota_history_penalty, int(round(self.dominant_quota_history_weight * dominant_history)))
        quota -= max(0, self.dense_dominant_streak - 1)
        quota = max(self.dominant_quota_floor, quota)
        return min(dense_cap, dominant_count, quota)

    def _update_root_history(
        self,
        feature_records: list[dict[str, Any]],
        selected_global: list[int],
        regime_signal: RegimeSignal,
        policy_name: str,
    ) -> None:
        selected_families = [str(feature_records[idx]["cut_family"]) for idx in selected_global]
        self.family_exposure_counts.update(selected_families)
        if policy_name.startswith("dense") and regime_signal.dominant_share >= self.strong_dominance_threshold:
            if regime_signal.dominant_family == self.last_dense_dominant_family:
                self.dense_dominant_streak += 1
            else:
                self.last_dense_dominant_family = regime_signal.dominant_family
                self.dense_dominant_streak = 1
            return
        self.last_dense_dominant_family = None
        self.dense_dominant_streak = 0

class Plan5ContextCutsel(Plan5InteractionCutsel):
    def __init__(
        self,
        log_path: str | Path | None = None,
        instance_id: str = "",
        run_id: str = "",
        max_selected_cap: int = 24,
        candidate_cap: int = 180,
        min_selected_per_round: int = 6,
        min_score_factor: float = 0.28,
        good_score_factor: float = 0.92,
        max_parallelism: float = 0.975,
        good_max_parallelism: float = 0.92,
        min_orthogonality_gain: float = 1e-4,
        linear_weights: dict[str, float] | None = None,
        pair_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            log_path=log_path,
            instance_id=instance_id,
            run_id=run_id,
            max_selected_cap=max_selected_cap,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
        )
        self.linear_weights = linear_weights or {
            "efficacy": 0.55,
            "obj_parallelism": 0.20,
            "density": -0.08,
            "cutoff_distance": 0.10,
            "expected_improvement": 0.05,
            "integer_support": 0.04,
            "numerics": 0.04,
            "family_strength": 0.06,
            "local_penalty": -0.02,
        }
        self.pair_weights = pair_weights or {
            "parallelism": -0.30,
            "overlap": -0.15,
            "same_family": -0.06,
            "family_diversity": 0.02,
        }
        self.candidate_cap = candidate_cap
        self.min_selected_per_round = min_selected_per_round
        self.min_score_factor = min_score_factor
        self.good_score_factor = good_score_factor
        self.max_parallelism = max_parallelism
        self.good_max_parallelism = good_max_parallelism
        self.min_orthogonality_gain = min_orthogonality_gain

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if len(cuts) == 0:
            return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

        feature_records = [self._single_row_features(cut) for cut in cuts]
        original_linear_scores = self._contextual_linear_scores(feature_records)

        top_indices = np.argsort(original_linear_scores)[::-1][: min(self.candidate_cap, len(cuts))].tolist()
        working_cuts = [cuts[idx] for idx in top_indices]
        working_features = [feature_records[idx] for idx in top_indices]
        working_linear_scores = original_linear_scores[top_indices]
        pair_matrix = self._context_pair_matrix(working_cuts, working_features)
        selected_local = self._adaptive_pairwise_select(
            working_cuts,
            working_features,
            working_linear_scores,
            pair_matrix,
            maxnselectedcuts,
        )
        if len(selected_local) > 1:
            selected_local = self._local_refine(selected_local, working_linear_scores, pair_matrix)
        if not selected_local:
            fallback_count = min(1, len(top_indices), maxnselectedcuts, self.max_selected_cap)
            selected_local = tuple(range(fallback_count))
        selected_global = [top_indices[idx] for idx in selected_local]
        nselected = len(selected_global)
        objective_value = self._evaluate_subset(selected_local, working_linear_scores, pair_matrix)

        selected_set = set(selected_global)
        remaining = [idx for idx in np.argsort(original_linear_scores)[::-1].tolist() if idx not in selected_set]
        order = selected_global + remaining
        sorted_cuts = [cuts[idx] for idx in order]

        if root:
            self.root_rounds += 1
            self.total_root_candidates += len(cuts)
            self.total_root_selected += nselected
            self.last_root_selected_names = [self._row_name(cuts[i], f"cut_{i}") for i in selected_global]
            self._log_round(
                CutRoundRecord(
                    round_index=self.root_rounds,
                    root=True,
                    candidate_count=len(cuts),
                    forcedcut_count=len(forcedcuts),
                    maxnselectedcuts=maxnselectedcuts,
                    selected_count=nselected,
                    selector_name="P5-context-cutsel",
                    selected_cut_names=self.last_root_selected_names,
                    mean_efficacy=_mean_or_zero([record["efficacy"] for record in feature_records]),
                    mean_obj_parallelism=_mean_or_zero([record["obj_parallelism"] for record in feature_records]),
                    objective_value=float(objective_value),
                )
            )
        return {"cuts": sorted_cuts, "nselectedcuts": nselected, "result": SCIP_RESULT.SUCCESS}

    def _contextual_linear_scores(self, feature_records: list[dict[str, Any]]) -> np.ndarray:
        family_strength = _family_strength_scores(feature_records)
        efficacy = _robust_scale(np.array([record["efficacy"] for record in feature_records], dtype=float))
        cutoff_distance = _robust_scale(
            np.log1p(np.array([max(record["cutoff_distance"], 0.0) for record in feature_records], dtype=float))
        )
        obj_parallelism = _robust_scale(np.array([record["obj_parallelism"] for record in feature_records], dtype=float))
        density = np.array([record["support_density"] for record in feature_records], dtype=float)
        expected_improvement = _robust_scale(
            np.array([record["expected_improvement"] for record in feature_records], dtype=float)
        )
        int_support = np.array([record["int_support_ratio"] for record in feature_records], dtype=float)
        numerics = _robust_scale(
            np.array([1.0 / np.log1p(max(1.0, record["dynamism"])) for record in feature_records], dtype=float)
        )
        local_cut = np.array([1.0 if record["is_local"] else 0.0 for record in feature_records], dtype=float)
        return (
            self.linear_weights["efficacy"] * efficacy
            + self.linear_weights["obj_parallelism"] * obj_parallelism
            + self.linear_weights["density"] * density
            + self.linear_weights["cutoff_distance"] * cutoff_distance
            + self.linear_weights["expected_improvement"] * expected_improvement
            + self.linear_weights["integer_support"] * int_support
            + self.linear_weights["numerics"] * numerics
            + self.linear_weights["family_strength"] * family_strength
            + self.linear_weights["local_penalty"] * local_cut
        )

    def _context_pair_matrix(self, cuts, feature_records: list[dict[str, Any]]) -> np.ndarray:
        scip = self.model
        ncuts = len(cuts)
        matrix = np.zeros((ncuts, ncuts), dtype=float)
        for i in range(ncuts):
            support_i = set(feature_records[i]["support"])
            for j in range(i + 1, ncuts):
                support_j = set(feature_records[j]["support"])
                overlap = _support_overlap_ratio(support_i, support_j)
                try:
                    parallelism = float(scip.getRowParallelism(cuts[i], cuts[j]))
                except Exception:
                    parallelism = 0.0
                same_family = 1.0 if feature_records[i]["cut_family"] == feature_records[j]["cut_family"] else 0.0
                family_diversity = 1.0 - same_family
                score = (
                    self.pair_weights["parallelism"] * parallelism
                    + self.pair_weights["overlap"] * overlap
                    + self.pair_weights["same_family"] * same_family
                    + self.pair_weights["family_diversity"] * family_diversity
                )
                matrix[i, j] = score
                matrix[j, i] = score
        return matrix

    def _adaptive_pairwise_select(
        self,
        cuts,
        feature_records: list[dict[str, Any]],
        linear_scores: np.ndarray,
        pair_matrix: np.ndarray,
        maxnselectedcuts: int,
    ) -> tuple[int, ...]:
        nselected_cap = min(maxnselectedcuts, len(cuts), self.max_selected_cap)
        if nselected_cap <= 0:
            return tuple()
        best_linear = float(np.max(linear_scores))
        selected: list[int] = []
        remaining = set(range(linear_scores.shape[0]))
        while remaining and len(selected) < nselected_cap:
            best_idx = None
            best_value = float("-inf")
            for idx in remaining:
                marginal = float(linear_scores[idx])
                if selected:
                    marginal += float(np.sum(pair_matrix[idx, selected]))
                if marginal > best_value:
                    best_value = marginal
                    best_idx = idx
            if best_idx is None:
                break
            if selected and len(selected) >= self.min_selected_per_round and best_value < self.min_score_factor * best_linear:
                break
            selected.append(int(best_idx))
            remaining.remove(int(best_idx))
            remaining = self._filter_parallel_cuts(
                selected_idx=int(best_idx),
                remaining=remaining,
                cuts=cuts,
                feature_records=feature_records,
                linear_scores=linear_scores,
                best_linear=best_linear,
            )
        return tuple(selected)

    def _filter_parallel_cuts(
        self,
        selected_idx: int,
        remaining: set[int],
        cuts,
        feature_records: list[dict[str, Any]],
        linear_scores: np.ndarray,
        best_linear: float,
    ) -> set[int]:
        scip = self.model
        selected_record = feature_records[selected_idx]
        selected_eff = max(selected_record["efficacy"], 1e-9)
        filtered_remaining: set[int] = set()
        for idx in remaining:
            record = feature_records[idx]
            try:
                parallelism = float(scip.getRowParallelism(cuts[selected_idx], cuts[idx]))
            except Exception:
                parallelism = 0.0
            candidate_eff = max(record["efficacy"], 1e-9)
            weighted_parallelism = parallelism * candidate_eff / selected_eff
            max_allowed = self.good_max_parallelism if linear_scores[idx] >= self.good_score_factor * best_linear else self.max_parallelism
            adaptive_limit = self._adaptive_parallelism_limit(selected_eff, candidate_eff, parallelism, max_allowed)
            if weighted_parallelism >= 1.0 or parallelism > adaptive_limit:
                continue
            filtered_remaining.add(idx)
        return filtered_remaining

    def _adaptive_parallelism_limit(
        self,
        selected_eff: float,
        candidate_eff: float,
        parallelism: float,
        fallback_limit: float,
    ) -> float:
        denominator = 2.0 * selected_eff * candidate_eff
        if denominator <= 1e-12:
            return fallback_limit
        numerator = (
            selected_eff**2
            + candidate_eff**2
            - (1.0 + self.min_orthogonality_gain) ** 2 * selected_eff**2 * max(0.0, 1.0 - parallelism**2)
        )
        adaptive = numerator / denominator
        return float(np.clip(max(fallback_limit, adaptive), -1.0, 1.0))


def infer_candidate_regime(
    feature_records: list[dict[str, Any]],
    probe_cap: int = 60,
    dominance_threshold: float = 0.55,
) -> RegimeSignal:
    if not feature_records:
        return RegimeSignal(
            regime_name="no_rootcuts",
            dominant_family="none",
            dominant_share=0.0,
            probe_count=0,
            family_count=0,
        )
    probe_count = min(probe_cap, len(feature_records))
    ranked = sorted(feature_records, key=lambda record: float(record["efficacy"]), reverse=True)[:probe_count]
    weighted_counts: Counter[str] = Counter()
    for record in ranked:
        weighted_counts[record["cut_family"]] += max(float(record["efficacy"]), 1e-9)
    if not weighted_counts:
        return RegimeSignal(
            regime_name="mixed",
            dominant_family="unknown",
            dominant_share=0.0,
            probe_count=probe_count,
            family_count=0,
        )
    dominant_family, dominant_weight = weighted_counts.most_common(1)[0]
    total_weight = max(sum(weighted_counts.values()), 1e-9)
    dominant_share = float(dominant_weight / total_weight)
    family_count = len(weighted_counts)
    if family_count == 1 or dominant_share >= dominance_threshold:
        regime_name = dominant_family if dominant_family in {"gom", "cmir", "implbd", "clique", "flowcover"} else "other"
    else:
        regime_name = "mixed"
    return RegimeSignal(
        regime_name=regime_name,
        dominant_family=dominant_family,
        dominant_share=dominant_share,
        probe_count=probe_count,
        family_count=family_count,
    )


def route_probe_to_mode(probe_summary: dict[str, Any] | None) -> str:
    return route_probe_to_mode_with_policy(probe_summary, policy_name="raise_portfolio")


def route_probe_to_mode_with_policy(
    probe_summary: dict[str, Any] | None,
    policy_name: str = "raise_portfolio",
) -> str:
    mode, _ = route_probe_to_decision(probe_summary, policy_name=policy_name)
    return mode


def route_probe_to_decision(
    probe_summary: dict[str, Any] | None,
    policy_name: str = "raise_portfolio",
) -> tuple[str, str]:
    if not probe_summary:
        return "raise_cut", "no_probe_summary_abstain_to_raise_cut"
    candidate_count = int(probe_summary.get("candidate_count", 0) or 0)
    mean_obj_parallelism = float(probe_summary.get("mean_obj_parallelism", 0.0) or 0.0)
    regime_name = str(probe_summary.get("regime_name", "mixed") or "mixed")
    dominant_family = str(probe_summary.get("dominant_family", "mixed") or "mixed")
    dominant_share = float(probe_summary.get("dominant_share", 0.0) or 0.0)
    family_count = int(probe_summary.get("family_count", 0) or 0)
    if policy_name == "raise_portfolio_ud":
        if (
            regime_name == "mixed"
            and candidate_count >= 250
            and dominant_share <= 0.80
            and family_count >= 3
            and dominant_family != "cmir"
        ):
            return "scip_dynamic", "ud_large_diffuse_mixed_noncmir"
        return "raise_cut", "ud_abstain_to_raise_cut"
    if policy_name == "raise_portfolio_rc":
        if candidate_count >= 250 and mean_obj_parallelism <= 0.02:
            return "scip_ensemble", "rc_large_pool_low_obj_parallelism"
        return "raise_cut", "rc_abstain_to_raise_cut"
    if candidate_count >= 400 and mean_obj_parallelism <= 0.01:
        return "scip_ensemble", "large_pool_ultra_low_obj_parallelism"
    if regime_name == "clique" and dominant_share >= 0.70:
        return "plan5_adaptive", "high_dominance_clique"
    return "raise_cut", "default_raise_cut"


def _resolve_budget_floor(nselected_cap: int, regime_config: RegimeConfig) -> int:
    if nselected_cap <= 0:
        return 0
    ratio_floor = int(round(regime_config.budget_floor_ratio * nselected_cap))
    return min(nselected_cap, max(regime_config.min_selected_per_round, ratio_floor))


def _resolve_dominant_quota(
    budget_floor: int,
    regime_signal: RegimeSignal,
    regime_config: RegimeConfig,
    dominance_threshold: float,
) -> int:
    if budget_floor <= 0 or regime_signal.dominant_share < dominance_threshold:
        return 0
    if regime_config.dominant_quota_ratio <= 0.0:
        return 0
    quota = int(round(regime_config.dominant_quota_ratio * budget_floor))
    quota = max(regime_config.dominant_quota_min, quota)
    return min(budget_floor, quota)


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _support_overlap_ratio(support_i: set[int], support_j: set[int]) -> float:
    union = len(support_i | support_j)
    if union == 0:
        return 0.0
    return len(support_i & support_j) / union


def greedy_adaptive_select(
    linear_scores: np.ndarray,
    pair_matrix: np.ndarray,
    nselected_cap: int,
    min_selected: int,
    relative_marginal_threshold: float,
    absolute_marginal_threshold: float,
    max_nonpositive_streak: int,
) -> tuple[int, ...]:
    if nselected_cap <= 0 or linear_scores.size == 0:
        return tuple()
    selected: list[int] = []
    remaining = set(range(linear_scores.shape[0]))
    first_marginal = None
    nonpositive_streak = 0
    while len(selected) < nselected_cap and remaining:
        best_idx = None
        best_value = float("-inf")
        for idx in remaining:
            marginal = float(linear_scores[idx])
            if selected:
                marginal += float(np.sum(pair_matrix[idx, selected]))
            if marginal > best_value:
                best_value = marginal
                best_idx = idx
        if best_idx is None:
            break
        if first_marginal is None:
            first_marginal = max(abs(best_value), absolute_marginal_threshold)
        if len(selected) >= min_selected:
            stop_threshold = max(absolute_marginal_threshold, relative_marginal_threshold * first_marginal)
            if best_value < stop_threshold:
                break
            if best_value <= absolute_marginal_threshold:
                nonpositive_streak += 1
            else:
                nonpositive_streak = 0
            if nonpositive_streak > max_nonpositive_streak:
                break
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    if not selected:
        fallback_count = min(max(1, min_selected), nselected_cap, linear_scores.shape[0])
        fallback = np.argsort(linear_scores)[::-1][:fallback_count].tolist()
        return tuple(int(idx) for idx in fallback)
    return tuple(selected)


def greedy_regime_select(
    linear_scores: np.ndarray,
    pair_matrix: np.ndarray,
    families: list[str],
    nselected_cap: int,
    budget_floor: int,
    dominant_family: str,
    dominant_quota: int,
    quota_bonus: float,
    relative_marginal_threshold: float,
    absolute_marginal_threshold: float,
    max_nonpositive_streak: int,
) -> tuple[int, ...]:
    if nselected_cap <= 0 or linear_scores.size == 0:
        return tuple()
    selected: list[int] = []
    remaining = set(range(linear_scores.shape[0]))
    first_marginal = None
    nonpositive_streak = 0
    dominant_selected = 0
    while len(selected) < nselected_cap and remaining:
        best_idx = None
        best_value = float("-inf")
        for idx in remaining:
            marginal = float(linear_scores[idx])
            if selected:
                marginal += float(np.sum(pair_matrix[idx, selected]))
            if dominant_quota > 0 and families[idx] == dominant_family and dominant_selected < dominant_quota:
                shortfall = (dominant_quota - dominant_selected) / max(1, dominant_quota)
                marginal += quota_bonus * shortfall
            if marginal > best_value:
                best_value = marginal
                best_idx = idx
        if best_idx is None:
            break
        if first_marginal is None:
            first_marginal = max(abs(best_value), absolute_marginal_threshold)
        if len(selected) >= budget_floor:
            stop_threshold = max(absolute_marginal_threshold, relative_marginal_threshold * first_marginal)
            if best_value < stop_threshold:
                break
            if best_value <= absolute_marginal_threshold:
                nonpositive_streak += 1
            else:
                nonpositive_streak = 0
            if nonpositive_streak > max_nonpositive_streak:
                break
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
        if dominant_quota > 0 and families[best_idx] == dominant_family:
            dominant_selected += 1
    if not selected:
        fallback_count = min(max(1, budget_floor), nselected_cap, linear_scores.shape[0])
        fallback = np.argsort(linear_scores)[::-1][:fallback_count].tolist()
        return tuple(int(idx) for idx in fallback)
    return tuple(selected)


def compute_round_aware_dense_cap(
    regime_name: str,
    round_index: int,
    max_cap: int,
    maxnselectedcuts: int,
    candidate_count: int,
    dominant_share: float,
    dominant_history: int,
    dominant_streak: int,
    base_caps: dict[str, int],
    floor_caps: dict[str, int],
    decay_start_round: int,
    per_round_decay: int,
    strong_dominance_threshold: float,
    history_penalty_weight: float,
    max_history_penalty: int,
    streak_penalty: int,
) -> int:
    if max_cap <= 0 or maxnselectedcuts <= 0 or candidate_count <= 0:
        return 0
    regime_key = regime_name if regime_name in base_caps else "default"
    cap = int(base_caps.get(regime_key, max_cap))
    floor_cap = int(floor_caps.get(regime_key, 1))
    if dominant_share >= strong_dominance_threshold:
        if round_index > decay_start_round:
            cap -= per_round_decay * (round_index - decay_start_round)
        if dominant_history >= 2 * max_cap:
            cap -= min(max_history_penalty, int(round(history_penalty_weight * dominant_history)))
        if round_index > decay_start_round and dominant_streak >= 2:
            cap -= streak_penalty * max(0, dominant_streak - 1)
    cap = max(floor_cap, cap)
    cap = min(cap, max_cap, maxnselectedcuts, candidate_count)
    return max(1, int(cap))


def reorder_dense_candidates_with_quota(
    ranked_indices: list[int],
    families: list[str],
    dominant_family: str,
    dominant_quota: int,
) -> list[int]:
    if dominant_quota <= 0 or not ranked_indices:
        return list(ranked_indices)
    leading: list[int] = []
    deferred: list[int] = []
    dominant_taken = 0
    for idx in ranked_indices:
        if families[idx] == dominant_family and dominant_taken >= dominant_quota:
            deferred.append(idx)
            continue
        leading.append(idx)
        if families[idx] == dominant_family:
            dominant_taken += 1
    return leading + deferred


def _family_strength_scores(feature_records: list[dict[str, Any]]) -> np.ndarray:
    family_values: dict[str, float] = {}
    grouped: dict[str, list[float]] = {}
    for record in feature_records:
        grouped.setdefault(record["cut_family"], []).append(
            float(record["efficacy"] + 0.25 * record["obj_parallelism"] + 0.15 * record["cutoff_distance"])
        )
    for family, values in grouped.items():
        family_values[family] = float(np.quantile(np.array(values, dtype=float), 0.8))
    scores = np.array([family_values[record["cut_family"]] for record in feature_records], dtype=float)
    return _robust_scale(scores)


def _robust_scale(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    median = np.median(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    if abs(iqr) < eps:
        centered = values - median
        max_abs = np.max(np.abs(centered))
        if max_abs < eps:
            return np.zeros_like(values)
        return centered / max_abs
    return (values - median) / (iqr + eps)
