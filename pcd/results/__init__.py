"""Persistent, replayable design-study results."""

from .maintenance import prune_generations
from .report import (
    best_decision_summary,
    candidate_summary,
    pareto_front_table,
    read_best_candidate,
    read_evaluation_table,
    selected_evaluation,
    study_artifact_path,
)
from .snapshot import quasi_static_snapshot_table
from .store import FileResultStore, raw_evaluation_identity, raw_evaluation_key

__all__ = [
    "FileResultStore",
    "best_decision_summary",
    "candidate_summary",
    "pareto_front_table",
    "prune_generations",
    "quasi_static_snapshot_table",
    "raw_evaluation_identity",
    "raw_evaluation_key",
    "read_best_candidate",
    "read_evaluation_table",
    "selected_evaluation",
    "study_artifact_path",
]
