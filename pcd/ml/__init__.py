"""Leakage-aware preparation and evaluation for optional surrogate models."""

from .ac_evaluation import AcModelComparison, evaluate_ac_models
from .corpus import AcResponseRecord, PreparedAcCorpus, prepare_ac_corpus
from .dataset import PreparedDataset, assign_group_holdout, prepare_evaluation_dataset
from .evaluation import SurrogateEvaluation, evaluate_surrogate_dataset
from .ranking import RankingEvaluation, evaluate_candidate_ranking

__all__ = [
    "AcModelComparison",
    "AcResponseRecord",
    "PreparedAcCorpus",
    "PreparedDataset",
    "RankingEvaluation",
    "SurrogateEvaluation",
    "assign_group_holdout",
    "evaluate_ac_models",
    "evaluate_candidate_ranking",
    "evaluate_surrogate_dataset",
    "prepare_ac_corpus",
    "prepare_evaluation_dataset",
]
