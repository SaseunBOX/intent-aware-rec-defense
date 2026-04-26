"""Reusable policy reranking helpers from `scripts/eval_policy_rerank_v1.py`."""

from __future__ import annotations

import pandas as pd

from src.evaluation.baseline_eval import TOPK, rank_by_score


def apply_policy_score(intent_label: str, risk_label: str, pop_score: float) -> float:
    """Apply the current v1 intent-aware / risk-aware reranking rule."""
    score = float(pop_score)

    if intent_label == "normal_interest":
        if risk_label == "harmful_promotional":
            score -= 1_000_000

    elif intent_label == "sensitive_help_seeking":
        if risk_label == "harmful_promotional":
            score -= 2_000_000
        elif risk_label == "sensitive_educational":
            score += 1_000

    elif intent_label == "clearly_harmful_intent":
        if risk_label == "harmful_promotional":
            score = -9_999_999
        elif risk_label == "sensitive_educational":
            score -= 500

    return score


def add_policy_scores(dev_impr: pd.DataFrame) -> pd.DataFrame:
    """Attach a `policy_score` column using the current prototype logic."""
    ranked = dev_impr.copy()
    ranked["policy_score"] = ranked.apply(
        lambda row: apply_policy_score(
            row["intent_label"],
            row["risk_label"],
            row["pop_score"],
        ),
        axis=1,
    )
    return ranked


def build_policy_topk(dev_impr: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    """Build top-k rankings after applying the v1 policy score."""
    ranked = add_policy_scores(dev_impr)
    return rank_by_score(ranked, score_col="policy_score", rank_col="rank_policy", topk=topk)

