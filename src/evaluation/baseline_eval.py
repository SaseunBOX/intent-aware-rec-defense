"""Shared baseline loading, scoring, and ranking helpers.

These functions consolidate logic repeated across:
- `scripts/run_baseline_pop.py`
- `scripts/eval_baseline_four_metrics.py`
- `scripts/eval_baseline_four_metrics_injected.py`
- `scripts/eval_policy_vs_baseline_compare.py`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import hitrate_at_k, ndcg_at_k


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"
INTERACTIONS_INJECTED_PATH = PROCESSED / "interactions_injected.csv"
RISK_PATH = PROCESSED / "risk_labels.csv"
INTENT_PATH = PROCESSED / "intent_labels.csv"
TOPK = 10


def load_interactions(
    path: Path = INTERACTIONS_PATH,
    include_id_timestamp_source: bool = False,
) -> pd.DataFrame:
    """Load interactions with the columns commonly used by the evaluation scripts."""
    usecols = [
        "session_id",
        "user_id",
        "item_id",
        "event_type",
        "clicked",
        "position",
        "split",
        "impression_id",
    ]
    if include_id_timestamp_source:
        usecols = [
            "interaction_id",
            "session_id",
            "user_id",
            "timestamp",
            "item_id",
            "item_source",
            "event_type",
            "clicked",
            "position",
            "split",
            "impression_id",
        ]
    return pd.read_csv(path, usecols=usecols)


def load_risk_labels(path: Path = RISK_PATH) -> pd.DataFrame:
    """Load item risk labels."""
    return pd.read_csv(path, usecols=["item_id", "risk_label"])


def load_intent_labels(path: Path = INTENT_PATH) -> pd.DataFrame:
    """Load session intent labels."""
    return pd.read_csv(path, usecols=["session_id", "intent_label"])


def load_eval_inputs(
    interactions_path: Path = INTERACTIONS_PATH,
    risk_path: Path = RISK_PATH,
    intent_path: Path = INTENT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load interactions, risk labels, and intent labels together."""
    return (
        load_interactions(interactions_path),
        load_risk_labels(risk_path),
        load_intent_labels(intent_path),
    )


def build_popularity_scores(interactions: pd.DataFrame) -> pd.Series:
    """Build popularity counts from clicked train impressions."""
    train_clicks = interactions[
        (interactions["split"] == "train")
        & (interactions["event_type"] == "impression")
        & (interactions["clicked"] == 1)
    ].copy()
    return train_clicks.groupby("item_id").size().sort_values(ascending=False)


def get_dev_impressions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Filter to dev impressions only."""
    return interactions[
        (interactions["split"] == "dev")
        & (interactions["event_type"] == "impression")
    ].copy()


def prepare_dev_impressions(
    interactions: pd.DataFrame,
    pop_scores: pd.Series,
    risk: pd.DataFrame | None = None,
    intent: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach scores and optional labels to dev impression rows."""
    dev_impr = get_dev_impressions(interactions)
    dev_impr["pop_score"] = dev_impr["item_id"].map(pop_scores).fillna(0)

    if risk is not None:
        dev_impr = dev_impr.merge(risk, on="item_id", how="left")
        dev_impr["risk_label"] = dev_impr["risk_label"].fillna("benign")

    if intent is not None:
        dev_impr = dev_impr.merge(intent, on="session_id", how="left")
        dev_impr["intent_label"] = dev_impr["intent_label"].fillna("normal_interest")

    return dev_impr


def rank_by_score(
    dev_impr: pd.DataFrame,
    score_col: str,
    rank_col: str,
    topk: int = TOPK,
) -> pd.DataFrame:
    """Rank within session by a score column and keep top-k rows."""
    ranked = dev_impr.sort_values(
        by=["session_id", score_col, "position"],
        ascending=[True, False, True],
    ).copy()
    ranked[rank_col] = ranked.groupby("session_id").cumcount() + 1
    return ranked[ranked[rank_col] <= topk].copy()


def build_baseline_topk(dev_impr: pd.DataFrame, topk: int = TOPK) -> pd.DataFrame:
    """Build baseline top-k using popularity score."""
    return rank_by_score(dev_impr, score_col="pop_score", rank_col="rank_baseline", topk=topk)


def evaluate_quality(
    topk_df: pd.DataFrame,
    rank_col: str,
    topk: int = TOPK,
    intent_filter: str | None = None,
) -> tuple[float, float, int]:
    """Evaluate HitRate and NDCG over ranked session lists."""
    subset = topk_df.copy()
    if intent_filter is not None and "intent_label" in subset.columns:
        subset = subset[subset["intent_label"] == intent_filter].copy()

    hr_scores = []
    ndcg_scores = []
    n_eval = 0

    for _, group in subset.groupby("session_id", sort=False):
        labels = group.sort_values(rank_col)["clicked"].astype(int).tolist()
        if sum(labels) == 0:
            continue
        hr_scores.append(hitrate_at_k(labels, topk))
        ndcg_scores.append(ndcg_at_k(labels, topk))
        n_eval += 1

    if n_eval == 0:
        return 0.0, 0.0, 0
    return sum(hr_scores) / n_eval, sum(ndcg_scores) / n_eval, n_eval


def eval_quality_on_normal_interest(
    topk_df: pd.DataFrame,
    rank_col: str,
    topk: int = TOPK,
) -> tuple[float, float, int]:
    """Evaluate quality on `normal_interest` sessions, matching the prototype scripts."""
    return evaluate_quality(
        topk_df=topk_df,
        rank_col=rank_col,
        topk=topk,
        intent_filter="normal_interest",
    )
