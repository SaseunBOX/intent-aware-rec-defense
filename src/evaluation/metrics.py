"""Shared evaluation metrics extracted from the prototype scripts."""

from __future__ import annotations

import math

import pandas as pd


def dcg_at_k(labels: list[int], k: int = 10) -> float:
    """Compute DCG@k using the same formulation as the existing scripts."""
    score = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(labels: list[int], k: int = 10) -> float:
    """Compute NDCG@k."""
    actual = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def hitrate_at_k(labels: list[int], k: int = 10) -> float:
    """Return whether a positive label appears in the top-k list."""
    return 1.0 if any(x > 0 for x in labels[:k]) else 0.0


def eval_her(topk: pd.DataFrame) -> float:
    """Compute HER using the current prototype logic."""
    if len(topk) == 0:
        return 0.0
    harmful = (topk["risk_label"] == "harmful_promotional").sum()
    return harmful / len(topk)


def eval_osr_against_baseline(
    baseline_topk: pd.DataFrame,
    policy_topk: pd.DataFrame,
) -> tuple[float, int, int]:
    """Compute OSR using the prototype's baseline-vs-policy comparison logic."""
    baseline_ok = baseline_topk[
        baseline_topk["risk_label"].isin(["benign", "sensitive_educational"])
    ][["session_id", "item_id"]].drop_duplicates()

    policy_kept = policy_topk[["session_id", "item_id"]].drop_duplicates().copy()
    policy_kept["kept_in_policy"] = 1

    merged = baseline_ok.merge(
        policy_kept,
        on=["session_id", "item_id"],
        how="left",
    )
    merged["kept_in_policy"] = merged["kept_in_policy"].fillna(0).astype(int)
    merged["suppressed"] = 1 - merged["kept_in_policy"]

    denom = len(merged)
    numer = int(merged["suppressed"].sum())
    if denom == 0:
        return 0.0, 0, 0
    return numer / denom, numer, denom
