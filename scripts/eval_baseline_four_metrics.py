from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"
RISK_PATH = PROCESSED / "risk_labels.csv"
INTENT_PATH = PROCESSED / "intent_labels.csv"

TOPK = 10


def dcg_at_k(labels: list[int], k: int = 10) -> float:
    score = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(labels: list[int], k: int = 10) -> float:
    actual = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def hitrate_at_k(labels: list[int], k: int = 10) -> float:
    return 1.0 if any(x > 0 for x in labels[:k]) else 0.0


def load_data():
    interactions = pd.read_csv(
        INTERACTIONS_PATH,
        usecols=[
            "session_id",
            "user_id",
            "item_id",
            "event_type",
            "clicked",
            "position",
            "split",
            "impression_id",
        ],
    )
    risk = pd.read_csv(RISK_PATH, usecols=["item_id", "risk_label"])
    intent = pd.read_csv(INTENT_PATH, usecols=["session_id", "intent_label"])
    return interactions, risk, intent


def build_pop_scores(interactions: pd.DataFrame) -> pd.Series:
    train_clicks = interactions[
        (interactions["split"] == "train")
        & (interactions["event_type"] == "impression")
        & (interactions["clicked"] == 1)
    ].copy()

    return train_clicks.groupby("item_id").size().sort_values(ascending=False)


def build_dev_ranked(interactions: pd.DataFrame, pop_scores: pd.Series, risk: pd.DataFrame, intent: pd.DataFrame) -> pd.DataFrame:
    dev_impr = interactions[
        (interactions["split"] == "dev")
        & (interactions["event_type"] == "impression")
    ].copy()

    dev_impr["pop_score"] = dev_impr["item_id"].map(pop_scores).fillna(0)

    dev_impr = dev_impr.merge(risk, on="item_id", how="left")
    dev_impr = dev_impr.merge(intent, on="session_id", how="left")

    dev_impr["risk_label"] = dev_impr["risk_label"].fillna("benign")
    dev_impr["intent_label"] = dev_impr["intent_label"].fillna("normal_interest")

    dev_impr = dev_impr.sort_values(
        by=["session_id", "pop_score", "position"],
        ascending=[True, False, True],
    ).copy()

    dev_impr["rank_after_sort"] = dev_impr.groupby("session_id").cumcount() + 1
    topk = dev_impr[dev_impr["rank_after_sort"] <= TOPK].copy()
    return topk


def eval_quality_on_normal_interest(topk: pd.DataFrame) -> tuple[float, float, int]:
    subset = topk[topk["intent_label"] == "normal_interest"].copy()

    hr_scores = []
    ndcg_scores = []
    n_eval = 0

    for _, group in subset.groupby("session_id", sort=False):
        labels = group.sort_values("rank_after_sort")["clicked"].astype(int).tolist()
        if sum(labels) == 0:
            continue
        hr_scores.append(hitrate_at_k(labels, TOPK))
        ndcg_scores.append(ndcg_at_k(labels, TOPK))
        n_eval += 1

    if n_eval == 0:
        return 0.0, 0.0, 0

    return sum(hr_scores) / n_eval, sum(ndcg_scores) / n_eval, n_eval


def eval_her(topk: pd.DataFrame) -> float:
    if len(topk) == 0:
        return 0.0
    harmful = (topk["risk_label"] == "harmful_promotional").sum()
    return harmful / len(topk)


def eval_osr_baseline() -> float:
    # Baseline has no safety intervention, so no acceptable item is suppressed.
    return 0.0


def main() -> None:
    print("[STEP] Loading interactions / risk / intent ...")
    interactions, risk, intent = load_data()

    print("[STEP] Building popularity scores from train clicks ...")
    pop_scores = build_pop_scores(interactions)

    print("[STEP] Ranking dev candidates and taking Top-10 ...")
    topk = build_dev_ranked(interactions, pop_scores, risk, intent)

    print("[STEP] Evaluating four metrics ...")
    hr10, ndcg10, n_eval = eval_quality_on_normal_interest(topk)
    her = eval_her(topk)
    osr = eval_osr_baseline()

    print()
    print("[RESULT] Baseline Four-Metric Evaluation")
    print(f"  topk_rows       = {len(topk):,}")
    print(f"  eval_sessions   = {n_eval:,}")
    print(f"  HER             = {her:.6f}")
    print(f"  OSR             = {osr:.6f}")
    print(f"  HitRate@10      = {hr10:.6f}")
    print(f"  NDCG@10         = {ndcg10:.6f}")


if __name__ == "__main__":
    main()