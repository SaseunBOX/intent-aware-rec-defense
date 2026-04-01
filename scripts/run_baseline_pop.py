from __future__ import annotations
#提供一个可解释、简单、稳定的推荐对照组。
#Provide an interpretable, simple and stable control group for recommendations.
import math
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"


def load_interactions() -> pd.DataFrame:
    if not INTERACTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INTERACTIONS_PATH}")

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
    df = pd.read_csv(INTERACTIONS_PATH, usecols=usecols)
    return df


def build_popularity_scores(df: pd.DataFrame) -> pd.Series:
    train_clicks = df[
        (df["split"] == "train")
        & (df["event_type"] == "impression")
        & (df["clicked"] == 1)
    ].copy()

    pop = train_clicks.groupby("item_id").size().sort_values(ascending=False)
    return pop


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


def evaluate_dev(df: pd.DataFrame, pop_scores: pd.Series) -> tuple[float, float, int]:
    dev_impr = df[
        (df["split"] == "dev")
        & (df["event_type"] == "impression")
    ].copy()

    dev_impr["pop_score"] = dev_impr["item_id"].map(pop_scores).fillna(0)

    eval_rows = 0
    hr_scores = []
    ndcg_scores = []

    grouped = dev_impr.groupby("session_id", sort=False)

    for session_id, group in grouped:
        group = group.sort_values(
            by=["pop_score", "position"],
            ascending=[False, True]
        )

        labels = group["clicked"].astype(int).tolist()

        # 跳过没有正样本的 session
        if sum(labels) == 0:
            continue

        hr_scores.append(hitrate_at_k(labels, 10))
        ndcg_scores.append(ndcg_at_k(labels, 10))
        eval_rows += 1

    if eval_rows == 0:
        return 0.0, 0.0, 0

    hr10 = sum(hr_scores) / len(hr_scores)
    ndcg10 = sum(ndcg_scores) / len(ndcg_scores)
    return hr10, ndcg10, eval_rows


def main() -> None:
    print("[STEP] Loading interactions.csv ...")
    df = load_interactions()

    print("[STEP] Building popularity scores from train clicks ...")
    pop_scores = build_popularity_scores(df)

    print("[STEP] Evaluating on dev impressions ...")
    hr10, ndcg10, n_eval = evaluate_dev(df, pop_scores)

    print()
    print("[RESULT] Popularity Baseline")
    print(f"  eval_sessions = {n_eval:,}")
    print(f"  HitRate@10    = {hr10:.6f}")
    print(f"  NDCG@10       = {ndcg10:.6f}")


if __name__ == "__main__":
    main()