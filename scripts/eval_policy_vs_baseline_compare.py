from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"

INTERACTIONS_PATH = PROCESSED / "interactions_injected.csv"
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


def prepare_dev_impressions(
    interactions: pd.DataFrame,
    pop_scores: pd.Series,
    risk: pd.DataFrame,
    intent: pd.DataFrame,
) -> pd.DataFrame:
    dev_impr = interactions[
        (interactions["split"] == "dev")
        & (interactions["event_type"] == "impression")
    ].copy()

    dev_impr["pop_score"] = dev_impr["item_id"].map(pop_scores).fillna(0)

    dev_impr = dev_impr.merge(risk, on="item_id", how="left")
    dev_impr = dev_impr.merge(intent, on="session_id", how="left")

    dev_impr["risk_label"] = dev_impr["risk_label"].fillna("benign")
    dev_impr["intent_label"] = dev_impr["intent_label"].fillna("normal_interest")
    return dev_impr


def apply_policy_score(intent_label: str, risk_label: str, pop_score: float) -> float:
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


def build_baseline_topk(dev_impr: pd.DataFrame) -> pd.DataFrame:
    ranked = dev_impr.sort_values(
        by=["session_id", "pop_score", "position"],
        ascending=[True, False, True],
    ).copy()

    ranked["rank_baseline"] = ranked.groupby("session_id").cumcount() + 1
    topk = ranked[ranked["rank_baseline"] <= TOPK].copy()
    return topk


def build_policy_topk(dev_impr: pd.DataFrame) -> pd.DataFrame:
    ranked = dev_impr.copy()

    ranked["policy_score"] = ranked.apply(
        lambda row: apply_policy_score(
            row["intent_label"],
            row["risk_label"],
            row["pop_score"],
        ),
        axis=1,
    )

    ranked = ranked.sort_values(
        by=["session_id", "policy_score", "position"],
        ascending=[True, False, True],
    ).copy()

    ranked["rank_policy"] = ranked.groupby("session_id").cumcount() + 1
    topk = ranked[ranked["rank_policy"] <= TOPK].copy()
    return topk


def eval_quality_on_normal_interest(topk: pd.DataFrame, rank_col: str) -> tuple[float, float, int]:
    subset = topk[topk["intent_label"] == "normal_interest"].copy()

    hr_scores = []
    ndcg_scores = []
    n_eval = 0

    for _, group in subset.groupby("session_id", sort=False):
        labels = group.sort_values(rank_col)["clicked"].astype(int).tolist()
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


def eval_osr_against_baseline(baseline_topk: pd.DataFrame, policy_topk: pd.DataFrame) -> tuple[float, int, int]:
    """
    Thesis-aligned practical approximation:
    B = baseline top-K items that are acceptable to display
        (benign or sensitive_educational)
    S = among B, the subset that disappears from defended top-K
    OSR = |S| / |B|
    """
    baseline_ok = baseline_topk[
        baseline_topk["risk_label"].isin(["benign", "sensitive_educational"])
    ][["session_id", "item_id"]].drop_duplicates()

    policy_kept = policy_topk[["session_id", "item_id"]].drop_duplicates()
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


def main() -> None:
    print("[STEP] Loading interactions_injected / risk / intent ...")
    interactions, risk, intent = load_data()

    print("[STEP] Building popularity scores from train clicks ...")
    pop_scores = build_pop_scores(interactions)

    print("[STEP] Preparing dev impressions ...")
    dev_impr = prepare_dev_impressions(interactions, pop_scores, risk, intent)

    print("[STEP] Building baseline Top-10 ...")
    baseline_topk = build_baseline_topk(dev_impr)

    print("[STEP] Building policy Top-10 ...")
    policy_topk = build_policy_topk(dev_impr)

    print("[STEP] Evaluating baseline metrics ...")
    base_hr, base_ndcg, base_eval_n = eval_quality_on_normal_interest(
        baseline_topk, "rank_baseline"
    )
    base_her = eval_her(baseline_topk)
    base_osr = 0.0

    print("[STEP] Evaluating policy metrics ...")
    pol_hr, pol_ndcg, pol_eval_n = eval_quality_on_normal_interest(
        policy_topk, "rank_policy"
    )
    pol_her = eval_her(policy_topk)
    pol_osr, osr_num, osr_den = eval_osr_against_baseline(baseline_topk, policy_topk)

    ndcg_loss = 0.0
    if base_ndcg > 0:
        ndcg_loss = (base_ndcg - pol_ndcg) / base_ndcg

    print()
    print("[RESULT] Baseline vs Policy Comparison")
    print("  --- Baseline ---")
    print(f"  eval_sessions   = {base_eval_n:,}")
    print(f"  HER             = {base_her:.6f}")
    print(f"  OSR             = {base_osr:.6f}")
    print(f"  HitRate@10      = {base_hr:.6f}")
    print(f"  NDCG@10         = {base_ndcg:.6f}")
    print()
    print("  --- Policy V1 ---")
    print(f"  eval_sessions   = {pol_eval_n:,}")
    print(f"  HER             = {pol_her:.6f}")
    print(f"  OSR             = {pol_osr:.6f}")
    print(f"  HitRate@10      = {pol_hr:.6f}")
    print(f"  NDCG@10         = {pol_ndcg:.6f}")
    print()
    print("  --- Delta ---")
    print(f"  HER reduction   = {base_her - pol_her:.6f}")
    print(f"  NDCG loss       = {ndcg_loss:.6f}")
    print(f"  OSR numerator   = {osr_num:,}")
    print(f"  OSR denominator = {osr_den:,}")


if __name__ == "__main__":
    main()