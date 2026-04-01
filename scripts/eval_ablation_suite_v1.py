from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

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


def eval_her(topk: pd.DataFrame) -> float:
    if len(topk) == 0:
        return 0.0
    harmful = (topk["risk_label"] == "harmful_promotional").sum()
    return harmful / len(topk)


def eval_osr_against_baseline(
    baseline_topk: pd.DataFrame,
    policy_topk: pd.DataFrame,
) -> tuple[float, int, int]:
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


def score_baseline(row: pd.Series) -> float:
    return float(row["pop_score"])


def score_risk_only(row: pd.Series) -> float:
    score = float(row["pop_score"])
    if row["risk_label"] == "harmful_promotional":
        score -= 1_500_000
    return score


def score_intent_only(row: pd.Series) -> float:
    score = float(row["pop_score"])

    if row["intent_label"] == "sensitive_help_seeking":
        if row["risk_label"] == "sensitive_educational":
            score += 1_000

    elif row["intent_label"] == "clearly_harmful_intent":
        if row["risk_label"] == "sensitive_educational":
            score -= 500

    return score


def score_full_policy(row: pd.Series) -> float:
    score = float(row["pop_score"])

    if row["intent_label"] == "normal_interest":
        if row["risk_label"] == "harmful_promotional":
            score -= 1_000_000

    elif row["intent_label"] == "sensitive_help_seeking":
        if row["risk_label"] == "harmful_promotional":
            score -= 2_000_000
        elif row["risk_label"] == "sensitive_educational":
            score += 1_000

    elif row["intent_label"] == "clearly_harmful_intent":
        if row["risk_label"] == "harmful_promotional":
            score = -9_999_999
        elif row["risk_label"] == "sensitive_educational":
            score -= 500

    return score


def build_topk(dev_impr: pd.DataFrame, score_fn, rank_col: str) -> pd.DataFrame:
    ranked = dev_impr.copy()
    ranked["model_score"] = ranked.apply(score_fn, axis=1)
    ranked = ranked.sort_values(
        by=["session_id", "model_score", "position"],
        ascending=[True, False, True],
    ).copy()
    ranked[rank_col] = ranked.groupby("session_id").cumcount() + 1
    topk = ranked[ranked[rank_col] <= TOPK].copy()
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


def evaluate_system(
    system_name: str,
    topk: pd.DataFrame,
    rank_col: str,
    baseline_topk: pd.DataFrame,
    baseline_ndcg: float,
) -> dict:
    hr10, ndcg10, n_eval = eval_quality_on_normal_interest(topk, rank_col)
    her = eval_her(topk)

    if system_name == "Injected_Baseline":
        osr = 0.0
        osr_num = 0
        osr_den = 0
    else:
        osr, osr_num, osr_den = eval_osr_against_baseline(baseline_topk, topk)

    utility_loss = 0.0
    if baseline_ndcg > 0:
        utility_loss = (baseline_ndcg - ndcg10) / baseline_ndcg

    return {
        "System": system_name,
        "HER": her,
        "OSR": osr,
        "HitRate@10": hr10,
        "NDCG@10": ndcg10,
        "UtilityLoss_vs_Baseline": utility_loss,
        "EvalSessions": n_eval,
        "OSR_numerator": osr_num,
        "OSR_denominator": osr_den,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading data ...")
    interactions, risk, intent = load_data()

    print("[STEP] Building popularity scores ...")
    pop_scores = build_pop_scores(interactions)

    print("[STEP] Preparing dev impressions ...")
    dev_impr = prepare_dev_impressions(interactions, pop_scores, risk, intent)

    print("[STEP] Building rankings ...")
    baseline_topk = build_topk(dev_impr, score_baseline, "rank_baseline")
    risk_only_topk = build_topk(dev_impr, score_risk_only, "rank_risk_only")
    intent_only_topk = build_topk(dev_impr, score_intent_only, "rank_intent_only")
    full_policy_topk = build_topk(dev_impr, score_full_policy, "rank_full_policy")

    print("[STEP] Evaluating systems ...")
    base_hr, base_ndcg, _ = eval_quality_on_normal_interest(baseline_topk, "rank_baseline")

    results = [
        evaluate_system("Injected_Baseline", baseline_topk, "rank_baseline", baseline_topk, base_ndcg),
        evaluate_system("Risk_Only_V1", risk_only_topk, "rank_risk_only", baseline_topk, base_ndcg),
        evaluate_system("Intent_Only_V1", intent_only_topk, "rank_intent_only", baseline_topk, base_ndcg),
        evaluate_system("Full_Policy_V1", full_policy_topk, "rank_full_policy", baseline_topk, base_ndcg),
    ]

    out_df = pd.DataFrame(results)
    out_path = RESULTS / "main_metrics_table_v2.csv"
    out_df.to_csv(out_path, index=False)

    print()
    print("[DONE] Saved:")
    print(f"  {out_path}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()