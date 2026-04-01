from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "tables"

TOPK_PATH = RESULTS / "full_pipeline_v2_topk.csv"
ACTIONS_PATH = RESULTS / "full_pipeline_v2_actions.csv"
ITEMS_PATH = PROCESSED / "items.csv"
INTENT_PROBS_PATH = MODELS / "intent" / "intent_probabilities_all.csv"

TRUE_INTENT_OUT = RESULTS / "subgroup_metrics_by_true_intent_v1.csv"
PRED_INTENT_OUT = RESULTS / "subgroup_metrics_by_predicted_intent_v1.csv"
SOURCE_OUT = RESULTS / "subgroup_metrics_by_source_v1.csv"

ACTION_TRUE_OUT = RESULTS / "action_breakdown_by_true_intent_v1.csv"
ACTION_PRED_OUT = RESULTS / "action_breakdown_by_predicted_intent_v1.csv"

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


def load_tables():
    topk = pd.read_csv(TOPK_PATH)
    actions = pd.read_csv(ACTIONS_PATH)
    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["item_id", "source", "category", "subcategory"],
    )
    intent_probs = pd.read_csv(
        INTENT_PROBS_PATH,
        usecols=["session_id", "intent_label", "predicted_intent_label", "split"],
    )
    return topk, actions, items, intent_probs


def evaluate_by_intent(topk: pd.DataFrame, intent_col: str) -> pd.DataFrame:
    rows = []

    for intent_value, subset in topk.groupby(intent_col, dropna=False):
        subset = subset.copy()
        n_sessions = subset["session_id"].nunique()
        n_rows = len(subset)

        harmful_rate = (subset["final_risk_label"] == "harmful_promotional").mean()
        supportive_rate = (subset["final_risk_label"] == "sensitive_educational").mean()
        clicked_rate = subset["clicked"].mean()
        avg_rank = subset["rank_final"].mean()

        hr_scores = []
        ndcg_scores = []
        n_eval_sessions = 0

        for _, group in subset.groupby("session_id", sort=False):
            labels = group.sort_values("rank_final")["clicked"].astype(int).tolist()
            if sum(labels) == 0:
                continue
            hr_scores.append(hitrate_at_k(labels, TOPK))
            ndcg_scores.append(ndcg_at_k(labels, TOPK))
            n_eval_sessions += 1

        hitrate = sum(hr_scores) / n_eval_sessions if n_eval_sessions > 0 else 0.0
        ndcg = sum(ndcg_scores) / n_eval_sessions if n_eval_sessions > 0 else 0.0

        rows.append(
            {
                "intent_column": intent_col,
                "intent_group": intent_value,
                "n_sessions": n_sessions,
                "n_rows": n_rows,
                "avg_rows_per_session": n_rows / n_sessions if n_sessions > 0 else 0.0,
                "HER": harmful_rate,
                "SupportiveRate": supportive_rate,
                "ClickedRate": clicked_rate,
                "AvgFinalRank": avg_rank,
                "HitRate@10": hitrate,
                "NDCG@10": ndcg,
                "n_eval_sessions": n_eval_sessions,
            }
        )

    out = pd.DataFrame(rows).sort_values(by="n_sessions", ascending=False).reset_index(drop=True)
    return out


def evaluate_by_source(topk: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    merged = topk.merge(
        items.rename(columns={"item_id": "final_item_id", "source": "final_source"}),
        on="final_item_id",
        how="left",
    )

    rows = []
    total_rows = len(merged)

    for source_value, subset in merged.groupby("final_source", dropna=False):
        n_rows = len(subset)
        rows.append(
            {
                "final_source": source_value,
                "n_rows": n_rows,
                "row_share": n_rows / total_rows if total_rows > 0 else 0.0,
                "ClickedRate": subset["clicked"].mean(),
                "HER_within_source": (subset["final_risk_label"] == "harmful_promotional").mean(),
                "SupportiveRate_within_source": (subset["final_risk_label"] == "sensitive_educational").mean(),
                "AvgFinalRank": subset["rank_final"].mean(),
            }
        )

    out = pd.DataFrame(rows).sort_values(by="n_rows", ascending=False).reset_index(drop=True)
    return out


def action_breakdown(actions: pd.DataFrame, intent_probs: pd.DataFrame, intent_col: str) -> pd.DataFrame:
    merged = actions.merge(
        intent_probs[["session_id", "intent_label", "predicted_intent_label"]],
        on="session_id",
        how="left",
    )

    total_by_group = (
        merged.groupby(intent_col, dropna=False)
        .size()
        .rename("total_actions")
        .reset_index()
    )

    pivot = (
        merged.groupby([intent_col, "action"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    out = pivot.merge(total_by_group, on=intent_col, how="left")

    for action_name in ["keep", "down_rank", "replace", "block"]:
        if action_name not in out.columns:
            out[action_name] = 0
        out[f"{action_name}_share"] = out[action_name] / out["total_actions"].replace(0, 1)

    session_counts = (
        merged.groupby(intent_col)["session_id"]
        .nunique()
        .rename("n_sessions")
        .reset_index()
    )
    out = out.merge(session_counts, on=intent_col, how="left")

    out = out.sort_values(by="total_actions", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading tables ...")
    topk, actions, items, intent_probs = load_tables()

    print("[STEP] Computing subgroup metrics by true intent ...")
    true_intent_df = evaluate_by_intent(topk, "intent_label")
    true_intent_df.to_csv(TRUE_INTENT_OUT, index=False)

    print("[STEP] Computing subgroup metrics by predicted intent ...")
    pred_intent_df = evaluate_by_intent(topk, "predicted_intent_label")
    pred_intent_df.to_csv(PRED_INTENT_OUT, index=False)

    print("[STEP] Computing subgroup metrics by source ...")
    source_df = evaluate_by_source(topk, items)
    source_df.to_csv(SOURCE_OUT, index=False)

    print("[STEP] Computing action breakdown by true intent ...")
    action_true_df = action_breakdown(actions, intent_probs, "intent_label")
    action_true_df.to_csv(ACTION_TRUE_OUT, index=False)

    print("[STEP] Computing action breakdown by predicted intent ...")
    action_pred_df = action_breakdown(actions, intent_probs, "predicted_intent_label")
    action_pred_df.to_csv(ACTION_PRED_OUT, index=False)

    print()
    print("[DONE] Saved:")
    print(f"  - {TRUE_INTENT_OUT}")
    print(f"  - {PRED_INTENT_OUT}")
    print(f"  - {SOURCE_OUT}")
    print(f"  - {ACTION_TRUE_OUT}")
    print(f"  - {ACTION_PRED_OUT}")
    print()
    print("[TRUE INTENT SUBGROUP METRICS]")
    print(true_intent_df.to_string(index=False))
    print()
    print("[SOURCE SUBGROUP METRICS]")
    print(source_df.to_string(index=False))


if __name__ == "__main__":
    main()
