from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SWEEPS = RESULTS / "sweeps"

INTERACTIONS_PATH = PROCESSED / "interactions_injected.csv"
ITEMS_PATH = PROCESSED / "items.csv"

INTENT_PROBS_PATH = MODELS / "intent" / "intent_probabilities_all_v2_light.csv"
RISK_PROBS_PATH = MODELS / "risk" / "risk_probabilities_all.csv"

OUT_PATH = SWEEPS / "full_pipeline_v2_light_weight_sweep.csv"

TOPK = 10

# fixed thresholds from current full pipeline
W1 = 0.8
W2 = 1.4
W3 = 2.0

TAU_DOWN = 0.35
TAU_REPLACE = 0.55
TAU_BLOCK = 0.75
TAU_SAFE = 0.40

M = np.array([
    [ 1.0,  0.3, -1.0],
    [ 0.8,  1.2, -1.5],
    [ 0.2, -0.2, -2.0],
], dtype=float)

CONFIGS = []
for alpha in [0.10, 0.20, 0.30]:
    for beta in [0.40, 0.60, 0.80]:
        for gamma in [0.05, 0.15, 0.25]:
            CONFIGS.append(
                {
                    "config_name": f"a{alpha:.2f}_b{beta:.2f}_g{gamma:.2f}",
                    "ALPHA": alpha,
                    "BETA": beta,
                    "GAMMA": gamma,
                }
            )


def dcg_at_k(labels: list[int], k: int = 10) -> float:
    score = 0.0
    for i, rel in enumerate(labels[:k], start=1):
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(labels: list[int], k: int = 10) -> float:
    actual = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return 0.0 if ideal == 0 else actual / ideal


def hitrate_at_k(labels: list[int], k: int = 10) -> float:
    return 1.0 if any(x > 0 for x in labels[:k]) else 0.0


def load_all():
    interactions = pd.read_csv(
        INTERACTIONS_PATH,
        usecols=[
            "session_id",
            "user_id",
            "item_id",
            "item_source",
            "event_type",
            "clicked",
            "position",
            "split",
            "impression_id",
        ],
    )

    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["item_id", "source", "title", "text", "category", "subcategory"],
    )

    intent_probs = pd.read_csv(INTENT_PROBS_PATH)
    risk_probs = pd.read_csv(RISK_PROBS_PATH)

    return interactions, items, intent_probs, risk_probs


def build_pop_scores(interactions: pd.DataFrame) -> pd.Series:
    train_clicks = interactions[
        (interactions["split"] == "train")
        & (interactions["event_type"] == "impression")
        & (interactions["clicked"] == 1)
    ].copy()
    return train_clicks.groupby("item_id").size().sort_values(ascending=False)


def normalize_scores(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)


def prepare_dev_impressions(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    intent_probs: pd.DataFrame,
    risk_probs: pd.DataFrame,
    pop_scores: pd.Series,
) -> pd.DataFrame:
    dev_impr = interactions[
        (interactions["split"] == "dev")
        & (interactions["event_type"] == "impression")
    ].copy()

    dev_impr["baseline_score"] = dev_impr["item_id"].map(pop_scores).fillna(0)
    dev_impr["baseline_score_norm"] = normalize_scores(dev_impr["baseline_score"])

    risk_cols = [
        "item_id",
        "risk_label",
        "predicted_risk_label",
        "q_benign",
        "q_sensitive_educational",
        "q_harmful_promotional",
    ]
    intent_cols = [
        "session_id",
        "intent_label",
        "predicted_intent_label",
        "p_normal_interest",
        "p_sensitive_help_seeking",
        "p_clearly_harmful_intent",
    ]

    dev_impr = dev_impr.merge(items, on="item_id", how="left")
    dev_impr = dev_impr.merge(risk_probs[risk_cols], on="item_id", how="left")
    dev_impr = dev_impr.merge(intent_probs[intent_cols], on="session_id", how="left")

    dev_impr["risk_label"] = dev_impr["risk_label"].fillna("benign")
    dev_impr["predicted_risk_label"] = dev_impr["predicted_risk_label"].fillna("benign")
    dev_impr["predicted_intent_label"] = dev_impr["predicted_intent_label"].fillna("normal_interest")

    for col in [
        "q_benign",
        "q_sensitive_educational",
        "q_harmful_promotional",
        "p_normal_interest",
        "p_sensitive_help_seeking",
        "p_clearly_harmful_intent",
    ]:
        dev_impr[col] = dev_impr[col].fillna(0.0)

    return dev_impr


def compute_phi(row: pd.Series) -> float:
    p = np.array([
        row["p_normal_interest"],
        row["p_sensitive_help_seeking"],
        row["p_clearly_harmful_intent"],
    ], dtype=float)

    q = np.array([
        row["q_benign"],
        row["q_sensitive_educational"],
        row["q_harmful_promotional"],
    ], dtype=float)

    return float(p @ M @ q)


def compute_risk_penalty(row: pd.Series) -> float:
    q_h = float(row["q_harmful_promotional"])
    p1 = float(row["p_normal_interest"])
    p2 = float(row["p_sensitive_help_seeking"])
    p3 = float(row["p_clearly_harmful_intent"])
    return q_h * (W1 * p1 + W2 * p2 + W3 * p3)


def compute_guidance_bonus(row: pd.Series) -> float:
    return float(row["p_sensitive_help_seeking"] * row["q_sensitive_educational"])


def build_baseline_topk(dev_impr: pd.DataFrame) -> pd.DataFrame:
    ranked = dev_impr.sort_values(
        by=["session_id", "baseline_score", "position"],
        ascending=[True, False, True],
    ).copy()
    ranked["rank_baseline"] = ranked.groupby("session_id").cumcount() + 1
    return ranked[ranked["rank_baseline"] <= TOPK].copy()


def build_safe_pool(dev_impr: pd.DataFrame, pop_scores: pd.Series) -> pd.DataFrame:
    pool = dev_impr[
        ["item_id", "source", "title", "text",
         "q_benign", "q_sensitive_educational", "q_harmful_promotional",
         "predicted_risk_label"]
    ].drop_duplicates("item_id").copy()

    pool["safe_priority"] = np.where(
        pool["q_sensitive_educational"] >= TAU_SAFE, 2,
        np.where(pool["q_benign"] >= 0.5, 1, 0)
    )
    pool["baseline_score"] = pool["item_id"].map(pop_scores).fillna(0)
    pool = pool[pool["safe_priority"] > 0].copy()
    pool = pool.sort_values(by=["safe_priority", "baseline_score"], ascending=[False, False])
    return pool


def pick_replacement(used_ids: set[str], safe_pool: pd.DataFrame):
    for row in safe_pool.itertuples(index=False):
        if row.item_id in used_ids:
            continue
        return row.item_id, row.predicted_risk_label
    return None, None


def run_policy(dev_impr: pd.DataFrame, pop_scores: pd.Series, alpha: float, beta: float, gamma: float):
    safe_pool = build_safe_pool(dev_impr, pop_scores)

    ranked = dev_impr.copy()
    ranked["phi"] = ranked.apply(compute_phi, axis=1)
    ranked["risk_penalty"] = ranked.apply(compute_risk_penalty, axis=1)
    ranked["guidance_bonus"] = ranked.apply(compute_guidance_bonus, axis=1)

    ranked["final_score_raw"] = (
        ranked["baseline_score_norm"]
        + alpha * ranked["phi"]
        - beta * ranked["risk_penalty"]
        + gamma * ranked["guidance_bonus"]
    )

    ranked = ranked.sort_values(
        by=["session_id", "final_score_raw", "position"],
        ascending=[True, False, True],
    ).copy()

    action_logs = []
    final_rows = []

    for session_id, group in ranked.groupby("session_id", sort=False):
        group = group.copy()
        used_ids = set()
        final_rank = 0

        for row in group.itertuples(index=False):
            if final_rank >= TOPK:
                break

            item_id = row.item_id
            risk_label = row.predicted_risk_label
            q_h = float(row.q_harmful_promotional)

            action = "keep"
            final_item_id = item_id
            final_risk_label = risk_label

            if risk_label == "harmful_promotional":
                if q_h >= TAU_BLOCK:
                    action = "block"
                elif q_h >= TAU_REPLACE:
                    repl_id, repl_risk = pick_replacement(used_ids, safe_pool)
                    if repl_id is not None:
                        action = "replace"
                        final_item_id = repl_id
                        final_risk_label = repl_risk if repl_risk is not None else "benign"
                    else:
                        action = "block"
                elif q_h >= TAU_DOWN:
                    action = "down_rank"

            if action == "block":
                action_logs.append(
                    {
                        "session_id": session_id,
                        "original_item_id": item_id,
                        "final_item_id": None,
                        "action": "block",
                    }
                )
                continue

            if final_item_id in used_ids:
                continue

            final_rank += 1
            used_ids.add(final_item_id)

            final_rows.append(
                {
                    "session_id": session_id,
                    "user_id": row.user_id,
                    "split": row.split,
                    "clicked": row.clicked if final_item_id == item_id else 0,
                    "rank_final": final_rank,
                    "original_item_id": item_id,
                    "final_item_id": final_item_id,
                    "action": action,
                    "intent_label": row.intent_label,
                    "predicted_intent_label": row.predicted_intent_label,
                    "risk_label": row.risk_label,
                    "predicted_risk_label": row.predicted_risk_label,
                    "final_risk_label": final_risk_label,
                }
            )

            action_logs.append(
                {
                    "session_id": session_id,
                    "original_item_id": item_id,
                    "final_item_id": final_item_id,
                    "action": action,
                }
            )

    return pd.DataFrame(final_rows), pd.DataFrame(action_logs)


def eval_her(topk: pd.DataFrame) -> float:
    if len(topk) == 0:
        return 0.0
    return (topk["final_risk_label"] == "harmful_promotional").sum() / len(topk)


def eval_osr_against_baseline(
    baseline_topk: pd.DataFrame,
    policy_topk: pd.DataFrame,
) -> tuple[float, int, int]:
    baseline_ok = baseline_topk[
        baseline_topk["risk_label"].isin(["benign", "sensitive_educational"])
    ][["session_id", "item_id"]].drop_duplicates()

    policy_kept = policy_topk[["session_id", "final_item_id"]].drop_duplicates()
    policy_kept = policy_kept.rename(columns={"final_item_id": "item_id"})
    policy_kept["kept_in_policy"] = 1

    merged = baseline_ok.merge(policy_kept, on=["session_id", "item_id"], how="left")
    merged["kept_in_policy"] = merged["kept_in_policy"].fillna(0).astype(int)
    merged["suppressed"] = 1 - merged["kept_in_policy"]

    denom = len(merged)
    numer = int(merged["suppressed"].sum())

    if denom == 0:
        return 0.0, 0, 0

    return numer / denom, numer, denom


def eval_quality_on_normal_interest(topk: pd.DataFrame) -> tuple[float, float, int]:
    subset = topk[topk["intent_label"] == "normal_interest"].copy()

    hr_scores = []
    ndcg_scores = []
    n_eval = 0

    for _, group in subset.groupby("session_id", sort=False):
        labels = group.sort_values("rank_final")["clicked"].astype(int).tolist()
        if sum(labels) == 0:
            continue
        hr_scores.append(hitrate_at_k(labels, TOPK))
        ndcg_scores.append(ndcg_at_k(labels, TOPK))
        n_eval += 1

    if n_eval == 0:
        return 0.0, 0.0, 0

    return sum(hr_scores) / n_eval, sum(ndcg_scores) / n_eval, n_eval


def main():
    SWEEPS.mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading shared data ...")
    interactions, items, intent_probs, risk_probs = load_all()

    print("[STEP] Building shared baseline tables ...")
    pop_scores = build_pop_scores(interactions)
    dev_impr = prepare_dev_impressions(interactions, items, intent_probs, risk_probs, pop_scores)
    baseline_topk = build_baseline_topk(dev_impr)

    baseline_eval = baseline_topk.rename(columns={"rank_baseline": "rank_final"}).copy()
    base_hr, base_ndcg, _ = eval_quality_on_normal_interest(baseline_eval)
    base_her = eval_her(
        baseline_topk.rename(columns={"item_id": "final_item_id", "risk_label": "final_risk_label"})
    )

    rows = []

    for cfg in CONFIGS:
        print(f"[STEP] Running {cfg['config_name']} ...")
        topk, actions = run_policy(
            dev_impr=dev_impr,
            pop_scores=pop_scores,
            alpha=cfg["ALPHA"],
            beta=cfg["BETA"],
            gamma=cfg["GAMMA"],
        )

        her = eval_her(topk)
        osr, osr_num, osr_den = eval_osr_against_baseline(baseline_topk, topk)
        hr10, ndcg10, n_eval = eval_quality_on_normal_interest(topk)

        utility_loss = 0.0 if base_ndcg == 0 else (base_ndcg - ndcg10) / base_ndcg

        rows.append(
            {
                "config_name": cfg["config_name"],
                "ALPHA": cfg["ALPHA"],
                "BETA": cfg["BETA"],
                "GAMMA": cfg["GAMMA"],
                "HER": her,
                "HER_reduction_vs_baseline": base_her - her,
                "OSR": osr,
                "HitRate@10": hr10,
                "NDCG@10": ndcg10,
                "UtilityLoss_vs_Baseline": utility_loss,
                "EvalSessions": n_eval,
                "OSR_numerator": osr_num,
                "OSR_denominator": osr_den,
                "Action_keep": int((actions["action"] == "keep").sum()),
                "Action_down_rank": int((actions["action"] == "down_rank").sum()),
                "Action_replace": int((actions["action"] == "replace").sum()),
                "Action_block": int((actions["action"] == "block").sum()),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        by=["HER", "UtilityLoss_vs_Baseline", "OSR"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out_df.to_csv(OUT_PATH, index=False)

    print()
    print("[DONE] Saved:")
    print(f"  - {OUT_PATH}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
