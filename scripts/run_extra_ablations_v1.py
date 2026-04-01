from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/e/intent_aware_rec_defense")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_full_pipeline_v2 as fp  # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

OUT_PATH = TABLES / "extra_ablation_table_v1.csv"

TOPK = 10


def eval_quality_on_normal_interest(topk: pd.DataFrame) -> tuple[float, float, int]:
    subset = topk[topk["intent_label"] == "normal_interest"].copy()

    hr_scores = []
    ndcg_scores = []
    n_eval = 0

    for _, group in subset.groupby("session_id", sort=False):
        labels = group.sort_values("rank_final")["clicked"].astype(int).tolist()
        if sum(labels) == 0:
            continue
        hr_scores.append(1.0 if any(x > 0 for x in labels[:TOPK]) else 0.0)

        dcg = 0.0
        for i, rel in enumerate(labels[:TOPK], start=1):
            if rel > 0:
                dcg += rel / math.log2(i + 1)

        ideal_labels = sorted(labels, reverse=True)
        ideal = 0.0
        for i, rel in enumerate(ideal_labels[:TOPK], start=1):
            if rel > 0:
                ideal += rel / math.log2(i + 1)

        ndcg_scores.append(0.0 if ideal == 0 else dcg / ideal)
        n_eval += 1

    if n_eval == 0:
        return 0.0, 0.0, 0

    return sum(hr_scores) / n_eval, sum(ndcg_scores) / n_eval, n_eval


def build_baseline_reference():
    interactions, items, intent_probs, risk_probs = fp.load_all()
    pop_scores = fp.build_pop_scores(interactions)
    dev_impr = fp.prepare_dev_impressions(interactions, items, intent_probs, risk_probs, pop_scores)
    baseline_topk = fp.build_baseline_topk(dev_impr)

    baseline_eval = baseline_topk.rename(columns={"rank_baseline": "rank_final"}).copy()
    base_hr, base_ndcg, _ = eval_quality_on_normal_interest(baseline_eval)

    return interactions, items, intent_probs, risk_probs, pop_scores, dev_impr, baseline_topk, base_hr, base_ndcg


def build_safe_pool(dev_impr: pd.DataFrame, pop_scores: pd.Series) -> pd.DataFrame:
    pool = dev_impr[
        [
            "item_id", "source", "title", "text",
            "q_benign", "q_sensitive_educational", "q_harmful_promotional",
            "predicted_risk_label"
        ]
    ].drop_duplicates("item_id").copy()

    pool["safe_priority"] = np.where(
        pool["q_sensitive_educational"] >= fp.TAU_SAFE, 2,
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


def run_no_replacement(dev_impr: pd.DataFrame, pop_scores: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = dev_impr.copy()
    ranked["phi"] = ranked.apply(fp.compute_phi, axis=1)
    ranked["risk_penalty"] = ranked.apply(fp.compute_risk_penalty, axis=1)
    ranked["guidance_bonus"] = ranked.apply(fp.compute_guidance_bonus, axis=1)

    ranked["final_score_raw"] = (
        ranked["baseline_score_norm"]
        + fp.ALPHA * ranked["phi"]
        - fp.BETA * ranked["risk_penalty"]
        + fp.GAMMA * ranked["guidance_bonus"]
    )

    ranked = ranked.sort_values(
        by=["session_id", "final_score_raw", "position"],
        ascending=[True, False, True],
    ).copy()

    final_rows = []
    action_logs = []

    for session_id, group in ranked.groupby("session_id", sort=False):
        used_ids = set()
        final_rank = 0

        for row in group.itertuples(index=False):
            if final_rank >= TOPK:
                break

            q_h = float(row.q_harmful_promotional)
            risk_label = row.predicted_risk_label

            action = "keep"
            final_item_id = row.item_id
            final_risk_label = risk_label
            final_score = float(row.final_score_raw)

            if risk_label == "harmful_promotional":
                if q_h >= fp.TAU_BLOCK:
                    action = "block"
                elif q_h >= fp.TAU_REPLACE:
                    action = "block"   # 与 Full_Pipeline_V2 的区别：不 replace，直接 block
                elif q_h >= fp.TAU_DOWN:
                    action = "down_rank"
                    final_score = final_score - 0.15

            if action == "block":
                action_logs.append({
                    "session_id": session_id,
                    "original_item_id": row.item_id,
                    "final_item_id": None,
                    "action": "block",
                    "predicted_risk_label": risk_label,
                    "q_harmful_promotional": q_h,
                })
                continue

            if final_item_id in used_ids:
                continue

            final_rank += 1
            used_ids.add(final_item_id)

            final_rows.append({
                "session_id": session_id,
                "user_id": row.user_id,
                "split": row.split,
                "clicked": row.clicked,
                "rank_final": final_rank,
                "original_item_id": row.item_id,
                "final_item_id": final_item_id,
                "action": action,
                "intent_label": row.intent_label,
                "risk_label": row.risk_label,
                "final_risk_label": final_risk_label,
            })

            action_logs.append({
                "session_id": session_id,
                "original_item_id": row.item_id,
                "final_item_id": final_item_id,
                "action": action,
                "predicted_risk_label": risk_label,
                "q_harmful_promotional": q_h,
            })

    return pd.DataFrame(final_rows), pd.DataFrame(action_logs)


def run_global_threshold(dev_impr: pd.DataFrame, pop_scores: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe_pool = build_safe_pool(dev_impr, pop_scores)

    ranked = dev_impr.copy()
    ranked["phi"] = ranked.apply(fp.compute_phi, axis=1)
    ranked["risk_penalty"] = ranked.apply(fp.compute_risk_penalty, axis=1)
    ranked["guidance_bonus"] = ranked.apply(fp.compute_guidance_bonus, axis=1)

    ranked["final_score_raw"] = (
        ranked["baseline_score_norm"]
        + fp.ALPHA * ranked["phi"]
        - fp.BETA * ranked["risk_penalty"]
        + fp.GAMMA * ranked["guidance_bonus"]
    )

    ranked = ranked.sort_values(
        by=["session_id", "final_score_raw", "position"],
        ascending=[True, False, True],
    ).copy()

    final_rows = []
    action_logs = []

    GLOBAL_TAU = 0.60

    for session_id, group in ranked.groupby("session_id", sort=False):
        used_ids = set()
        final_rank = 0

        for row in group.itertuples(index=False):
            if final_rank >= TOPK:
                break

            q_h = float(row.q_harmful_promotional)
            risk_label = row.predicted_risk_label

            action = "keep"
            final_item_id = row.item_id
            final_risk_label = risk_label

            if risk_label == "harmful_promotional" and q_h >= GLOBAL_TAU:
                repl_id, repl_risk = pick_replacement(used_ids, safe_pool)
                if repl_id is not None:
                    action = "replace"
                    final_item_id = repl_id
                    final_risk_label = repl_risk if repl_risk is not None else "benign"
                else:
                    action = "block"

            if action == "block":
                action_logs.append({
                    "session_id": session_id,
                    "original_item_id": row.item_id,
                    "final_item_id": None,
                    "action": "block",
                    "predicted_risk_label": risk_label,
                    "q_harmful_promotional": q_h,
                })
                continue

            if final_item_id in used_ids:
                continue

            final_rank += 1
            used_ids.add(final_item_id)

            final_rows.append({
                "session_id": session_id,
                "user_id": row.user_id,
                "split": row.split,
                "clicked": row.clicked if final_item_id == row.item_id else 0,
                "rank_final": final_rank,
                "original_item_id": row.item_id,
                "final_item_id": final_item_id,
                "action": action,
                "intent_label": row.intent_label,
                "risk_label": row.risk_label,
                "final_risk_label": final_risk_label,
            })

            action_logs.append({
                "session_id": session_id,
                "original_item_id": row.item_id,
                "final_item_id": final_item_id,
                "action": action,
                "predicted_risk_label": risk_label,
                "q_harmful_promotional": q_h,
            })

    return pd.DataFrame(final_rows), pd.DataFrame(action_logs)


def summarize(system_name, final_topk, actions, baseline_topk, baseline_ndcg):
    her = fp.eval_her(final_topk)
    osr, osr_num, osr_den = fp.eval_osr_against_baseline(baseline_topk, final_topk)
    hr10, ndcg10, n_eval = eval_quality_on_normal_interest(final_topk)

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
        "Action_keep": int((actions["action"] == "keep").sum()),
        "Action_down_rank": int((actions["action"] == "down_rank").sum()),
        "Action_replace": int((actions["action"] == "replace").sum()),
        "Action_block": int((actions["action"] == "block").sum()),
    }


def main():
    TABLES.mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading shared data ...")
    interactions, items, intent_probs, risk_probs, pop_scores, dev_impr, baseline_topk, base_hr, base_ndcg = build_baseline_reference()

    print("[STEP] Running No_Replacement ablation ...")
    nr_topk, nr_actions = run_no_replacement(dev_impr, pop_scores)

    print("[STEP] Running Global_Threshold ablation ...")
    gt_topk, gt_actions = run_global_threshold(dev_impr, pop_scores)

    rows = [
        summarize("No_Replacement_V1", nr_topk, nr_actions, baseline_topk, base_ndcg),
        summarize("Global_Threshold_V1", gt_topk, gt_actions, baseline_topk, base_ndcg),
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    print()
    print("[DONE] Saved:")
    print(f"  {OUT_PATH}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()