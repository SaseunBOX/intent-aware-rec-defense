from __future__ import annotations
#把 baseline、intent probability、risk probability 融合，
# 输出最终防御后的 top-k 结果，并且给出 keep/down-rank/replace/block 的动作日志。
## Merge the baseline, intent probability, and risk probability,
# Output the final defense-based top-k results, 
# and provide the action logs of keep/down-rank/replace/block.
import math
from pathlib import Path

import numpy as np
import pandas as pd
#The full pipeline starts from a normalized baseline ranking score, 
# then fuses three safety-aware terms: policy compatibility, 
# harmful-risk penalty, and supportive-guidance bonus. 
# Finally, it applies hard ranking-time actions such as keep, down-rank, replace, and block.
# 整个流程始于一个标准化的基线排名分数，
# 然后融合了三个具有安全意识的要素：政策兼容性、有害风险惩罚以及支持性指导奖励。
# 最后，它会执行一些严格的排名时间操作，如保留、降低排名、替换和阻止。
#运行
#python run_full_pipeline_v2.py
#主函数会做这些事：
#载入 interactions_injected.csv、items.csv
#载入 intent_probabilities_all.csv、risk_probabilities_all.csv
#构建 baseline top-k
#运行 thesis-style full pipeline v2
#输出三个关键文件：
#full_pipeline_v2_metrics.csv
#full_pipeline_v2_actions.csv
#full_pipeline_v2_topk.csv

ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

INTERACTIONS_PATH = PROCESSED / "interactions_injected.csv"
ITEMS_PATH = PROCESSED / "items.csv"

INTENT_PROBS_PATH = MODELS / "intent" / "intent_probabilities_all.csv"
RISK_PROBS_PATH = MODELS / "risk" / "risk_probabilities_all.csv"

TABLE_OUT = RESULTS / "tables" / "full_pipeline_v2_metrics.csv"
ACTION_LOG_OUT = RESULTS / "tables" / "full_pipeline_v2_actions.csv"
TOPK_OUT = RESULTS / "tables" / "full_pipeline_v2_topk.csv"

TOPK = 10

# ===== Thesis-style parameters =====
# The alpha, beta, gamma parameters are set based on dev set tuning 
# to achieve a good balance between safety and utility.
# ALPHA controls the weight of the policy compatibility term,
# which encourages the model to align with the safety policy's intent-risk compatibility matrix.
# ALPHA 参数控制政策兼容性项的权重，
# 该项鼓励模型与安全政策的意图-风险兼容矩阵保持一致。
ALPHA = 0.20   # policy compatibility weight
# The harmful penalty weight (BETA) and guidance bonus weight (GAMMA) are tuned
# to achieve a good balance between safety and utility on the dev set.
# 有害惩罚权重（BETA）和指导奖励权重（GAMMA）经过调整，以在开发集上实现安全性和实用性之间的良好平衡。
BETA = 0.60    # harmful penalty weight
# The guidance bonus weight (GAMMA) is set to a moderate value to encourage 
# the model to promote items that are both relevant to sensitive 
# help-seeking intent and have educational risk profiles, without overpowering the harmful penalty.
#指导奖励权重（GAMMA）设置为一个适中的值，
# 以鼓励模型推广那些既与敏感求助意图相关又具有教育性风险特征的项目，同时不过分削弱有害惩罚的影响。
GAMMA = 0.15   # guidance bonus weight
# The risk penalty weights (W1, W2, W3) are set to reflect the increasing severity of the user's intent.
# For example, W3 is the highest because clearly harmful intent should be penalized more when
# the item has a high harmful risk score.
#风险惩罚权重（W1、W2、W3）设置为反映用户意图的严重程度递增。
# 例如，W3 是最高的，因为当项目具有高有害风险分数时，明显有害的意图应该受到更多的惩罚。
# harmful penalty weights: w3 >= w2 >= w1
W1 = 0.8
W2 = 1.4
W3 = 2.0
# The tau thresholds for different actions are set based 
# on dev set analysis of the q_harmful_promotional distribution.
# For example, TAU_BLOCK is set to 0.75 because items with q_harmful_promotional 
# above this threshold are very likely to be truly harmful, 
# and blocking them provides a good safety boost with acceptable utility loss.
# 不同操作的 tau 阈值是基于开发集对 q_harmful_promotional 分布的分析设置的。
# 例如，TAU_BLOCK 设置为 0.75，因为 q_harmful_promotional 
# 超过此阈值的项目很可能是真正有害的，阻止它们可以提供良好的安全提升，同时具有可接受的实用性损失。
TAU_DOWN = 0.35
TAU_REPLACE = 0.55
TAU_BLOCK = 0.75
TAU_SAFE = 0.40

#
#
#
#
#

# Intent order:
# 0 normal_interest
# 1 sensitive_help_seeking
# 2 clearly_harmful_intent
#
# Risk order:
# 0 benign
# 1 sensitive_educational
# 2 harmful_promotional

#This is the policy matrix.
#It maps the compatibility between user-intent
#  probabilities and item-risk probabilities.
#This makes the policy layer more structured than 
# a simple keyword filter or a single threshold rule.
#这是政策矩阵。
#它映射了用户意图概率和项目风险概率之间的兼容性。
#这使得政策层比简单的关键词过滤或单一阈值规则更有结构性。
M = np.array([
    [ 1.0,  0.3, -1.0],   # normal_interest
    [ 0.8,  1.2, -1.5],   # sensitive_help_seeking
    [ 0.2, -0.2, -2.0],   # clearly_harmful_intent
], dtype=float)


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
    harmful = (topk["final_risk_label"] == "harmful_promotional").sum()
    return harmful / len(topk)


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

#Here I compute the policy compatibility term, the harmful-risk penalty, and the guidance bonus.
#These three terms are then combined with the normalized baseline score.
#在这里，我计算了政策兼容性项、有害风险惩罚和指导奖励。
#这三个项然后与标准化的基线分数结合起来。
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
#############
#The final ranking score combines baseline relevance, policy compatibility, 
# harmful-risk suppression, and supportive guidance preference.
#最终排名分数结合了基线相关性、政策兼容性、有害风险抑制和支持性指导偏好。

def build_baseline_topk(dev_impr: pd.DataFrame) -> pd.DataFrame:
    ranked = dev_impr.sort_values(
        by=["session_id", "baseline_score", "position"],
        ascending=[True, False, True],
    ).copy()
    ranked["rank_baseline"] = ranked.groupby("session_id").cumcount() + 1
    return ranked[ranked["rank_baseline"] <= TOPK].copy()


def build_safe_pool(dev_impr: pd.DataFrame, pop_scores: pd.Series) -> pd.DataFrame:
    pool = dev_impr[[
        "item_id", "source", "title", "text",
        "q_benign", "q_sensitive_educational", "q_harmful_promotional",
        "predicted_risk_label"
    ]].drop_duplicates("item_id").copy()

    # 候选安全替代池：优先 sensitive_educational，再回退 benign
    pool["safe_priority"] = np.where(
        pool["q_sensitive_educational"] >= TAU_SAFE, 2,
        np.where(pool["q_benign"] >= 0.5, 1, 0)
    )
    pool["baseline_score"] = pool["item_id"].map(pop_scores).fillna(0)
    pool = pool[pool["safe_priority"] > 0].copy()
    pool = pool.sort_values(
        by=["safe_priority", "baseline_score"],
        ascending=[False, False],
    )
    return pool


def pick_replacement(
    used_ids: set[str],
    safe_pool: pd.DataFrame,
) -> tuple[str | None, str | None]:
    for row in safe_pool.itertuples(index=False):
        if row.item_id in used_ids:
            continue
        return row.item_id, row.predicted_risk_label
    return None, None
####################

def run_policy(dev_impr: pd.DataFrame, pop_scores: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe_pool = build_safe_pool(dev_impr, pop_scores)

    ranked = dev_impr.copy()
    ranked["phi"] = ranked.apply(compute_phi, axis=1)
    ranked["risk_penalty"] = ranked.apply(compute_risk_penalty, axis=1)
    ranked["guidance_bonus"] = ranked.apply(compute_guidance_bonus, axis=1)

    ranked["final_score_raw"] = (
        ranked["baseline_score_norm"]
        + ALPHA * ranked["phi"]
        - BETA * ranked["risk_penalty"]
        + GAMMA * ranked["guidance_bonus"]
    )

    ranked = ranked.sort_values(
        by=["session_id", "final_score_raw", "position"],
        ascending=[True, False, True],
    ).copy()

    action_logs = []
    final_rows = []
#对于每个 session 的 impression，按照 final_score_raw 排序后，逐行应用 policy：
#如果预测为 harmful_promotional 且 q_harmful_promotional >= TAU_BLOCK，则 block；
#如果预测为 harmful_promotional 且 TAU_REPLACE <= q_harmful_prom    otional < TAU_BLOCK，则尝试替换，如果 safe_pool 中有未使用的安全替代品，则 replace，否则 block；
#如果预测为 harmful_promotional 且 TAU_DOWN <= q_harmful_promotional < TAU_REPLACE，则 down-rank（final_score 减去一个固定值，使其在排序中下降）；
#否则 keep 原 item。
# For replacement, I pick the highest baseline popularity item from the safe pool that has not been used in the current session.
# I log the action taken for each item in the session, along with the original and final item IDs, the predicted risk label, and the q_harmful_promotional score, to allow for detailed analysis of the policy's behavior.
# For each impression of each session, after sorting by final_score_raw, I apply the policy row by row:
# If predicted as harmful_promotional and q_harmful_promotional >= TAU_BLOCK, then block;
# If predicted as harmful_promotional and TAU_REPLACE <= q_harmful_promotional < TAU_BLOCK, then try to replace, if there is an unused safe alternative in the safe_pool, then replace, otherwise block;
# If predicted as harmful_promotional and TAU_DOWN <= q_harmful_promotional < TAU_REPLACE, then down-rank (subtract a fixed value from final_score to make it drop in the ranking);
# Otherwise keep the original item.
###
#This is the intervention stage.
#Depending on the predicted harmful probability and the thresholds, 
# a candidate item can be kept, down-ranked, replaced with a safer item, or blocked entirely.
#This is the most important implementation part of the defense pipeline.


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
            final_score = float(row.final_score_raw)

            if risk_label == "harmful_promotional":
                if q_h >= TAU_BLOCK:
                    action = "block"

                elif q_h >= TAU_REPLACE:
                    repl_id, repl_risk = pick_replacement(used_ids, safe_pool)
                    if repl_id is not None:
                        action = "replace"
                        final_item_id = repl_id
                        final_risk_label = repl_risk if repl_risk is not None else "benign"
                        final_score = final_score + 0.05
                    else:
                        action = "block"

                elif q_h >= TAU_DOWN:
                    action = "down_rank"
                    final_score = final_score - 0.15

            if action == "block":
                action_logs.append({
                    "session_id": session_id,
                    "original_item_id": item_id,
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
                "clicked": row.clicked if final_item_id == item_id else 0,
                "rank_final": final_rank,
                "original_item_id": item_id,
                "final_item_id": final_item_id,
                "action": action,
                "baseline_score": row.baseline_score,
                "baseline_score_norm": row.baseline_score_norm,
                "phi": row.phi,
                "risk_penalty": row.risk_penalty,
                "guidance_bonus": row.guidance_bonus,
                "final_score": final_score,
                "intent_label": row.intent_label,
                "predicted_intent_label": row.predicted_intent_label,
                "risk_label": row.risk_label,
                "predicted_risk_label": row.predicted_risk_label,
                "final_risk_label": final_risk_label,
            })

            action_logs.append({
                "session_id": session_id,
                "original_item_id": item_id,
                "final_item_id": final_item_id,
                "action": action,
                "predicted_risk_label": risk_label,
                "q_harmful_promotional": q_h,
            })

    topk = pd.DataFrame(final_rows)
    actions = pd.DataFrame(action_logs)
    return topk, actions


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


def main() -> None:
    RESULTS.joinpath("tables").mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading data ...")
    interactions, items, intent_probs, risk_probs = load_all()

    print("[STEP] Building baseline popularity scores ...")
    pop_scores = build_pop_scores(interactions)

    print("[STEP] Preparing dev impression table ...")
    dev_impr = prepare_dev_impressions(interactions, items, intent_probs, risk_probs, pop_scores)

    print("[STEP] Building baseline top-k ...")
    baseline_topk = build_baseline_topk(dev_impr)

    print("[STEP] Running thesis-style full pipeline v2 ...")
    final_topk, actions = run_policy(dev_impr, pop_scores)

    print("[STEP] Evaluating metrics ...")
    her = eval_her(final_topk)
    osr, osr_num, osr_den = eval_osr_against_baseline(baseline_topk, final_topk)
    hr10, ndcg10, n_eval = eval_quality_on_normal_interest(final_topk)

    baseline_hr, baseline_ndcg, _ = eval_quality_on_normal_interest(
        baseline_topk.rename(columns={"rank_baseline": "rank_final"})
    )

    utility_loss = 0.0
    if baseline_ndcg > 0:
        utility_loss = (baseline_ndcg - ndcg10) / baseline_ndcg

    metrics_df = pd.DataFrame([{
        "System": "Full_Pipeline_V2",
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
    }])

    metrics_df.to_csv(TABLE_OUT, index=False)
    actions.to_csv(ACTION_LOG_OUT, index=False)
    final_topk.to_csv(TOPK_OUT, index=False)

    print()
    print("[DONE] Full pipeline v2 outputs saved:")
    print(f"  - {TABLE_OUT}")
    print(f"  - {ACTION_LOG_OUT}")
    print(f"  - {TOPK_OUT}")
    print()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()