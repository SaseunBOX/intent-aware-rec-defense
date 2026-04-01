from __future__ import annotations
#探索不同 intervention 强度下的 trade-off，不只给一个结果，而是给一个安全—效用前沿。
#Explore the trade-offs under different intervention intensities. Instead of providing a single result, present a safety-utility frontier.
import sys
from pathlib import Path

import pandas as pd


#This script does not change the model.
#  It changes the intervention thresholds to study the safety–utility frontier. 
# So instead of asking whether the system works at one point, it asks what trade-offs it can support.
# 本脚本不会改变模型。
# 它会调整干预阈值，以研究安全与效益的边界。
# 因此，它不再只是询问系统在某一时刻是否有效，而是询问它能够支持何种权衡取舍。
#命令python run_threshold_sweep_v2.py
#运行后会得到：
#results/sweeps/full_pipeline_v2_threshold_sweep.csv
#然后运行：
#python make_tradeoff_plots_v2.py
#它会生成：
#her_vs_ndcg_loss.png
#her_vs_osr.png
#osr_vs_ndcg_loss.png
#ablation_her_osr.png
#These plots visualize the trade-off frontier. 
# Lower HER means stronger harmful-exposure suppression, 
# #while higher NDCG loss means more utility degradation. 
# The goal is not to maximize one metric in isolation, 
# but to find an operating point that balances safety and recommendation quality.
#论文里也明确说 sweep 是为了选 Balanced mode 和 High-safety mode。
ROOT = Path("/mnt/e/intent_aware_rec_defense")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_full_pipeline_v2 as fp  # noqa: E402

SWEEPS = ROOT / "results" / "sweeps"
OUT_PATH = SWEEPS / "full_pipeline_v2_threshold_sweep.csv"


CONFIGS = [
    {"config_name": "cfg_1_lenient",   "tau_down": 0.45, "tau_replace": 0.65, "tau_block": 0.85},
    {"config_name": "cfg_2_current",   "tau_down": 0.35, "tau_replace": 0.55, "tau_block": 0.75},
    {"config_name": "cfg_3_balanced",  "tau_down": 0.30, "tau_replace": 0.50, "tau_block": 0.70},
    {"config_name": "cfg_4_safer",     "tau_down": 0.25, "tau_replace": 0.45, "tau_block": 0.65},
    {"config_name": "cfg_5_aggressive","tau_down": 0.20, "tau_replace": 0.40, "tau_block": 0.60},
]


def eval_baseline_her(baseline_topk: pd.DataFrame) -> float:
    if len(baseline_topk) == 0:
        return 0.0
    harmful = (baseline_topk["risk_label"] == "harmful_promotional").sum()
    return harmful / len(baseline_topk)


def main() -> None:
    SWEEPS.mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading shared data once ...")
    interactions, items, intent_probs, risk_probs = fp.load_all()

    print("[STEP] Building shared baseline tables once ...")
    pop_scores = fp.build_pop_scores(interactions)
    dev_impr = fp.prepare_dev_impressions(
        interactions, items, intent_probs, risk_probs, pop_scores
    )
    baseline_topk = fp.build_baseline_topk(dev_impr)

    baseline_eval_topk = baseline_topk.rename(columns={"rank_baseline": "rank_final"}).copy()
    base_hr, base_ndcg, base_eval_sessions = fp.eval_quality_on_normal_interest(baseline_eval_topk)
    base_her = eval_baseline_her(baseline_topk)

    print("[BASELINE REFERENCE]")
    print(f"  HER        = {base_her:.6f}")
    print(f"  HitRate@10 = {base_hr:.6f}")
    print(f"  NDCG@10    = {base_ndcg:.6f}")
    print(f"  EvalSess   = {base_eval_sessions:,}")
    print()

    rows = []

    for cfg in CONFIGS:
        print(f"[STEP] Running sweep config: {cfg['config_name']} ...")

        fp.TAU_DOWN = cfg["tau_down"]
        fp.TAU_REPLACE = cfg["tau_replace"]
        fp.TAU_BLOCK = cfg["tau_block"]

        final_topk, actions = fp.run_policy(dev_impr, pop_scores)

        her = fp.eval_her(final_topk)
        osr, osr_num, osr_den = fp.eval_osr_against_baseline(baseline_topk, final_topk)
        hr10, ndcg10, n_eval = fp.eval_quality_on_normal_interest(final_topk)

        utility_loss = 0.0
        if base_ndcg > 0:
            utility_loss = (base_ndcg - ndcg10) / base_ndcg

        her_reduction = base_her - her

        rows.append({
            "config_name": cfg["config_name"],
            "tau_down": cfg["tau_down"],
            "tau_replace": cfg["tau_replace"],
            "tau_block": cfg["tau_block"],
            "HER": her,
            "HER_reduction_vs_baseline": her_reduction,
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
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        by=["HER", "UtilityLoss_vs_Baseline", "OSR"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out_df.to_csv(OUT_PATH, index=False)

    print()
    print("[DONE] Sweep results saved:")
    print(f"  {OUT_PATH}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()