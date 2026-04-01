from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

OLD_MAIN = TABLES / "thesis_main_results_v2.csv"
LIGHT_COMPARE = TABLES / "full_pipeline_v2_light_compare.csv"

OUT_MAIN = TABLES / "thesis_main_results_v3.csv"
OUT_SUMMARY = RESULTS / "chapter4_5_summary_v3.md"


def main():
    print("[STEP] Loading existing result tables ...")
    old_main = pd.read_csv(OLD_MAIN)
    light_cmp = pd.read_csv(LIGHT_COMPARE)

    light_cmp = light_cmp.copy()
    light_cmp["Notes"] = [
        "Original thesis-style full pipeline with old intent model",
        "New intent_model_v2_light with best-NDCG weight configuration",
        "New intent_model_v2_light with balanced weight configuration",
    ]

    keep_cols = [
        "System",
        "HER",
        "HitRate@10",
        "NDCG@10",
        "UtilityLoss_vs_Baseline",
        "EvalSessions",
        "Action_keep",
        "Action_down_rank",
        "Action_replace",
        "Action_block",
        "Notes",
    ]

    light_cmp = light_cmp[keep_cols]

    old_main["Action_keep"] = None
    old_main["Action_down_rank"] = None
    old_main["Action_replace"] = None
    old_main["Action_block"] = None

    old_main = old_main[
        [
            "System",
            "HER",
            "HitRate@10",
            "NDCG@10",
            "UtilityLoss_vs_Baseline",
            "EvalSessions",
            "Action_keep",
            "Action_down_rank",
            "Action_replace",
            "Action_block",
            "Notes",
        ]
    ]

    out = pd.concat([old_main, light_cmp], ignore_index=True)

    order = [
        "Injected_Baseline",
        "Risk_Only_V1",
        "Intent_Only_V1",
        "Full_Policy_V1",
        "No_Replacement_V1",
        "Global_Threshold_V1",
        "Full_Pipeline_V2",
        "Balanced_Mode_Selected",
        "High_Safety_Mode_Selected",
        "Full_Pipeline_V2_Current",
        "Full_Pipeline_V2_LightIntent_BestNDCG",
        "Full_Pipeline_V2_LightIntent_Balanced",
    ]
    out["sort_key"] = out["System"].map({name: i for i, name in enumerate(order)})
    out = out.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    out.to_csv(OUT_MAIN, index=False)

    baseline = out[out["System"] == "Full_Pipeline_V2_Current"].iloc[0]
    best_ndcg = out[out["System"] == "Full_Pipeline_V2_LightIntent_BestNDCG"].iloc[0]
    balanced = out[out["System"] == "Full_Pipeline_V2_LightIntent_Balanced"].iloc[0]

    summary = f"""# Chapter 4 and Chapter 5 Writing Summary V3

## Intent Model Upgrade Effect
The upgraded session-level intent model improved downstream ranking utility without changing harmful-exposure suppression.

### Baseline Full Pipeline with old intent model
- HER = {baseline['HER']:.6f}
- NDCG@10 = {baseline['NDCG@10']:.6f}
- UtilityLoss_vs_Baseline = {baseline['UtilityLoss_vs_Baseline']:.6f}

### New intent_model_v2_light with best-NDCG configuration
- HER = {best_ndcg['HER']:.6f}
- NDCG@10 = {best_ndcg['NDCG@10']:.6f}
- UtilityLoss_vs_Baseline = {best_ndcg['UtilityLoss_vs_Baseline']:.6f}

### New intent_model_v2_light with balanced configuration
- HER = {balanced['HER']:.6f}
- NDCG@10 = {balanced['NDCG@10']:.6f}
- UtilityLoss_vs_Baseline = {balanced['UtilityLoss_vs_Baseline']:.6f}

## Interpretation
The upgraded intent model increased NDCG@10 while HER remained unchanged. This suggests that better intent estimation mainly improved ranking calibration among retained candidates, while harmful suppression continued to be dominated by the hard threshold policy.

## Thesis Writing Note
This result is useful because it shows that improving the intent layer is not only a standalone classification improvement. It also yields measurable downstream benefit in recommendation quality, even when the intervention structure is unchanged.
"""
    OUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print("[DONE] Saved:")
    print(f"  - {OUT_MAIN}")
    print(f"  - {OUT_SUMMARY}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
