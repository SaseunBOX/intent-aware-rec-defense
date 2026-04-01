from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

MAIN_V2_PATH = RESULTS / "main_metrics_table_v2.csv"
FULL_V2_PATH = TABLES / "full_pipeline_v2_metrics.csv"
OPS_PATH = TABLES / "operating_points_v1.csv"
INTENT_METRICS_PATH = TABLES / "intent_model_v1_metrics.csv"
RISK_METRICS_PATH = TABLES / "risk_model_v1_metrics.csv"

OUT_MAIN_RESULTS = TABLES / "thesis_main_results_v1.csv"
OUT_MODEL_QUALITY = TABLES / "model_quality_summary_v1.csv"
OUT_SUMMARY_MD = RESULTS / "chapter4_5_summary_v1.md"


def round_cols(df: pd.DataFrame, cols: list[str], ndigits: int = 6) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(ndigits)
    return df


def load_tables():
    main_v2 = pd.read_csv(MAIN_V2_PATH)
    full_v2 = pd.read_csv(FULL_V2_PATH)
    ops = pd.read_csv(OPS_PATH)
    intent_metrics = pd.read_csv(INTENT_METRICS_PATH)
    risk_metrics = pd.read_csv(RISK_METRICS_PATH)
    return main_v2, full_v2, ops, intent_metrics, risk_metrics


def build_main_results(main_v2: pd.DataFrame, full_v2: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "System",
        "HER",
        "OSR",
        "HitRate@10",
        "NDCG@10",
        "UtilityLoss_vs_Baseline",
        "EvalSessions",
        "OSR_numerator",
        "OSR_denominator",
    ]

    ablation = main_v2[keep_cols].copy()
    ablation["Notes"] = [
        "Injected baseline after external candidate injection",
        "Risk-only control",
        "Intent-only control",
        "Rule-based full policy v1",
    ]

    full_row = full_v2[keep_cols].copy()
    full_row["Notes"] = "Thesis-style full pipeline v2 with phi/penalty/bonus and hard actions"

    ops_rows = ops.copy()
    ops_rows["System"] = ops_rows["mode"].replace(
        {
            "Balanced_mode": "Balanced_Mode_Selected",
            "High_safety_mode": "High_Safety_Mode_Selected",
        }
    )
    ops_rows["Notes"] = ops_rows["notes"]
    ops_rows = ops_rows[
        [
            "System",
            "HER",
            "OSR",
            "HitRate@10",
            "NDCG@10",
            "UtilityLoss_vs_Baseline",
            "EvalSessions",
            "OSR_numerator",
            "OSR_denominator",
            "Notes",
        ]
    ].copy()

    full_row = full_row[
        [
            "System",
            "HER",
            "OSR",
            "HitRate@10",
            "NDCG@10",
            "UtilityLoss_vs_Baseline",
            "EvalSessions",
            "OSR_numerator",
            "OSR_denominator",
            "Notes",
        ]
    ]

    ablation = ablation[
        [
            "System",
            "HER",
            "OSR",
            "HitRate@10",
            "NDCG@10",
            "UtilityLoss_vs_Baseline",
            "EvalSessions",
            "OSR_numerator",
            "OSR_denominator",
            "Notes",
        ]
    ]

    out = pd.concat([ablation, full_row, ops_rows], ignore_index=True)

    order = [
        "Injected_Baseline",
        "Risk_Only_V1",
        "Intent_Only_V1",
        "Full_Policy_V1",
        "Full_Pipeline_V2",
        "Balanced_Mode_Selected",
        "High_Safety_Mode_Selected",
    ]
    out["sort_key"] = out["System"].map({name: i for i, name in enumerate(order)})
    out = out.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    out = round_cols(
        out,
        ["HER", "OSR", "HitRate@10", "NDCG@10", "UtilityLoss_vs_Baseline"],
        ndigits=6,
    )
    return out


def build_model_quality(intent_metrics: pd.DataFrame, risk_metrics: pd.DataFrame) -> pd.DataFrame:
    intent_acc = intent_metrics[intent_metrics["label"] == "overall_accuracy"].copy()
    risk_acc = risk_metrics[risk_metrics["label"] == "overall_accuracy"].copy()

    rows = []

    if not intent_acc.empty:
        rows.append(
            {
                "model_name": "intent_model_v1",
                "overall_accuracy": float(intent_acc.iloc[0]["precision"]),
                "notes": "Session-level intent probability model",
            }
        )

    if not risk_acc.empty:
        rows.append(
            {
                "model_name": "risk_model_v1",
                "overall_accuracy": float(risk_acc.iloc[0]["precision"]),
                "notes": "Item-level risk probability model",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["overall_accuracy"] = out["overall_accuracy"].round(6)
    return out


def build_summary_markdown(main_results: pd.DataFrame, model_quality: pd.DataFrame) -> str:
    baseline = main_results[main_results["System"] == "Injected_Baseline"].iloc[0]
    full_v2 = main_results[main_results["System"] == "Full_Pipeline_V2"].iloc[0]
    balanced = main_results[main_results["System"] == "Balanced_Mode_Selected"].iloc[0]
    high_safe = main_results[main_results["System"] == "High_Safety_Mode_Selected"].iloc[0]

    her_reduction_full = round(float(baseline["HER"]) - float(full_v2["HER"]), 6)
    ndcg_loss_full = round(float(full_v2["UtilityLoss_vs_Baseline"]), 6)

    lines = []
    lines.append("# Chapter 4 and Chapter 5 Writing Summary V1")
    lines.append("")
    lines.append("## 1. Model Quality Summary")
    if not model_quality.empty:
        for row in model_quality.itertuples(index=False):
            lines.append(
                f"- {row.model_name}: overall accuracy = {row.overall_accuracy:.6f} ({row.notes})"
            )
    else:
        lines.append("- Model quality summary unavailable.")
    lines.append("")
    lines.append("## 2. Main Experimental Findings")
    lines.append(
        f"- The injected baseline produced HER = {baseline['HER']:.6f}, OSR = {baseline['OSR']:.6f}, "
        f"HitRate@10 = {baseline['HitRate@10']:.6f}, and NDCG@10 = {baseline['NDCG@10']:.6f}."
    )
    lines.append(
        f"- The thesis-style Full Pipeline V2 reduced HER to {full_v2['HER']:.6f}, "
        f"achieving a HER reduction of {her_reduction_full:.6f} relative to the injected baseline."
    )
    lines.append(
        f"- Under Full Pipeline V2, OSR = {full_v2['OSR']:.6f}, "
        f"NDCG@10 = {full_v2['NDCG@10']:.6f}, and utility loss vs baseline = {ndcg_loss_full:.6f}."
    )
    lines.append(
        "- Compared with the injected baseline, the full pipeline substantially reduced harmful exposure while preserving a high recommendation utility level."
    )
    lines.append("")
    lines.append("## 3. Ablation Interpretation")
    lines.append(
        "- The risk-only control reduced HER more strongly than the intent-only control in the current setup."
    )
    lines.append(
        "- The intent-only control changed OSR more noticeably than the risk-only control, indicating that user-intent-sensitive routing mainly affected acceptable-content retention."
    )
    lines.append(
        "- The earlier Full_Policy_V1 provided a useful bridge from simple rule-based reranking to the more thesis-aligned Full Pipeline V2."
    )
    lines.append("")
    lines.append("## 4. Operating Points")
    lines.append(
        f"- Balanced mode selected: {balanced['System']} with HER = {balanced['HER']:.6f}, "
        f"OSR = {balanced['OSR']:.6f}, NDCG@10 = {balanced['NDCG@10']:.6f}."
    )
    lines.append(
        f"- High-safety mode selected: {high_safe['System']} with HER = {high_safe['HER']:.6f}, "
        f"OSR = {high_safe['OSR']:.6f}, NDCG@10 = {high_safe['NDCG@10']:.6f}."
    )
    lines.append(
        "- In the current sweep, the two operating points coincided, suggesting that the tested search range already approached a dominant safety-preferred region of the trade-off frontier."
    )
    lines.append("")
    lines.append("## 5. Suggested Chapter 5 Conclusion Points")
    lines.append(
        "- The proposed intent-aware multi-stage defense can sharply suppress harmful exposure in offline recommendation simulation."
    )
    lines.append(
        "- The strongest gains come from combining probabilistic intent estimation, probabilistic risk estimation, score fusion, and hard intervention actions."
    )
    lines.append(
        "- The current prototype demonstrates feasibility, but future work should expand the safe replacement pool, refine classifier calibration, and broaden the sweep range for more clearly separated operating modes."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("[STEP] Loading existing result tables ...")
    main_v2, full_v2, ops, intent_metrics, risk_metrics = load_tables()

    print("[STEP] Building thesis main results table ...")
    thesis_main = build_main_results(main_v2, full_v2, ops)
    thesis_main.to_csv(OUT_MAIN_RESULTS, index=False)

    print("[STEP] Building model quality summary table ...")
    model_quality = build_model_quality(intent_metrics, risk_metrics)
    model_quality.to_csv(OUT_MODEL_QUALITY, index=False)

    print("[STEP] Writing Chapter 4/5 summary markdown ...")
    summary_md = build_summary_markdown(thesis_main, model_quality)
    OUT_SUMMARY_MD.write_text(summary_md, encoding="utf-8")

    print()
    print("[DONE] Result pack saved:")
    print(f"  - {OUT_MAIN_RESULTS}")
    print(f"  - {OUT_MODEL_QUALITY}")
    print(f"  - {OUT_SUMMARY_MD}")
    print()
    print("[THESIS MAIN RESULTS]")
    print(thesis_main.to_string(index=False))
    print()
    print("[MODEL QUALITY]")
    print(model_quality.to_string(index=False))


if __name__ == "__main__":
    main()