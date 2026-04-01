
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path("/mnt/e/intent_aware_rec_defense")
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "tables"

INTENT_PROBS_PATH = MODELS / "intent" / "intent_probabilities_all.csv"
RISK_PROBS_PATH = MODELS / "risk" / "risk_probabilities_all.csv"

INTENT_DETAILED_OUT = RESULTS / "intent_model_v2_detailed_metrics.csv"
RISK_DETAILED_OUT = RESULTS / "risk_model_v2_detailed_metrics.csv"
CALIBRATION_OUT = RESULTS / "model_calibration_summary_v1.csv"

INTENT_PRED_DIST_OUT = RESULTS / "intent_prediction_distribution_v1.csv"
RISK_PRED_DIST_OUT = RESULTS / "risk_prediction_distribution_v1.csv"

INTENT_UNCERTAIN_OUT = RESULTS / "intent_top_uncertain_cases_v1.csv"
RISK_UNCERTAIN_OUT = RESULTS / "risk_top_uncertain_cases_v1.csv"

INTENT_LABELS = [
    "normal_interest",
    "sensitive_help_seeking",
    "clearly_harmful_intent",
]

RISK_LABELS = [
    "benign",
    "sensitive_educational",
    "harmful_promotional",
]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)


def add_uncertainty_columns(
    df: pd.DataFrame,
    prob_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    probs = out[prob_cols].astype(float).values

    sorted_probs = np.sort(probs, axis=1)
    out["max_prob"] = sorted_probs[:, -1]
    out["second_prob"] = sorted_probs[:, -2]
    out["margin_top2"] = out["max_prob"] - out["second_prob"]
    out["entropy"] = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return out


def make_prediction_distribution(df: pd.DataFrame, pred_col: str, name: str) -> pd.DataFrame:
    counts = df[pred_col].value_counts(dropna=False).rename_axis("predicted_label").reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()
    counts["model_name"] = name
    return counts[["model_name", "predicted_label", "count", "share"]]


def multiclass_brier_score(
    df: pd.DataFrame,
    true_col: str,
    labels: list[str],
    prob_map: dict[str, str],
) -> float:
    y_true = df[true_col].tolist()
    total = 0.0
    for label in labels:
        y_bin = np.array([1.0 if y == label else 0.0 for y in y_true], dtype=float)
        p = df[prob_map[label]].astype(float).values
        total += brier_score_loss(y_bin, p)
    return total / len(labels)


def expected_calibration_error(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    conf_col: str,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    tmp = df[[true_col, pred_col, conf_col]].copy()
    tmp["correct"] = (tmp[true_col] == tmp[pred_col]).astype(int)
    tmp["bin"] = pd.cut(
        tmp[conf_col],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )

    grouped = (
        tmp.groupby("bin", observed=False)
        .agg(
            bin_count=("correct", "size"),
            avg_confidence=(conf_col, "mean"),
            empirical_accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    grouped["bin_count"] = grouped["bin_count"].fillna(0).astype(int)
    grouped["avg_confidence"] = grouped["avg_confidence"].fillna(0.0)
    grouped["empirical_accuracy"] = grouped["empirical_accuracy"].fillna(0.0)

    n = max(len(tmp), 1)
    grouped["ece_component"] = (
        grouped["bin_count"] / n
    ) * (grouped["avg_confidence"] - grouped["empirical_accuracy"]).abs()

    ece = grouped["ece_component"].sum()
    grouped["bin"] = grouped["bin"].astype(str)
    return float(ece), grouped


def build_detailed_metrics(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    labels: list[str],
    prob_map: dict[str, str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    y_true = df[true_col]
    y_pred = df[pred_col]

    rows = []

    for label in labels:
        rows.append(
            {
                "model_name": model_name,
                "label": label,
                "precision": precision_score(y_true, y_pred, labels=[label], average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0),
                "support": int((y_true == label).sum()),
            }
        )

    rows.extend(
        [
            {
                "model_name": model_name,
                "label": "macro_avg",
                "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "support": len(df),
            },
            {
                "model_name": model_name,
                "label": "weighted_avg",
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "support": len(df),
            },
            {
                "model_name": model_name,
                "label": "overall_accuracy",
                "precision": accuracy_score(y_true, y_pred),
                "recall": accuracy_score(y_true, y_pred),
                "f1_score": accuracy_score(y_true, y_pred),
                "support": len(df),
            },
        ]
    )

    metrics_df = pd.DataFrame(rows)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    brier = multiclass_brier_score(df, true_col, labels, prob_map)
    ece, calib_df = expected_calibration_error(df, true_col, pred_col, "max_prob")

    calib_df["model_name"] = model_name
    calib_df["brier_score_macro_ovr"] = brier
    calib_df["ece"] = ece

    return metrics_df, cm_df, brier, ece


def summarize_model(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    prob_cols: list[str],
    prob_map: dict[str, str],
    labels: list[str],
    model_name: str,
    pred_dist_out: Path,
    uncertain_out: Path,
    detailed_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_uncertainty_columns(df, prob_cols)

    metrics_df, cm_df, brier, ece = build_detailed_metrics(
        df, true_col, pred_col, labels, prob_map, model_name
    )

    metrics_df["avg_max_prob"] = df["max_prob"].mean()
    metrics_df["avg_margin_top2"] = df["margin_top2"].mean()
    metrics_df["avg_entropy"] = df["entropy"].mean()
    metrics_df["brier_score_macro_ovr"] = brier
    metrics_df["ece"] = ece

    metrics_df.to_csv(detailed_out, index=False)

    pred_dist_df = make_prediction_distribution(df, pred_col, model_name)
    pred_dist_df.to_csv(pred_dist_out, index=False)

    base_cols = [c for c in ["session_id", "user_id", "split", "item_id", "source"] if c in df.columns]
    uncertain_cols = base_cols + [true_col, pred_col, "max_prob", "second_prob", "margin_top2", "entropy"]
    uncertain_df = df.sort_values(by=["max_prob", "margin_top2", "entropy"], ascending=[True, True, False]).copy()
    uncertain_df = uncertain_df[uncertain_cols].head(50)
    uncertain_df.to_csv(uncertain_out, index=False)

    calib_rows = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "overall_accuracy": accuracy_score(df[true_col], df[pred_col]),
                "macro_f1": f1_score(df[true_col], df[pred_col], average="macro", zero_division=0),
                "weighted_f1": f1_score(df[true_col], df[pred_col], average="weighted", zero_division=0),
                "avg_max_prob": df["max_prob"].mean(),
                "avg_margin_top2": df["margin_top2"].mean(),
                "avg_entropy": df["entropy"].mean(),
                "brier_score_macro_ovr": brier,
                "ece": ece,
                "n_examples": len(df),
            }
        ]
    )

    return metrics_df, calib_rows


def main() -> None:
    ensure_dirs()

    print("[STEP] Loading probability outputs ...")
    intent_df = pd.read_csv(INTENT_PROBS_PATH)
    risk_df = pd.read_csv(RISK_PROBS_PATH)

    print("[STEP] Summarizing intent model quality ...")
    _, intent_calib = summarize_model(
        df=intent_df,
        true_col="intent_label",
        pred_col="predicted_intent_label",
        prob_cols=[
            "p_normal_interest",
            "p_sensitive_help_seeking",
            "p_clearly_harmful_intent",
        ],
        prob_map={
            "normal_interest": "p_normal_interest",
            "sensitive_help_seeking": "p_sensitive_help_seeking",
            "clearly_harmful_intent": "p_clearly_harmful_intent",
        },
        labels=INTENT_LABELS,
        model_name="intent_model_v1",
        pred_dist_out=INTENT_PRED_DIST_OUT,
        uncertain_out=INTENT_UNCERTAIN_OUT,
        detailed_out=INTENT_DETAILED_OUT,
    )

    print("[STEP] Summarizing risk model quality ...")
    _, risk_calib = summarize_model(
        df=risk_df,
        true_col="risk_label",
        pred_col="predicted_risk_label",
        prob_cols=[
            "q_benign",
            "q_sensitive_educational",
            "q_harmful_promotional",
        ],
        prob_map={
            "benign": "q_benign",
            "sensitive_educational": "q_sensitive_educational",
            "harmful_promotional": "q_harmful_promotional",
        },
        labels=RISK_LABELS,
        model_name="risk_model_v1",
        pred_dist_out=RISK_PRED_DIST_OUT,
        uncertain_out=RISK_UNCERTAIN_OUT,
        detailed_out=RISK_DETAILED_OUT,
    )

    calibration_df = pd.concat([intent_calib, risk_calib], ignore_index=True)
    calibration_df.to_csv(CALIBRATION_OUT, index=False)

    print()
    print("[DONE] Saved:")
    print(f"  - {INTENT_DETAILED_OUT}")
    print(f"  - {RISK_DETAILED_OUT}")
    print(f"  - {CALIBRATION_OUT}")
    print(f"  - {INTENT_PRED_DIST_OUT}")
    print(f"  - {RISK_PRED_DIST_OUT}")
    print(f"  - {INTENT_UNCERTAIN_OUT}")
    print(f"  - {RISK_UNCERTAIN_OUT}")
    print()
    print("[CALIBRATION SUMMARY]")
    print(calibration_df.to_string(index=False))


if __name__ == "__main__":
    main()
