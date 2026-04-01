from __future__ import annotations
#把 sweep 结果做成论文图，便于解释 Balanced mode 和 High-safety mode。
#Convert the sweep results into a figure for the paper, which will facilitate the explanation of the Balanced mode and the High-safety mode.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
RESULTS = ROOT / "results"

SWEEP_PATH = RESULTS / "sweeps" / "full_pipeline_v2_threshold_sweep.csv"
ABLATION_PATH = RESULTS / "main_metrics_table_v2.csv"

PLOTS_DIR = RESULTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HER_VS_NDCG_PATH = PLOTS_DIR / "her_vs_ndcg_loss.png"
HER_VS_OSR_PATH = PLOTS_DIR / "her_vs_osr.png"
OSR_VS_NDCG_PATH = PLOTS_DIR / "osr_vs_ndcg_loss.png"
ABLATION_BAR_PATH = PLOTS_DIR / "ablation_her_osr.png"


def choose_balanced_and_high_safety(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Balanced mode:
      minimize HER + lambda * UtilityLoss, with a light OSR preference.
    High-safety mode:
      prioritize the smallest HER, then smaller UtilityLoss, then smaller OSR.
    """
    lam = 1.0

    tmp = df.copy()
    tmp["balanced_objective"] = tmp["HER"] + lam * tmp["UtilityLoss_vs_Baseline"] + 0.1 * tmp["OSR"]

    balanced = tmp.sort_values(
        by=["balanced_objective", "HER", "UtilityLoss_vs_Baseline", "OSR"],
        ascending=[True, True, True, True],
    ).iloc[0]

    high_safety = tmp.sort_values(
        by=["HER", "UtilityLoss_vs_Baseline", "OSR"],
        ascending=[True, True, True],
    ).iloc[0]

    return balanced, high_safety


def plot_her_vs_ndcg_loss(df: pd.DataFrame, balanced: pd.Series, high_safety: pd.Series) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["HER"], df["UtilityLoss_vs_Baseline"], s=70)

    for _, row in df.iterrows():
        plt.annotate(
            row["config_name"],
            (row["HER"], row["UtilityLoss_vs_Baseline"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.scatter(
        [balanced["HER"]],
        [balanced["UtilityLoss_vs_Baseline"]],
        s=140,
        marker="*",
        label="Balanced mode",
    )
    plt.scatter(
        [high_safety["HER"]],
        [high_safety["UtilityLoss_vs_Baseline"]],
        s=140,
        marker="X",
        label="High-safety mode",
    )

    plt.xlabel("HER")
    plt.ylabel("NDCG loss")
    plt.title("HER vs NDCG loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HER_VS_NDCG_PATH, dpi=200)
    plt.close()


def plot_her_vs_osr(df: pd.DataFrame, balanced: pd.Series, high_safety: pd.Series) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["HER"], df["OSR"], s=70)

    for _, row in df.iterrows():
        plt.annotate(
            row["config_name"],
            (row["HER"], row["OSR"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.scatter(
        [balanced["HER"]],
        [balanced["OSR"]],
        s=140,
        marker="*",
        label="Balanced mode",
    )
    plt.scatter(
        [high_safety["HER"]],
        [high_safety["OSR"]],
        s=140,
        marker="X",
        label="High-safety mode",
    )

    plt.xlabel("HER")
    plt.ylabel("OSR")
    plt.title("HER vs OSR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HER_VS_OSR_PATH, dpi=200)
    plt.close()


def plot_osr_vs_ndcg_loss(df: pd.DataFrame, balanced: pd.Series, high_safety: pd.Series) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["OSR"], df["UtilityLoss_vs_Baseline"], s=70)

    for _, row in df.iterrows():
        plt.annotate(
            row["config_name"],
            (row["OSR"], row["UtilityLoss_vs_Baseline"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.scatter(
        [balanced["OSR"]],
        [balanced["UtilityLoss_vs_Baseline"]],
        s=140,
        marker="*",
        label="Balanced mode",
    )
    plt.scatter(
        [high_safety["OSR"]],
        [high_safety["UtilityLoss_vs_Baseline"]],
        s=140,
        marker="X",
        label="High-safety mode",
    )

    plt.xlabel("OSR")
    plt.ylabel("NDCG loss")
    plt.title("OSR vs NDCG loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OSR_VS_NDCG_PATH, dpi=200)
    plt.close()


def plot_ablation_her_osr(ablation_df: pd.DataFrame) -> None:
    plot_df = ablation_df[["System", "HER", "OSR"]].copy()

    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], plot_df["HER"], width=width, label="HER")
    plt.bar([i + width / 2 for i in x], plot_df["OSR"], width=width, label="OSR")

    plt.xticks(list(x), plot_df["System"], rotation=20)
    plt.ylabel("Rate")
    plt.title("Ablation HER / OSR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ABLATION_BAR_PATH, dpi=200)
    plt.close()


def main() -> None:
    print("[STEP] Loading sweep results ...")
    sweep_df = pd.read_csv(SWEEP_PATH)

    print("[STEP] Loading ablation/main table ...")
    ablation_df = pd.read_csv(ABLATION_PATH)

    balanced, high_safety = choose_balanced_and_high_safety(sweep_df)

    print("[STEP] Creating trade-off plots ...")
    plot_her_vs_ndcg_loss(sweep_df, balanced, high_safety)
    plot_her_vs_osr(sweep_df, balanced, high_safety)
    plot_osr_vs_ndcg_loss(sweep_df, balanced, high_safety)

    print("[STEP] Creating ablation HER/OSR bar chart ...")
    plot_ablation_her_osr(ablation_df)

    print()
    print("[DONE] Plot files saved:")
    print(f"  - {HER_VS_NDCG_PATH}")
    print(f"  - {HER_VS_OSR_PATH}")
    print(f"  - {OSR_VS_NDCG_PATH}")
    print(f"  - {ABLATION_BAR_PATH}")
    print()
    print("[OPERATING POINTS]")
    print("Balanced mode:")
    print(balanced.to_string())
    print()
    print("High-safety mode:")
    print(high_safety.to_string())


if __name__ == "__main__":
    main()