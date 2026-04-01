from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
SWEEP_PATH = ROOT / "results" / "sweeps" / "full_pipeline_v2_light_weight_sweep.csv"
PLOTS = ROOT / "results" / "plots"
TABLES = ROOT / "results" / "tables"

PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

PLOT1 = PLOTS / "weight_sweep_ndcg_vs_osr.png"
PLOT2 = PLOTS / "weight_sweep_ndcg_vs_utility_loss.png"
PLOT3 = PLOTS / "weight_sweep_alpha_beta_gamma_heatmap_proxy.png"
BEST_OUT = TABLES / "weight_sweep_best_configs_v1.csv"


def main():
    print("[STEP] Loading weight sweep table ...")
    df = pd.read_csv(SWEEP_PATH)

    # best by ndcg among same-HER points
    best_ndcg = df.sort_values(
        by=["NDCG@10", "OSR", "UtilityLoss_vs_Baseline"],
        ascending=[False, True, True]
    ).iloc[0]

    # best by lowest OSR
    best_osr = df.sort_values(
        by=["OSR", "NDCG@10", "UtilityLoss_vs_Baseline"],
        ascending=[True, False, True]
    ).iloc[0]

    # best balanced objective
    tmp = df.copy()
    tmp["balanced_objective"] = (
        tmp["HER"] + 0.5 * tmp["OSR"] + 1.0 * tmp["UtilityLoss_vs_Baseline"]
    )
    best_balanced = tmp.sort_values(
        by=["balanced_objective", "NDCG@10"],
        ascending=[True, False]
    ).iloc[0]

    best_df = pd.DataFrame([
        {
            "selection_rule": "best_ndcg",
            **best_ndcg.to_dict(),
        },
        {
            "selection_rule": "best_lowest_osr",
            **best_osr.to_dict(),
        },
        {
            "selection_rule": "best_balanced_objective",
            **best_balanced.to_dict(),
        },
    ])
    best_df.to_csv(BEST_OUT, index=False)

    print("[STEP] Plotting NDCG vs OSR ...")
    plt.figure(figsize=(8, 6))
    plt.scatter(df["OSR"], df["NDCG@10"])
    for _, row in df.iterrows():
        plt.annotate(
            row["config_name"],
            (row["OSR"], row["NDCG@10"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    plt.xlabel("OSR")
    plt.ylabel("NDCG@10")
    plt.title("Weight sweep: NDCG@10 vs OSR")
    plt.tight_layout()
    plt.savefig(PLOT1, dpi=200)
    plt.close()

    print("[STEP] Plotting NDCG vs UtilityLoss ...")
    plt.figure(figsize=(8, 6))
    plt.scatter(df["UtilityLoss_vs_Baseline"], df["NDCG@10"])
    for _, row in df.iterrows():
        plt.annotate(
            row["config_name"],
            (row["UtilityLoss_vs_Baseline"], row["NDCG@10"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    plt.xlabel("UtilityLoss_vs_Baseline")
    plt.ylabel("NDCG@10")
    plt.title("Weight sweep: NDCG@10 vs Utility loss")
    plt.tight_layout()
    plt.savefig(PLOT2, dpi=200)
    plt.close()

    print("[STEP] Plotting beta-gamma heatmap proxy at alpha=0.30 ...")
    subset = df[df["ALPHA"] == 0.30].copy()
    pivot = subset.pivot(index="BETA", columns="GAMMA", values="NDCG@10")

    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("GAMMA")
    plt.ylabel("BETA")
    plt.title("NDCG@10 heatmap proxy (ALPHA=0.30)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT3, dpi=200)
    plt.close()

    print()
    print("[DONE] Saved:")
    print(f"  - {PLOT1}")
    print(f"  - {PLOT2}")
    print(f"  - {PLOT3}")
    print(f"  - {BEST_OUT}")
    print()
    print("[BEST CONFIGS]")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
