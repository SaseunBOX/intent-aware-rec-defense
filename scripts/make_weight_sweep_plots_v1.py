from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path("/mnt/e/intent_aware_rec_defense")
SWEEP_PATH = ROOT / "results" / "sweeps" / "full_pipeline_v2_light_weight_sweep.csv"
BEST_PATH = ROOT / "results" / "tables" / "weight_sweep_best_configs_v1.csv"
PLOTS = ROOT / "results" / "plots"
TABLES = ROOT / "results" / "tables"

PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

PLOT1 = PLOTS / "weight_sweep_ndcg_vs_osr.png"
PLOT2 = PLOTS / "weight_sweep_ndcg_vs_utility_loss.png"
PLOT3 = PLOTS / "weight_sweep_alpha_beta_gamma_heatmap_proxy.png"

FIGSIZE = (15.2, 8.6)
BLUE = "tab:blue"
ORANGE = "tab:orange"
GREEN = "tab:green"
PURPLE = "tab:purple"


def ensure_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    best_ndcg = tmp.sort_values(
        by=["NDCG@10", "OSR", "UtilityLoss_vs_Baseline"],
        ascending=[False, True, True],
    ).iloc[0]

    best_osr = tmp.sort_values(
        by=["OSR", "NDCG@10", "UtilityLoss_vs_Baseline"],
        ascending=[True, False, True],
    ).iloc[0]

    tmp["balanced_objective"] = (
        tmp["HER"] + 0.5 * tmp["OSR"] + 1.0 * tmp["UtilityLoss_vs_Baseline"]
    )
    best_balanced = tmp.sort_values(
        by=["balanced_objective", "NDCG@10"],
        ascending=[True, False],
    ).iloc[0]

    best_df = pd.DataFrame([
        {"selection_rule": "best_ndcg", **best_ndcg.to_dict()},
        {"selection_rule": "best_lowest_osr", **best_osr.to_dict()},
        {"selection_rule": "best_balanced_objective", **best_balanced.to_dict()},
    ])
    best_df.to_csv(BEST_PATH, index=False)
    return best_df


def apply_cluster_jitter(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = df.copy()

    x_range = float(out[x_col].max() - out[x_col].min())
    y_range = float(out[y_col].max() - out[y_col].min())

    x_step = max(x_range * 0.070, 0.0028)
    y_step = max(y_range * 0.070, 0.00070)

    out["_cluster_x"] = out[x_col].round(6)
    out["_cluster_y"] = out[y_col].round(6)

    disp_x = {}
    disp_y = {}

    for _, group in out.groupby(["_cluster_x", "_cluster_y"], sort=False):
        n = len(group)
        if n == 1:
            idx = group.index[0]
            disp_x[idx] = float(group.iloc[0][x_col])
            disp_y[idx] = float(group.iloc[0][y_col])
            continue

        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        x_offsets = [((c - (cols - 1) / 2.0) * x_step) for c in range(cols)]
        y_offsets = [(((rows - 1) / 2.0 - r) * y_step) for r in range(rows)]

        for k, (idx, row) in enumerate(group.iterrows()):
            r = k // cols
            c = k % cols
            disp_x[idx] = float(row[x_col]) + x_offsets[c]
            disp_y[idx] = float(row[y_col]) + y_offsets[r]

    out["_x_disp"] = pd.Series(disp_x)
    out["_y_disp"] = pd.Series(disp_y)

    return out.drop(columns=["_cluster_x", "_cluster_y"])


def build_mapping_text(names: list[str]) -> str:
    lines = [f"{i+1}: {name}" for i, name in enumerate(names)]
    return "\n".join(lines)


def nearest_display_point(
    df: pd.DataFrame,
    x_value: float,
    y_value: float,
    x_col: str,
    y_col: str,
) -> tuple[float, float]:
    tmp = df.copy()
    tmp["_dist"] = (tmp[x_col] - x_value) ** 2 + (tmp[y_col] - y_value) ** 2
    row = tmp.sort_values("_dist").iloc[0]
    return float(row["_x_disp"]), float(row["_y_disp"])


def add_numbered_points(ax, df: pd.DataFrame) -> None:
    ax.scatter(
        df["_x_disp"],
        df["_y_disp"],
        s=180,
        marker="o",
        alpha=0.95,
        c=BLUE,
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
    )
    for _, row in df.iterrows():
        ax.text(
            row["_x_disp"],
            row["_y_disp"],
            str(int(row["point_id"])),
            fontsize=9,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
        )


def add_selected_markers(ax, df_disp: pd.DataFrame, best_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    marker_specs = [
        ("best_ndcg", "*", 420, ORANGE, (-0.0015, 0.0000)),
        ("best_lowest_osr", "X", 340, GREEN, (0.0015, 0.0000)),
        ("best_balanced_objective", "D", 270, PURPLE, (0.0000, 0.0008)),
    ]

    for rule, marker, size, color, (dx, dy) in marker_specs:
        rows = best_df[best_df["selection_rule"] == rule]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        x_val = float(row[x_col])
        y_val = float(row[y_col])
        x_disp, y_disp = nearest_display_point(df_disp, x_val, y_val, x_col, y_col)
        ax.scatter(
            [x_disp + dx],
            [y_disp + dy],
            s=size,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )


def add_mapping_box(fig, names: list[str]) -> None:
    block = build_mapping_text(names)
    fig.text(
        0.78,
        0.95,
        block,
        ha="left",
        va="top",
        fontsize=8.2,
        family="monospace",
        linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92),
    )


def add_scatter_legend(ax) -> None:
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            label="All weight-sweep configs",
        ),
        Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            markersize=15,
            markerfacecolor=ORANGE,
            markeredgecolor=ORANGE,
            label="Best NDCG",
        ),
        Line2D(
            [0], [0],
            marker="X",
            linestyle="None",
            markersize=12,
            markerfacecolor=GREEN,
            markeredgecolor=GREEN,
            label="Lowest OSR",
        ),
        Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            markersize=11,
            markerfacecolor=PURPLE,
            markeredgecolor=PURPLE,
            label="Balanced objective",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=18, pad=14)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=11)


def make_scatter_plot(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = df.copy()
    plot_df["point_id"] = range(1, len(plot_df) + 1)
    plot_df["point_name"] = plot_df["config_name"].astype(str)
    plot_df = apply_cluster_jitter(plot_df, x_col=x_col, y_col=y_col)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(right=0.67)

    add_numbered_points(ax, plot_df)
    add_selected_markers(ax, plot_df, best_df, x_col=x_col, y_col=y_col)
    add_scatter_legend(ax)
    add_mapping_box(fig, plot_df["point_name"].tolist())
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    plt.tight_layout(rect=[0, 0, 0.67, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_heatmap(df: pd.DataFrame, best_df: pd.DataFrame, output_path: Path) -> None:
    subset = df[df["ALPHA"] == 0.30].copy()
    pivot = subset.pivot(index="BETA", columns="GAMMA", values="NDCG@10")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(right=0.74)

    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index], fontsize=12)

    style_axes(
        ax,
        xlabel="GAMMA",
        ylabel="BETA",
        title="Weight sweep: alpha-beta-gamma heatmap proxy (ALPHA=0.30)",
    )
    ax.grid(False)

    vals = pivot.values
    vmin = vals.min()
    vmax = vals.max()
    threshold = (vmin + vmax) / 2.0

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = vals[i, j]
            text_color = "black" if value >= threshold else "white"
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
                fontweight="bold",
            )

    highlighted_rules = {
        "best_ndcg": "Best NDCG",
        "best_lowest_osr": "Lowest OSR",
        "best_balanced_objective": "Balanced objective",
    }

    right_summary_lines = []

    for _, row in best_df.iterrows():
        rule = str(row["selection_rule"])
        name = highlighted_rules.get(rule, rule)
        alpha = float(row["ALPHA"])
        beta = float(row["BETA"])
        gamma = float(row["GAMMA"])
        ndcg = float(row["NDCG@10"])
        osr = float(row["OSR"])
        loss = float(row["UtilityLoss_vs_Baseline"])

        right_summary_lines.append(
            f"{name}\n"
            f"cfg={row['config_name']}\n"
            f"(a={alpha:.2f}, b={beta:.2f}, g={gamma:.2f})\n"
            f"NDCG={ndcg:.3f}, OSR={osr:.3f}, Loss={loss:.3f}"
        )

        if abs(alpha - 0.30) < 1e-9 and beta in pivot.index and gamma in pivot.columns:
            i = list(pivot.index).index(beta)
            j = list(pivot.columns).index(gamma)
            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                linewidth=2.5,
                edgecolor="white",
            )
            ax.add_patch(rect)

    fig.text(
        0.77,
        0.92,
        "\n\n".join(right_summary_lines),
        ha="left",
        va="top",
        fontsize=9.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92),
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("NDCG@10", rotation=90, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.74, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SWEEP_PATH.exists():
        raise FileNotFoundError(f"Weight sweep CSV not found: {SWEEP_PATH}")

    df = pd.read_csv(SWEEP_PATH)
    best_df = ensure_best_configs(df)

    make_scatter_plot(
        df=df,
        best_df=best_df,
        x_col="OSR",
        y_col="NDCG@10",
        xlabel="OSR",
        ylabel="NDCG@10",
        title="Weight sweep: NDCG@10 vs OSR",
        output_path=PLOT1,
    )

    make_scatter_plot(
        df=df,
        best_df=best_df,
        x_col="UtilityLoss_vs_Baseline",
        y_col="NDCG@10",
        xlabel="UtilityLoss_vs_Baseline",
        ylabel="NDCG@10",
        title="Weight sweep: NDCG@10 vs UtilityLoss",
        output_path=PLOT2,
    )

    make_heatmap(
        df=df,
        best_df=best_df,
        output_path=PLOT3,
    )

    print("[DONE] Saved final weight-sweep plots:")
    print(f"  - {PLOT1}")
    print(f"  - {PLOT2}")
    print(f"  - {PLOT3}")
    print(f"  - {BEST_PATH}")


if __name__ == "__main__":
    main()