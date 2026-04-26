from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path("/mnt/e/intent_aware_rec_defense")
SWEEP_PATH = ROOT / "results" / "sweeps" / "full_pipeline_v2_threshold_sweep.csv"
OPERATING_PATH = ROOT / "results" / "tables" / "operating_points_v1.csv"
PLOT_DIR = ROOT / "results" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE = (15.2, 8.6)
BLUE = "tab:blue"
ORANGE = "tab:orange"
GREEN = "tab:green"


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns were found: {candidates}")
    return None


def load_baseline_ndcg() -> float:
    candidates = [
        ROOT / "results" / "tables" / "thesis_main_results_v3.csv",
        ROOT / "results" / "tables" / "thesis_main_results_v2.csv",
        ROOT / "results" / "main_metrics_table_v2.csv",
        ROOT / "results" / "main_metrics_table_v1.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "System" in df.columns and "NDCG@10" in df.columns:
            row = df[df["System"].astype(str).str.contains("Injected_Baseline", case=False, na=False)]
            if len(row) > 0:
                return float(row.iloc[0]["NDCG@10"])
    return 0.5460639508934478


def add_loss_column(df: pd.DataFrame, baseline_ndcg: float) -> pd.DataFrame:
    out = df.copy()
    if "NDCG_loss" in out.columns:
        return out
    if "UtilityLoss" in out.columns:
        out["NDCG_loss"] = out["UtilityLoss"]
        return out
    if "UtilityLoss_vs_Baseline" in out.columns:
        out["NDCG_loss"] = out["UtilityLoss_vs_Baseline"]
        return out
    if "NDCG@10" in out.columns:
        out["NDCG_loss"] = (baseline_ndcg - out["NDCG@10"].astype(float)) / baseline_ndcg
        return out
    raise KeyError("Could not construct NDCG_loss column.")


def build_point_names(df: pd.DataFrame) -> list[str]:
    name_col = pick_col(
        df,
        ["config_name", "Config", "System", "config", "name", "label", "mode"],
        required=False,
    )
    if name_col is None:
        return [f"cfg_{i+1}" for i in range(len(df))]
    return df[name_col].astype(str).tolist()


def apply_cluster_jitter(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Stronger display-only jitter.
    Also scales up horizontal spread when x values are highly compressed near zero.
    """
    out = df.copy()

    x_range = float(out[x_col].max() - out[x_col].min())
    y_range = float(out[y_col].max() - out[y_col].min())

    x_step = max(x_range * 0.10, 2.0e-5)
    y_step = max(y_range * 0.16, 1.5e-4)

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
    sweep_df: pd.DataFrame,
    x_value: float,
    y_value: float,
    x_col: str,
    y_col: str,
) -> tuple[float, float]:
    tmp = sweep_df.copy()
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


def add_selected_markers(
    ax,
    sweep_df: pd.DataFrame,
    op_df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    if len(op_df) == 0:
        return

    mode_col = pick_col(op_df, ["Mode", "mode", "Selection_rule", "selection_rule"], required=False)

    for _, row in op_df.iterrows():
        mode_name = str(row[mode_col]) if mode_col is not None else "selected"
        mode_low = mode_name.lower()

        marker = "o"
        size = 220
        color = "tab:red"
        dx = 0.0
        dy = 0.0

        if "balanced" in mode_low:
            marker = "*"
            size = 420
            color = ORANGE
            dx = -2.5e-6
            dy = 8e-5
        elif "high" in mode_low:
            marker = "X"
            size = 340
            color = GREEN
            dx = 2.5e-6
            dy = -8e-5

        x_val = float(row[x_col])
        y_val = float(row[y_col])
        x_disp, y_disp = nearest_display_point(sweep_df, x_val, y_val, x_col, y_col)

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


def add_legend(ax) -> None:
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            label="All threshold-sweep configs",
        ),
        Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            markersize=15,
            markerfacecolor=ORANGE,
            markeredgecolor=ORANGE,
            label="Balanced mode",
        ),
        Line2D(
            [0], [0],
            marker="X",
            linestyle="None",
            markersize=12,
            markerfacecolor=GREEN,
            markeredgecolor=GREEN,
            label="High-safety mode",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=18, pad=14)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=11)


def plot_tradeoff(
    sweep_df: pd.DataFrame,
    op_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = sweep_df.copy()
    plot_df["point_name"] = build_point_names(plot_df)
    plot_df["point_id"] = range(1, len(plot_df) + 1)
    plot_df = apply_cluster_jitter(plot_df, x_col=x_col, y_col=y_col)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(right=0.67)

    add_numbered_points(ax, plot_df)
    add_selected_markers(ax, plot_df, op_df, x_col=x_col, y_col=y_col)
    add_legend(ax)
    add_mapping_box(fig, plot_df["point_name"].tolist())
    style_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title)

    plt.tight_layout(rect=[0, 0, 0.67, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SWEEP_PATH.exists():
        raise FileNotFoundError(f"Threshold sweep CSV not found: {SWEEP_PATH}")
    if not OPERATING_PATH.exists():
        raise FileNotFoundError(f"Operating-points CSV not found: {OPERATING_PATH}")

    baseline_ndcg = load_baseline_ndcg()

    sweep_df = pd.read_csv(SWEEP_PATH)
    op_df = pd.read_csv(OPERATING_PATH)

    sweep_df = add_loss_column(sweep_df, baseline_ndcg=baseline_ndcg)
    op_df = add_loss_column(op_df, baseline_ndcg=baseline_ndcg)

    her_col_sweep = pick_col(sweep_df, ["HER"])
    osr_col_sweep = pick_col(sweep_df, ["OSR"])
    loss_col_sweep = pick_col(sweep_df, ["NDCG_loss"])

    her_col_op = pick_col(op_df, ["HER"])
    osr_col_op = pick_col(op_df, ["OSR"])
    loss_col_op = pick_col(op_df, ["NDCG_loss"])

    sweep_plot = sweep_df.copy()
    sweep_plot["HER"] = sweep_plot[her_col_sweep]
    sweep_plot["OSR"] = sweep_plot[osr_col_sweep]
    sweep_plot["NDCG_loss"] = sweep_plot[loss_col_sweep]

    op_plot = op_df.copy()
    op_plot["HER"] = op_plot[her_col_op]
    op_plot["OSR"] = op_plot[osr_col_op]
    op_plot["NDCG_loss"] = op_plot[loss_col_op]

    plot_tradeoff(
        sweep_df=sweep_plot,
        op_df=op_plot,
        x_col="HER",
        y_col="NDCG_loss",
        xlabel="HER",
        ylabel="NDCG loss",
        title="HER vs NDCG loss",
        output_path=PLOT_DIR / "her_vs_ndcg_loss.png",
    )

    plot_tradeoff(
        sweep_df=sweep_plot,
        op_df=op_plot,
        x_col="HER",
        y_col="OSR",
        xlabel="HER",
        ylabel="OSR",
        title="HER vs OSR",
        output_path=PLOT_DIR / "her_vs_osr.png",
    )

    plot_tradeoff(
        sweep_df=sweep_plot,
        op_df=op_plot,
        x_col="OSR",
        y_col="NDCG_loss",
        xlabel="OSR",
        ylabel="NDCG loss",
        title="OSR vs NDCG loss",
        output_path=PLOT_DIR / "osr_vs_ndcg_loss.png",
    )

    print("[DONE] Saved final trade-off plots:")
    print(f"  - {PLOT_DIR / 'her_vs_ndcg_loss.png'}")
    print(f"  - {PLOT_DIR / 'her_vs_osr.png'}")
    print(f"  - {PLOT_DIR / 'osr_vs_ndcg_loss.png'}")


if __name__ == "__main__":
    main()