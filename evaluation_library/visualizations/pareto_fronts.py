# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from paretoset import paretoset

_METRICS = {
    "precision": {"name": r"Quality$_\text{Precision}$", "direction": 1},
    "recall": {"name": r"Recall$_\text{Marginal}$", "direction": 1},
    "density": {"name": r"Density$_\text{Marginal}$", "direction": 1},
    "coverage": {"name": r"Diversity$_\text{Coverage}$", "direction": 1},
    "clipscore": {"name": r"Consistency$_{\text{CLIPScore}}$", "direction": 1},
    "clipscore_torchmetrics": {"name": r"CLIPScore-torchmetrics$_{\text{Conditional}}$", "direction": 1},
    "vqascore": {"name": r"Consistency$_{\text{VQAScore}}$", "direction": 1},
    "fid_torchmetrics": {"name": r"FID-torchmetrics$_{\text{Marginal}}$", "direction": -1},
    "fid_torcheval": {"name": r"FID-torcheval$_{\text{Marginal}}$", "direction": -1},
}


def get_xy_metrics(metrics_for_three_axis):
    assert len(metrics_for_three_axis) == 3
    xy_metrics = []

    for indx, metric in enumerate(metrics_for_three_axis):
        xy_metrics += [(metric, metrics_for_three_axis[indx - 1])]

    return xy_metrics


def custom_formatter(x, pos):
    return f".{int(x*100)}"


def plot_pareto(xy_metrics, all_dfs, sweep_over, title=None, plot_palette=None):
    """Plots pareto fronts.
    xy_metrics: list of metric string pairings for xy-axes; Name
        must match those in _METRICS
    all_dfs: pandas dataframe containing a row for each datapoint,
        a column for each metric, and "model" column used for c
        color-coding and legend creation
    title: (Optional) string used for title and saving plot
    """
    nrows, ncols = len(xy_metrics) // 3, 3

    palette = sns.color_palette("Set2", 7)

    okabe_ito_palette = [
        (230 / 255, 159 / 255, 0),  # Orange
        (86 / 255, 180 / 255, 233 / 255),  # Sky Blue
        (240 / 255, 228 / 255, 66 / 255),  # Yellow
        (0, 158 / 255, 115 / 255),  # Bluish Green
        (0, 114 / 255, 178 / 255),  # Blue
        (213 / 255, 94 / 255, 0),  # Vermillion
        (204 / 255, 121 / 255, 167 / 255),  # Reddish Purple
        (148 / 255, 0 / 255, 211 / 255),  # Deep Violet
        (255 / 255, 105 / 255, 180 / 255),  # Hot Pink
        (0 / 255, 128 / 255, 128 / 255),  # Teal
    ]
    sns.set_theme(context="paper", palette=palette, font_scale=1.75, style="whitegrid")
    sweep_colors = {
        sweep_key: c
        for sweep_key, c in zip(
            sorted(list(set(all_dfs[sweep_over]))),
            okabe_ito_palette if plot_palette is None else plot_palette,
            strict=False,
        )
    }

    for r in range(nrows):
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5.3, 1 * 4.3))
        for i in range(ncols):
            (x_metric, y_metric) = xy_metrics[r * ncols + i]

            ax = axs.flatten()[i]

            x_direction = "max" if _METRICS[x_metric]["direction"] == 1 else "min"
            y_direction = "max" if _METRICS[y_metric]["direction"] == 1 else "min"

            if _METRICS[x_metric]["direction"] == -1:
                ax.invert_xaxis()

            mask = all_dfs[x_metric].notnull() & all_dfs[y_metric].notnull()
            plot_df = all_dfs[mask]
            frontier_mask_all = paretoset(
                all_dfs[[x_metric, y_metric]].to_numpy(), sense=[x_direction, y_direction]
            )
            sns.lineplot(
                data=plot_df[frontier_mask_all],
                x=x_metric,
                y=y_metric,
                ax=ax,
                sort=True,
                zorder=0,
                lw=1,
                legend=i == ncols - 1,
                c=(0.25, 0.25, 0.25),
                linestyle="--",
                label="pareto",
            )

            for sweep_key in sorted(plot_df[sweep_over].unique()):
                sweep_df = plot_df[plot_df[sweep_over] == sweep_key]
                sns.scatterplot(
                    data=sweep_df,
                    x=x_metric,
                    y=y_metric,
                    ax=ax,
                    s=75,
                    zorder=2,
                    legend=i == ncols - 1,
                    marker="o",
                    edgecolors=(0.25, 0.25, 0.25),
                    facecolors=[sweep_colors[sweep_key]],
                    label=sweep_key,
                )

            xmin, xmax = (0, 1)
            ymin, ymax = (0, 1)
            ax.text(
                xmin - 0.17,
                ymax,
                "high",
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax.transAxes,
                style="italic",
                rotation=90,
                fontsize=14,
            )
            ax.text(
                xmax,
                ymin - 0.15,
                "high",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
                style="italic",
                fontsize=14,
            )
            ax.text(
                xmin - 0.17,
                ymin,
                "low",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=ax.transAxes,
                style="italic",
                rotation=90,
                fontsize=14,
            )
            ax.text(
                xmin,
                ymin - 0.15,
                "low",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=ax.transAxes,
                style="italic",
                fontsize=14,
            )
            if i == ncols - 1:
                ax.legend().set_loc("lower right")
                ax.legend().set_bbox_to_anchor((1.6, 0.5))
                ax.legend(title=sweep_over)

            ax.set_xlabel(_METRICS[x_metric]["name"])
            ax.set_ylabel(_METRICS[y_metric]["name"])
            ax.tick_params(axis="both", which="major", labelsize=14, direction="in")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        fig.subplots_adjust(hspace=0.35, wspace=0.3)

        if not os.path.exists("./visuals"):
            os.makedirs("./visuals")
        if title:
            fig.suptitle(title)
            plt.savefig(
                f"./visuals/{title}_{r}.png",
                bbox_inches="tight",
            )

            print(f"Pareto plot saved at: {os.path.abspath('.')}/visuals/{title}_{r}.png")
        else:
            plt.savefig(
                f"./visuals/{r}.png",
                bbox_inches="tight",
            )
            print(f"Pareto plot saved at: {os.path.abspath('.')}visuals/{r}.png")


def main():
    parser = argparse.ArgumentParser(description="Pareto fronts input params")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file containing the results.")
    parser.add_argument("--metrics_for_three_axis", nargs="+", help="List of metrics for three axis")
    parser.add_argument("--sweep_over", type=str, default="model_id", help="Column name for parameter sweep")

    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    for dataset in set(df["dataset"]):
        xy_metrics = get_xy_metrics(args.metrics_for_three_axis)
        plot_pareto(
            xy_metrics,
            df[df["dataset"] == dataset],
            args.sweep_over,
            title=f"pareto_fronts_{dataset}",
            plot_palette=sns.color_palette("flare", as_cmap=False),
        )


if __name__ == "__main__":
    main()
