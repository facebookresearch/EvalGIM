# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


def create_polar_plot(df, metrics, regions, color_idx, custom_ranges):
    num_cols = len(metrics)
    fig = make_subplots(
        rows=1,
        cols=num_cols,
        specs=[[{"type": "polar"}] * num_cols],
        subplot_titles=metrics,
        horizontal_spacing=0.1,
    )

    for idx, metric in enumerate(metrics):
        metric_columns = [f"{metric}_{region}" for region in regions]

        for model, color_index in color_idx.items():
            df_temp = df[df["model_id"] == model]
            color = "#{:02x}{:02x}{:02x}".format(
                *[int(i * 256) for i in sns.color_palette(palette="Greens")[color_index]]
            )

            fig.add_trace(
                go.Scatterpolar(
                    r=df_temp.iloc[0][metric_columns].values,
                    theta=[col.split(f"{metric}_")[-1] for col in df_temp.iloc[0][metric_columns].index],
                    fill="none",
                    name=df_temp.iloc[0].model_id,
                    line_color=color,
                ),
                row=1,
                col=idx + 1,
            )

            update_polar_axes(fig, metric, row=1, col=idx + 1, ranges=custom_ranges)

    model_count = len(list(set(df["model_id"])))
    hide_extra_legends(fig, model_count)
    return fig


def update_polar_axes(fig, metric, row, col, ranges=None):
    default_ranges = {
        "clipscore": [15, 30],
        "coverage": [0, 1],
        "precision": [0, 1],
        "recall": [0, 1],
        "density": [0, 1],
        "vqa_score": [0, 1],
        "fid": [0, 20],
    }

    if ranges and metric in ranges:
        range_to_use = ranges[metric]
    elif "clipscore" in metric:
        range_to_use = default_ranges["clipscore"]
    elif any(word in metric for word in ("coverage", "precision", "recall", "density", "vqa_score")):
        range_to_use = default_ranges["coverage"]
    elif "fid" in metric:
        range_to_use = default_ranges["fid"]
    else:
        raise Exception(f"{metric} not supported!")

    fig.update_polars(radialaxis=dict(range=range_to_use), row=row, col=col)


def hide_extra_legends(fig, model_count):
    for idx, trace in enumerate(fig["data"]):
        if idx >= model_count:
            trace["showlegend"] = False


def main():
    parser = argparse.ArgumentParser(description="Process and plot group metrics.")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file containing the results.")
    parser.add_argument(
        "--ranges", type=str, nargs="+", help="Custom ranges for metrics (e.g., clipscore:15-30 fid:0-20)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Custom metrics for group polar plots. (e.g., recall precision clipscore)",
    )
    args = parser.parse_args()
    custom_ranges = {}
    if args.ranges:
        for r in args.ranges:
            metric, rng = r.split(":")
            min_val, max_val = map(float, rng.split("-"))
            custom_ranges[metric] = [min_val, max_val]

    regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    color_idx = {model: idx + 1 for idx, model in enumerate(sorted(list(set(df["model_id"]))))}

    fig = create_polar_plot(df, args.metrics, regions, color_idx, custom_ranges)
    fig.update_layout(width=1200, showlegend=True)
    if not os.path.exists("./visuals"):
        os.makedirs("./visuals")
    fig.write_image("./visuals/groups.png")
    print(f"plot saved as {os.path.abspath('.')}/visuals/groups.png")


if __name__ == "__main__":
    main()
