# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

clean_dataset_name = {
    "cc12m_validation_dataset": "CC12M",
    "cc12m_recaptioned_dataset": "CC12M Recap.",
    "coco_txt_dataset": "COCO",
    "geode_dataset": "GeoDE",
    "imagenet_validation_dataset": "ImageNet",
}


def load_data(file_path):
    return pd.read_csv(file_path, index_col=0)


def get_min_score(group):
    return group.loc[group["fid_torchmetrics"].idxmin()]


def process_data(df, vqascore):
    columns = ["model_id", "dataset", "clipscore", "fid_torchmetrics", "coverage", "precision"]
    if vqascore:
        columns.append("vqascore")
    hold = (
        df.groupby(by=["model_id", "dataset"])[columns]
        .apply(get_min_score)
        .drop(columns=["model_id", "dataset"])
        .reset_index()
    )
    hold["dataset"] = [clean_dataset_name[i] for i in hold["dataset"]]
    hold["clipscore"] = hold["clipscore"].round(1)
    hold["fid_torchmetrics"] = hold["fid_torchmetrics"].round()
    hold["coverage"] = hold["coverage"].round(2)
    hold["precision"] = hold["precision"].round(2)
    if vqascore:
        hold["vqascore"] = hold["vqascore"].round(2)

    return hold


def create_pivot_table(hold, vqascore):
    values = ["precision", "coverage", "fid_torchmetrics", "clipscore"]
    if vqascore:
        values.append("vqascore")
    pivot_table = pd.pivot_table(
        hold,
        index=["model_id"],
        columns="dataset",
        values=values,
        sort=False,
    )
    pivot_table.index = [i.replace("\t", "  ") for i in pivot_table.index]
    return pivot_table


def normalize_column(col):
    return (
        (col - np.min(col)) / (np.max(col) - np.min(col))
        if np.max(col) != np.min(col)
        else np.zeros_like(col)
    )


def plot(df, filename=None):
    fig, ax = plt.subplots(figsize=(15, len(df) * 0.6))
    ax.axis("off")

    row_labels = df.index.tolist()
    col_labels = df.columns

    data_values = df.values

    normalized_data = {}
    for metric, dataset in df.columns:
        col_key = (metric, dataset)
        normalized_data[col_key] = normalize_column(df[col_key].to_numpy())

    cmap = sns.light_palette("green", as_cmap=True)
    cmap_reverse = sns.light_palette("green", as_cmap=True, reverse=True)

    for (i, j), val in np.ndenumerate(data_values):
        metric, dataset = col_labels[j]
        col_key = (metric, dataset)
        norm_val = normalized_data[col_key][i]
        rgba_color = cmap_reverse(norm_val) if metric == "fid_torchmetrics" else cmap(norm_val)
        rect = plt.Rectangle([j, i], 1, 1, facecolor=rgba_color, edgecolor="white", lw=0.5)
        ax.add_patch(rect)
        ax.text(j + 0.5, i + 0.5, f"{val}", ha="center", va="center", fontsize=9, color="black")

    for i, label in enumerate(row_labels):
        ax.text(-0.5, i + 0.5, label, ha="right", va="center", fontsize=10, color="black")

    unique_metrics = col_labels.get_level_values(0).unique()
    metric_offsets = {metric: (col_labels.get_level_values(0) == metric).sum() for metric in unique_metrics}

    current_offset = 0
    for metric, count in metric_offsets.items():
        ax.text(
            current_offset + count / 2 - 0.5,
            len(row_labels) + 1.2,
            metric,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )
        current_offset += count

    for j, dataset in enumerate(col_labels.get_level_values(1)):
        ax.text(j + 0.5, len(row_labels) + 0.5, dataset, ha="center", va="center", fontsize=9)

    ax.set_xlim(-1, data_values.shape[1])
    ax.set_ylim(-1, data_values.shape[0] + 2)

    ax.hlines(
        np.arange(data_values.shape[0]), xmin=-1, xmax=data_values.shape[1], color="white", linewidth=0.5
    )
    ax.vlines(
        np.arange(data_values.shape[1]), ymin=-1, ymax=data_values.shape[0], color="white", linewidth=0.5
    )

    if filename:
        if not os.path.exists("./visuals"):
            os.makedirs("./visuals")
        plt.savefig(f"./visuals/{filename}", bbox_inches="tight", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process ranking plot")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file containing the results.")
    parser.add_argument("--include_vqascore", action="store_true")
    args = parser.parse_args()

    df = load_data(args.csv_path)
    hold = process_data(df, args.include_vqascore)
    pivot_table = create_pivot_table(hold, args.include_vqascore)
    plot(pivot_table, "rankings.png")
    print(f"plot saved as {os.path.abspath('.')}/visuals/rankings.png")


if __name__ == "__main__":
    main()
