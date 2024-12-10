# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_diversity_vs_quality(df, cfg, output_file):
    temp_df = df[df["cfg_scale"] == cfg]
    style_order = sorted(list(set(df["dataset"])))
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=temp_df,
        x="coverage",
        y="precision",
        style="dataset",
        hue="model_id",
        s=200,
        alpha=0.7,
        palette="colorblind",
        style_order=style_order,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.grid(visible=True)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.xlabel("coverage",fontsize=18)
    plt.ylabel("precision",fontsize=18)
    plt.legend(handles[-12:], labels[-12:], bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if not os.path.exists("./visuals"):
        os.makedirs("./visuals")
    plt.savefig(f"./visuals/{output_file}", bbox_inches="tight")


def plot_fid_and_consistency(df, cfg, output_file, consistency_metric):
    style_order = sorted(list(set(df["dataset"])))
    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    
    sns.scatterplot(
        data=df[df["cfg_scale"] == cfg],
        y="model_id",
        x="fid_torchmetrics",
        s=100,
        hue="model_id",
        alpha=0.7,
        style="dataset",
        palette="colorblind",
        style_order=style_order,
        ax=axs[0],
    )
    axs[0].get_legend().remove()
    axs[0].grid(visible=True, axis="both")
    axs[0].set_xlim([0, 100])
    axs[0].tick_params(axis='both', which='major', labelsize=13)
    axs[0].tick_params(axis='both', which='minor', labelsize=13)
    axs[0].set_xlabel("fid_torchmetrics", fontsize=13)
    axs[0].set_ylabel("model_id", fontsize=13)
    sns.scatterplot(
        data=df[df["cfg_scale"] == cfg],
        y="model_id",
        x=consistency_metric,
        s=100,
        hue="model_id",
        alpha=0.7,
        style="dataset",
        palette="colorblind",
        style_order=style_order,
        ax=axs[1],
    )
    for a, b in zip(axs[0].get_yticklabels(), axs[1].get_yticklabels(), strict=False):
        assert a._text == b._text, print(a, b)
    axs[1].set_yticks(axs[0].get_yticks(), [" " for _ in axs[1].get_yticks()])
    axs[1].set_ylabel("")
    axs[1].grid(visible=True, axis="both")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11)
    axs[1].set_xlim([15, 30])
    axs[1].tick_params(axis='both', which='major', labelsize=13)
    axs[1].tick_params(axis='both', which='minor', labelsize=13)
    axs[1].set_xlabel(consistency_metric, fontsize=13)
    plt.tight_layout()
    if not os.path.exists("./visuals"):
        os.makedirs("./visuals")
    plt.savefig(f"./visuals/{output_file}", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Process and plot dataset metrics.")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file containing the results.")
    parser.add_argument("--cfg", type=float, default=7.5, help="select a cfg for plotting.")
    parser.add_argument(
        "--consistency_metric",
        type=str,
        default="clipscore",
        help="choose between clipscore, vqascore",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    df["dataset"] = [i.split(".")[-1] for i in df["dataset"]]
    plot_diversity_vs_quality(df, args.cfg, "diversity_vs_quality.png")
    print(f"plot saved as {os.path.abspath('.')}/visuals/diversity_vs_quality.png")
    plot_fid_and_consistency(
        df, args.cfg, "fid_and_consistency.png", consistency_metric=args.consistency_metric
    )
    print(f"plot saved as {os.path.abspath('.')}/visuals/fid_and_consistency.png")


if __name__ == "__main__":
    main()
