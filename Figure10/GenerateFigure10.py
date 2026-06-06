"""
Generate the Figure-10 graph for a dataset using its hardcoded MUP.

The MUP and the exact experiment settings live in ``MupConstants.py``. This
script just loads the chosen dataset, pins that MUP as the subgroup, and runs
the Figure-10 coverage experiment via ``Graphs.GraphsPlot.plot_graph_10``,
saving the resulting plot as a PNG.

Usage:
    python3 Figure10/GenerateFigure10.py                 # all datasets
    python3 Figure10/GenerateFigure10.py "Adult Income"  # just one
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: render to file, no GUI window
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from Graphs.GraphsPlot import plot_graph_10
from Mups.DeepDiver import pattern_diver
from Figure10.MupConstants import MUP_CONFIGS

DATASETS_DIR = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")


def generate(name, cfg):
    print("=" * 70)
    print(f"DATASET: {name}   MUP: {cfg['subgroup_pattern']}")

    df = pd.read_csv(os.path.join(DATASETS_DIR, cfg["csv"]), encoding=cfg["encoding"])

    make_label = cfg.get("make_label")
    if make_label is not None:
        df = make_label(df)

    feature_cols = cfg["feature_cols"]
    label_col = cfg["label_col"]
    model_feature_cols = cfg["model_feature_cols"]

    # plot_graph_10 does not drop NaNs, so clean every column it touches here.
    cols_needed = list(dict.fromkeys(feature_cols + model_feature_cols + [label_col]))
    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    domains = [
        sorted(df[c].dropna().unique().tolist(), key=str) for c in feature_cols
    ]

    results_df = plot_graph_10(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        domains=domains,
        tau=cfg["tau"],
        algorithm=pattern_diver,                 # unused: subgroup_pattern pins the MUP
        subgroup_train_sizes=cfg["subgroup_train_sizes"],
        subgroup_test_size=cfg["subgroup_test_size"],
        random_state=cfg["random_state"],
        model_feature_cols=model_feature_cols,
        subgroup_pattern=cfg["subgroup_pattern"],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"Figure10_{name.replace(' ', '')}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    print(results_df.to_string(index=False))
    print(f"saved: {out_path}\n")
    return out_path


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in MUP_CONFIGS:
            raise SystemExit(
                f"Unknown dataset '{name}'. Choices: {list(MUP_CONFIGS)}"
            )
        generate(name, MUP_CONFIGS[name])
    else:
        for name, cfg in MUP_CONFIGS.items():
            generate(name, cfg)


if __name__ == "__main__":
    main()
