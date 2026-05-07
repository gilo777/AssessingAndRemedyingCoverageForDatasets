import pandas as pd
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from Graphs.GraphsPlot import plot_graph_10
from Mups.DeepDiver import deepdiver


def plot_graph_10_from_csv(
    csv_path,
    feature_cols,
    label_col,
    tau,
    algorithm,
    subgroup_train_sizes=None,
    subgroup_test_size=20
):
    df = pd.read_csv(csv_path)

    domains = [
        sorted(df[col].dropna().unique().tolist())
        for col in feature_cols
    ]

    results = plot_graph_10(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        domains=domains,
        tau=tau,
        algorithm=algorithm,
        subgroup_train_sizes=subgroup_train_sizes,
        subgroup_test_size=subgroup_test_size
    )

    return results


def main():
    results = plot_graph_10_from_csv(
        csv_path=os.path.join(PROJECT_ROOT, "Online_Retail.csv"),
        feature_cols=["A1", "A2", "A3"],
        label_col="label",
        tau=10,
        algorithm=deepdiver,
        subgroup_train_sizes=[0, 20, 40, 60, 80],
        subgroup_test_size=20
    )

    print(results)


if __name__ == "__main__":
    main()
