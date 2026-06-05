import pandas as pd
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from SetData import plot_graph_10_from_csv
from Mups.DeepDiver import pattern_diver


def main():
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "Data", "Online_Retail.csv"), encoding="latin-1")
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Country", "UnitPrice", "Quantity"])

    df["high_quantity"] = (df["Quantity"] > df["Quantity"].median()).astype(int)

    df["price_group"] = pd.cut(
        df["UnitPrice"],
        bins=3,
        labels=["low", "medium", "high"]
    )

    output_csv = os.path.join(PROJECT_ROOT, "Online_Retail_ready.csv")
    df.to_csv(output_csv, index=False)

    results = plot_graph_10_from_csv(
        csv_path="Online_Retail_ready.csv",
        feature_cols=["Country", "price_group"],
        label_col="high_quantity",
        tau=10,
        algorithm=pattern_diver,
        subgroup_train_sizes=[0, 5, 10, 15],
        subgroup_test_size=5
    )

    print(results)


if __name__ == "__main__":
    main()