import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: render to file, no GUI window
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from Algorithms.Mups.DeepDiver import pattern_diver
from Experiments.Figure10.MupConstants import MUP_CONFIGS


def pattern_to_string(pattern):
    return "".join("X" if v is None else str(v) for v in pattern)


def row_matches_pattern(row, pattern):
    for row_val, pattern_val in zip(row, pattern):
        if pattern_val is not None and row_val != pattern_val:
            return False
    return True


def get_rows_matching_pattern(df, feature_cols, pattern):
    mask = df[feature_cols].apply(
        lambda row: row_matches_pattern(tuple(row), pattern),
        axis=1
    )
    return df[mask].copy()


def plot_graph_10(
    df,
    feature_cols,
    label_col,
    domains,
    tau,
    algorithm,
    subgroup_train_sizes=None,
    subgroup_test_size=20,
    random_state=42,
    model_feature_cols=None,
    subgroup_pattern=None
):
    """
    Plots a Figure-10-style graph.

    Parameters:
    df - pandas DataFrame
    feature_cols - coverage dimensions: used to find MUPs and define the subgroup
    label_col - target column
    domains - list of possible values for each feature
    tau - coverage threshold
    algorithm - one of your MUP algorithms:
                pattern_diver / pattern_breaker / pattern_combiner
    subgroup_train_sizes - list like [0,20,40,60,80]
    subgroup_test_size - number of subgroup rows used for testing
    model_feature_cols - columns the classifier trains on. Defaults to
                feature_cols. Pass a richer set than the coverage dimensions so
                subgroup members differ in feature space (paper Figure 10), which
                lets added subgroup data actually improve the model.
    subgroup_pattern - if given (a tuple over feature_cols, with None == 'X' for
                unspecified cells), the experiment uses exactly this subgroup
                instead of discovering one with `algorithm`. Use this to pin a
                vetted MUP so the figure is deterministic. When None, the old
                behavior applies: run `algorithm` and pick the MUP with the most
                label variety.
    """

    if model_feature_cols is None:
        model_feature_cols = feature_cols

    dataset = [tuple(row) for row in df[feature_cols].to_numpy()]

    if subgroup_pattern is not None:
        chosen_mup = tuple(subgroup_pattern)
        if len(chosen_mup) != len(feature_cols):
            raise ValueError(
                f"subgroup_pattern has {len(chosen_mup)} cells but there are "
                f"{len(feature_cols)} feature_cols."
            )
    else:
        mups = algorithm(dataset, domains, tau)

        if not mups:
            raise ValueError("No MUPs found. Try increasing tau.")

        min_rows = subgroup_test_size + (max(subgroup_train_sizes) if subgroup_train_sizes else 1)

        def subgroup_size(mup):
            return len(get_rows_matching_pattern(df, feature_cols, mup))

        def label_variety(mup):
            rows = get_rows_matching_pattern(df, feature_cols, mup)
            if len(rows) < min_rows:
                return -1.0
            shares = rows[label_col].value_counts(normalize=True)
            if len(shares) < 2:
                return 0.0
            return float(shares.min())

        chosen_mup = max(mups, key=label_variety)
        if label_variety(chosen_mup) < 0:
            chosen_mup = max(mups, key=subgroup_size)

    subgroup_df = get_rows_matching_pattern(df, feature_cols, chosen_mup)
    non_subgroup_df = df.drop(subgroup_df.index).copy()

    print("Chosen MUP:", chosen_mup)
    print("Chosen pattern:", pattern_to_string(chosen_mup))
    print("Subgroup rows:", len(subgroup_df))

    if subgroup_train_sizes is None:
        available = len(subgroup_df) - subgroup_test_size

        if available <= 0:
            raise ValueError(
                "Not enough subgroup rows. Try lowering subgroup_test_size."
            )

        step = max(1, available // 4)
        subgroup_train_sizes = list(range(0, available + 1, step))

        if subgroup_train_sizes[-1] != available:
            subgroup_train_sizes.append(available)

    needed = subgroup_test_size + max(subgroup_train_sizes)

    if len(subgroup_df) < needed:
        raise ValueError(
            f"Not enough subgroup rows.\n"
            f"Rows found: {len(subgroup_df)}\n"
            f"Rows needed: {needed}\n"
            f"Try smaller subgroup_train_sizes or subgroup_test_size."
        )

    subgroup_test = subgroup_df.sample(
        n=subgroup_test_size,
        random_state=random_state
    )

    subgroup_remaining = subgroup_df.drop(subgroup_test.index)

    non_subgroup_train, non_subgroup_test = train_test_split(
        non_subgroup_df,
        test_size=0.25,
        random_state=random_state,
        stratify=non_subgroup_df[label_col]
    )

    overall_test = pd.concat([non_subgroup_test, subgroup_test])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), model_feature_cols)
        ]
    )

    results = []

    for k in subgroup_train_sizes:
        subgroup_train_k = subgroup_remaining.sample(
            n=k,
            random_state=random_state
        )

        train_df = pd.concat([non_subgroup_train, subgroup_train_k])

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", DecisionTreeClassifier(random_state=random_state))
            ]
        )

        model.fit(train_df[model_feature_cols], train_df[label_col])

        overall_pred = model.predict(overall_test[model_feature_cols])
        subgroup_pred = model.predict(subgroup_test[model_feature_cols])

        overall_accuracy = accuracy_score(
            overall_test[label_col],
            overall_pred
        )

        subgroup_accuracy = accuracy_score(
            subgroup_test[label_col],
            subgroup_pred
        )

        subgroup_f1 = f1_score(
            subgroup_test[label_col],
            subgroup_pred,
            average="weighted",
            zero_division=0
        )

        results.append({
            "subgroup_train_size": k,
            "overall_accuracy": overall_accuracy,
            "subgroup_accuracy": subgroup_accuracy,
            "subgroup_f1": subgroup_f1
        })

    results_df = pd.DataFrame(results)

    fig, ax1 = plt.subplots()

    ax1.plot(
        results_df["subgroup_train_size"],
        results_df["overall_accuracy"],
        marker="o",
        label="Overall Accuracy"
    )

    ax1.plot(
        results_df["subgroup_train_size"],
        results_df["subgroup_accuracy"],
        marker="o",
        label="Subgroup Accuracy"
    )

    ax1.set_xlabel("Number of subgroup records added")
    ax1.set_ylabel("Accuracy")

    ax2 = ax1.twinx()

    ax2.plot(
        results_df["subgroup_train_size"],
        results_df["subgroup_f1"],
        marker="s",
        linestyle="--",
        label="Subgroup F1"
    )

    ax2.set_ylabel("F1-measure")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower right"
    )

    plt.title(f"Effect of Lack of Coverage — MUP {pattern_to_string(chosen_mup)}")
    plt.show()

    return results_df

DATASETS_DIR = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = SCRIPT_DIR


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
