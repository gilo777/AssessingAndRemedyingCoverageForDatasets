import sys
import time
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Save graph only. Do not open/show a window.

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent          # Experiments/Figure16/
PROJECT_ROOT = SCRIPT_DIR.parent.parent              # project root
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from Algorithms.Mups.DeepDiver import pattern_diver
from Algorithms.Greedy.Greedy import greedy_coverage_enhancement
from Algorithms.Greedy.GreedyHelper import (
    Domains,
    always_valid,
    uncovered_patterns_at_level,
)


Dataset = List[Tuple[Any, ...]]


# ---------------------------------------------------------------------------
# Helpers for already-bucketed / already-categorical datasets
# ---------------------------------------------------------------------------

def clean_categorical_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Normalise categorical columns: fill NaN with "Unknown", cast to str.

    The MUP algorithms require all values to be hashable and non-None (None is
    the wildcard X).  This function ensures that: empty strings and NaN become
    the sentinel "Unknown", and all values are strings for consistent domain
    enumeration.
    """
    df = df.copy()

    for col in feature_cols:
        df[col] = df[col].where(df[col].notna(), "Unknown")
        df[col] = df[col].astype(str)
        df[col] = df[col].replace({"": "Unknown"})

    return df


def build_dataset_and_domains(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[Dataset, Domains]:
    """Convert a cleaned dataframe into the (dataset, domains) format.

    dataset: list of tuples — one per row, values in feature_cols order.
    domains: list of sorted unique-value lists — one per feature column.
    Sorting domains makes iteration order deterministic, which matters for
    reproducible bitmask positions in the greedy algorithm.
    """
    dataset = [
        tuple(row)
        for row in df[feature_cols].to_numpy()
    ]

    domains = [
        sorted(df[col].unique().tolist(), key=str)
        for col in feature_cols
    ]

    return dataset, domains


# ---------------------------------------------------------------------------
# Figure 16 experiment:
# Coverage Enhancement with various thresholds
# ---------------------------------------------------------------------------

def run_graph_16_experiment(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold_rates: List[float],
    target_levels: List[int],
) -> pd.DataFrame:
    """Run the Figure-16 experiment: greedy enhancement runtime vs. threshold rate.

    Figure 16 studies how the greedy coverage-enhancement runtime changes as tau
    (the coverage threshold) varies.  A higher tau means more rows are needed to
    "cover" a pattern, so more MUPs exist and M_lambda (the set of uncovered
    patterns at level lambda) is larger — the greedy algorithm has more work to do.

    For each (threshold_rate, target_level=ell) pair:
    1. Compute tau = ceil(threshold_rate * n).
    2. Run DeepDiver to find all MUPs under tau.
    3. Expand MUPs to M_lambda (all level-ell implied patterns).
    4. Run greedy_coverage_enhancement on M_lambda.
    5. Record runtime, input size (|M_lambda|), and output size (# suggestions).

    The nested loop mirrors the paper's Figure-16 setup: threshold rates on the
    x-axis, one curve per target level (ell).
    """
    df = clean_categorical_dataframe(df, feature_cols)
    dataset, domains = build_dataset_and_domains(df, feature_cols)

    n = len(dataset)

    print("Rows:", n)
    print("Features:", feature_cols)
    print("Domain sizes:", [len(domain) for domain in domains])
    print()

    results = []

    for threshold_rate in threshold_rates:
        # ceil ensures tau >= 1 even for tiny rates on small datasets.
        tau = max(1, math.ceil(threshold_rate * n))

        print("=" * 80)
        print(f"threshold_rate={threshold_rate:g}, tau={tau}")

        mup_start = time.perf_counter()
        mups = pattern_diver(dataset, domains, tau)
        mup_runtime = time.perf_counter() - mup_start

        print(f"MUPs found: {len(mups)}")
        print(f"MUP runtime: {mup_runtime:.4f}s")

        for ell in target_levels:
            if ell > len(feature_cols):
                continue

            print(f"  ell={ell}")

            enhancement_start = time.perf_counter()

            # Expand each MUP into level-ell patterns (M_lambda).
            patterns_to_hit = uncovered_patterns_at_level(
                mups=mups,
                domains=domains,
                target_level=ell,
            )

            # Run greedy set-cover: find the smallest set of tuples that hits
            # every pattern in M_lambda.
            suggestions = greedy_coverage_enhancement(
                patterns_to_hit=patterns_to_hit,
                domains=domains,
                validation_oracle=always_valid,
                generalize_output=False,
            )

            enhancement_runtime = time.perf_counter() - enhancement_start

            print(
                f"    M_lambda={len(patterns_to_hit)}, "
                f"output={len(suggestions)}, "
                f"runtime={enhancement_runtime:.4f}s"
            )

            results.append({
                "threshold_rate": threshold_rate,
                "tau": tau,
                "ell": ell,
                "runtime_sec": enhancement_runtime,
                "num_mups": len(mups),
                "num_patterns_to_hit": len(patterns_to_hit),
                "num_output_tuples": len(suggestions),
                "mup_runtime_sec": mup_runtime,
            })

    results_df = pd.DataFrame(results)

    return results_df


def plot_graph_16(
    results_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Render Figure 16: log-log plot of greedy runtime vs. threshold rate.

    Log-log axes are used because both the threshold rate and the runtime span
    several orders of magnitude.  One line per target level (ell).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for ell, group in results_df.groupby("ell"):
        group = group.sort_values("threshold_rate")

        plt.plot(
            group["threshold_rate"],
            group["runtime_sec"],
            marker="o",
            label=f"Greedy (ℓ={ell})",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Threshold rates")
    plt.ylabel("Runtime (s)")
    plt.title("Figure 16 - Coverage Enhancement with various thresholds")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved graph to: {save_path}")


def generate(csv_path, dataset_name="AdultIncomeDataSet.csv") -> pd.DataFrame:
    """Entry point for the Figure-16 experiment on the Adult Income dataset.

    Validates that the required columns are present, then runs the full
    threshold-sweep experiment and saves the plot.
    """
    df = pd.read_csv(csv_path, keep_default_na=False)

    # These six attributes are the coverage dimensions used in the paper's
    # Figure 16.  They are a mix of demographic and employment features that
    # capture meaningful real-world subgroup structure in the Adult dataset.
    adult_features = [
        "age",
        "workclass",
        "marital.status",
        "relationship",
        "race",
        "sex",
    ]

    missing_columns = [
        col for col in adult_features
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Experiment 16 cannot run on '{dataset_name}'. "
            f"Missing columns: {missing_columns}"
        )

    output_dir = SCRIPT_DIR
    dataset_stem = Path(dataset_name).stem

    graph_path = output_dir / f"graph16_{dataset_stem}.png"

    results_df = run_graph_16_experiment(
        df=df,
        feature_cols=adult_features,
        # Five log-spaced threshold rates matching the paper's Figure 16 x-axis.
        threshold_rates=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        target_levels=[3, 4, 5, 6],
    )

    plot_graph_16(
        results_df=results_df,
        save_path=graph_path,
    )

    return results_df


def run_adult_income_graph_16() -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "Datasets" / "AdultIncomeDataSet.csv"

    return generate(
        csv_path=csv_path,
        dataset_name="AdultIncomeDataSet.csv",
    )


if __name__ == "__main__":
    run_adult_income_graph_16()
