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

    df = clean_categorical_dataframe(df, feature_cols)
    dataset, domains = build_dataset_and_domains(df, feature_cols)

    n = len(dataset)

    print("Rows:", n)
    print("Features:", feature_cols)
    print("Domain sizes:", [len(domain) for domain in domains])
    print()

    results = []

    for threshold_rate in threshold_rates:
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

            patterns_to_hit = uncovered_patterns_at_level(
                mups=mups,
                domains=domains,
                target_level=ell,
            )

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

    df = pd.read_csv(csv_path, keep_default_na=False)

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
