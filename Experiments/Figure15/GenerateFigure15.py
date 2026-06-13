import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

# --- make the project root importable so `Mups.*` resolves -------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Algorithms.Mups.DeepDiver import pattern_diver
from Algorithms.Mups.MutualFuncs import level as pattern_level


# line styles cycled across the max_level curves, in plot order
_LINE_STYLES = [("o", "-"), ("s", "--"), ("^", ":"), ("D", "-."), ("v", "-"), ("*", "--")]


@dataclass
class Experiment15Config:

    # --- attributes of interest ------------------------------------------------
    # If None, columns are auto-detected as "categorical" (2..max_cardinality
    # distinct values). For real datasets you'll normally set this explicitly so
    # the projection order (first d columns) is the one you intend.
    feature_cols: Optional[List[str]] = None
    label_col: Optional[str] = None         # excluded from auto-detection
    max_cardinality: int = 500               # auto-detect upper bound

    # --- the dimension sweep (x-axis) -----------------------------------------
    # For each d we project onto feature_cols[:d]. Values larger than the number
    # of available feature columns are dropped automatically.
    dims: List[int] = field(default_factory=lambda: [5, 10, 15, 20])

    # --- the level cap, one curve per value -----------------------------------
    max_levels: List[int] = field(default_factory=lambda: [2, 4, 6, 8])

    # --- coverage threshold ----------------------------------------------------
    # threshold_rate maps to tau = max(1, round(rate * n)); the paper uses 0.1%.
    # An explicit `tau` (if not None) takes precedence -- handy for toy data
    # where 0.1% of a small n rounds down to 0.
    threshold_rate: Optional[float] = 0.001
    tau: Optional[int] = None

    # --- data shaping ----------------------------------------------------------
    sample_size: Optional[int] = None       # subsample n rows if set
    random_state: int = 42
    n_bins: Optional[int] = None            # bucketize high-cardinality numerics (paper §II)

    # --- missing-value handling ------------------------------------------------
    # "fill": treat missing as its own category (sentinel) -- no row loss, the
    #         sensible default for "found data" coverage analysis.
    # "drop": drop rows with any NaN in the feature columns (the old behaviour);
    #         raises an actionable error if that empties the frame.
    na_policy: str = "fill"
    na_sentinel: str = "__NA__"

    # CSV encodings tried in order before giving up (real exports are often not
    # UTF-8; AirBnB-style files are typically latin-1 / Windows-1252).
    encodings: List[str] = field(default_factory=lambda: ["utf-8", "latin-1", "cp1252"])

    # --- plotting / output -----------------------------------------------------
    runtime_log_scale: bool = True          # Figure 15 uses a log y-axis
    show: bool = False
    save: bool = True
    save_dir: Optional[str] = None          # default: Figure15/


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _read_csv_robust(csv_path: str, config: Experiment15Config) -> pd.DataFrame:

    last_err = None
    for enc in config.encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError as err:
            last_err = err
            continue
    # Last resort: decode lossily rather than crash, and say so.
    print(
        f"  [warn] could not decode {os.path.basename(csv_path)} with "
        f"{config.encodings}; falling back to lossy utf-8 (errors replaced)."
    )
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    return df


def _select_features(df: pd.DataFrame, config: Experiment15Config) -> List[str]:
    if config.feature_cols is not None:
        missing = [c for c in config.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not in dataframe: {missing}")
        return list(config.feature_cols)

    cols = []
    for c in df.columns:
        if config.label_col is not None and c == config.label_col:
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= config.max_cardinality:
            cols.append(c)

    if not cols:
        raise ValueError(
            "No categorical feature columns auto-detected. "
            "Set feature_cols explicitly, raise max_cardinality, or set n_bins."
        )
    return cols


def _prepare_df(df: pd.DataFrame, cols: List[str], config: Experiment15Config) -> pd.DataFrame:
    df = df.copy()

    if config.n_bins:
        for c in cols:
            if (
                pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique(dropna=True) > config.n_bins
            ):
                df[c] = pd.cut(df[c], bins=config.n_bins).astype(str)

    # Surface missingness so nothing is silently lost or filled.
    na_counts = df[cols].isna().sum()
    na_cols = na_counts[na_counts > 0]
    if len(na_cols) > 0:
        worst = na_counts.sort_values(ascending=False).head(5)
        summary = ", ".join(f"{c}={int(v)}" for c, v in worst.items())
        print(f"  [na] missing values in feature columns (top): {summary}")

    if config.na_policy == "drop":
        before = len(df)
        df = df.dropna(subset=cols)
        if len(df) == 0:
            worst = na_counts.sort_values(ascending=False).head(5)
            culprits = ", ".join(f"{c} ({int(v)} NaN)" for c, v in worst.items())
            raise ValueError(
                "Dropping rows with NaN in the feature columns removed every row "
                f"(started with {before}). The NaN-heaviest columns are: {culprits}.\n"
                "Fix by either: (a) set na_policy='fill' to treat missing as its "
                "own category, or (b) set feature_cols explicitly to the attributes "
                "of interest (the paper picks specific attributes, not every "
                "categorical column)."
            )
    elif config.na_policy == "fill":
        df[cols] = df[cols].astype(object).where(df[cols].notna(), config.na_sentinel)
    else:
        raise ValueError(f"Unknown na_policy {config.na_policy!r}; use 'fill' or 'drop'.")

    if config.sample_size is not None and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.random_state)

    return df


def _build_projection(df: pd.DataFrame, cols: List[str]):

    values = df[cols].to_numpy(dtype=object).tolist()
    dataset = [tuple(row) for row in values]

    domains = []
    for c in cols:
        uniq = df[c].dropna().unique().tolist()
        try:
            uniq = sorted(uniq)
        except TypeError:
            pass
        domains.append(uniq)

    return dataset, domains


def _resolve_dims(requested: List[int], available: int) -> List[int]:

    seen = set()
    dims = []
    for d in requested:
        d = int(d)
        if 1 <= d <= available and d not in seen:
            seen.add(d)
            dims.append(d)
    return dims


def _resolve_tau(n: int, config: Experiment15Config) -> int:
    if config.tau is not None:
        return int(config.tau)
    return max(1, round(config.threshold_rate * n))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_experiment(
    csv_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    config: Optional[Experiment15Config] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    config = config or Experiment15Config()

    if dataset_name is None:
        dataset_name = Path(csv_path).stem if csv_path else "dataset"
    elif dataset_name.lower().endswith(".csv"):
        dataset_name = dataset_name[:-4]

    if df is None:
        if csv_path is None:
            raise ValueError("Provide either csv_path or df.")
        df = _read_csv_robust(csv_path, config)

    feature_cols = _select_features(df, config)
    df = _prepare_df(df, feature_cols, config)
    n = len(df)
    if n == 0:
        raise ValueError("No rows left after dropping NaNs / sampling.")

    dims = _resolve_dims(config.dims, available=len(feature_cols))
    if not dims:
        raise ValueError(
            f"None of dims={config.dims} fit the {len(feature_cols)} available "
            f"feature columns. Lower the dims or add feature columns."
        )

    tau = _resolve_tau(n, config)

    rows = []
    for d in dims:
        cols_d = feature_cols[:d]
        dataset, domains = _build_projection(df, cols_d)

        for max_level in config.max_levels:
            start = time.perf_counter()
            mups = pattern_diver(dataset, domains, tau, max_level=max_level)
            runtime = time.perf_counter() - start

            # sanity: nothing above the cap slipped through
            assert all(pattern_level(m) <= max_level for m in mups), \
                f"level cap violated at d={d}, max_level={max_level}"

            rows.append({
                "dimensions": d,
                "max_level": max_level,
                "tau": tau,
                "n": n,
                "runtime": runtime,
                "num_mups": len(mups),
            })
            print(
                f"  d={d:>3}  max_level={max_level:>2}  "
                f"tau={tau:<5} runtime={runtime:8.4f}s  #MUPs={len(mups)}"
            )

    results_df = pd.DataFrame(rows)
    _plot(results_df, dataset_name, config)
    return results_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(results_df: pd.DataFrame, dataset_name: str, config: Experiment15Config) -> None:
    import matplotlib
    if not config.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # plot higher caps first so the legend reads 8, 6, 4, 2 like the paper
    levels_sorted = sorted(results_df["max_level"].unique(), reverse=True)
    for i, lvl in enumerate(levels_sorted):
        sub = results_df[results_df["max_level"] == lvl].sort_values("dimensions")
        marker, linestyle = _LINE_STYLES[i % len(_LINE_STYLES)]
        ax.plot(
            sub["dimensions"],
            sub["runtime"],
            marker=marker,
            linestyle=linestyle,
            label=f"max \u2113 = {lvl}",
        )

    ax.set_xlabel("Dimensions (number of attributes)")
    ax.set_ylabel("Runtime (s)")
    if config.runtime_log_scale:
        ax.set_yscale("log")
    ax.legend()

    tau = int(results_df["tau"].iloc[0])
    n = int(results_df["n"].iloc[0])
    ax.set_title(
        f"MUP identification with DeepDiver, varying dimensions "
        f"({dataset_name}, n={n}, \u03c4={tau})"
    )

    if config.save:
        save_dir = config.save_dir or SCRIPT_DIR
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"{dataset_name}_figure15.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"  [Fig 15] saved -> {png_path}")

    if config.show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-contained smoke test (no CSV needed)
# ---------------------------------------------------------------------------

def _smoke_test():
    """Tiny synthetic dataset, just to confirm the pipeline runs end-to-end."""
    import random

    rng = random.Random(0)
    n_attrs = 8
    n_rows = 200
    # binary attributes; a couple are skewed so some low-level MUPs appear
    rows = []
    for _ in range(n_rows):
        rows.append({
            f"A{i}": (0 if rng.random() < 0.85 else 1) if i < 3 else rng.randint(0, 1)
            for i in range(n_attrs)
        })
    df = pd.DataFrame(rows)

    config = Experiment15Config(
        feature_cols=[f"A{i}" for i in range(n_attrs)],
        dims=[2, 4, 6, 8],
        max_levels=[2, 4, 6],
        tau=10,            # explicit, since 0.1% of 200 rounds to 0
        save=False,
        show=False,
    )

    print("Running Experiment 15 smoke test...")
    results = run_experiment(df=df, dataset_name="smoke", config=config)
    print("\nResults:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    _smoke_test()