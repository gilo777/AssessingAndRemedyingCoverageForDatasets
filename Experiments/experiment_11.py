"""Experiment 11 -- MUP identification, varying threshold (paper Figure 12).

Reproduces the paper's "MUP identification - varying threshold" experiment
(§ V-C-1, Figure 12). For a fixed dataset, dimensionality d and size n, we
sweep the coverage threshold and, at each threshold, run every MUP-identification
algorithm, timing it and recording how many MUPs it finds.

The produced plot mirrors Figure 12:
  * x-axis           : threshold rate (tau as a fraction of n), evenly spaced
  * left  y-axis     : runtime in seconds, one line+marker per algorithm
  * right y-axis     : number of MUPs, drawn as bars behind the lines

All algorithms return the same MUP set, so the bar series is shared; we also use
this as a built-in agreement check across the three implementations.

Everything you might want to vary lives in `Experiment11Config` below, so the
single place to control the experiment is that dataclass (edit the defaults, or
pass your own instance to `run_experiment(config=...)`).

APRIORI from Figure 12 is intentionally not included.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

# --- make the project root importable so `Mups.*` resolves -------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Mups.TopDown import pattern_breaker
from Mups.BottomUp import pattern_combiner
from Mups.DeepDiver import pattern_diver


# name -> (display label used in the paper, callable)
_ALGORITHMS = {
    "pattern_breaker": ("PATTERN-BREAKER", pattern_breaker),
    "pattern_combiner": ("PATTERN-COMBINER", pattern_combiner),
    "pattern_diver": ("DEEPDIVER", pattern_diver),
}

# line styles assigned to algorithms in plot order
_LINE_STYLES = [("o", "-"), ("s", "--"), ("^", ":"), ("D", "-.")]

# Tried in order when no explicit encoding is configured. latin-1 maps every
# byte, so it always decodes (last-resort) even for non-UTF-8 "found data".
_ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


@dataclass
class Experiment11Config:
    """All tunable knobs for Experiment 11. Edit the defaults or pass an instance."""

    # --- attributes of interest ------------------------------------------------
    # If None, columns are auto-detected as "categorical" (2..max_cardinality
    # distinct values). For real datasets you'll normally set this explicitly.
    feature_cols: Optional[List[str]] = None
    label_col: Optional[str] = None          # excluded from auto-detected features

    # --- threshold sweep -------------------------------------------------------
    # Two ways to drive the sweep:
    #   * tau_values     : explicit integer thresholds (clearest for small data)
    #   * threshold_rates: tau is derived as round(rate * n) -- this is what the
    #                      paper does (Figure 12 uses 1e-6 .. 1e-2).
    # tau_values takes precedence if set.
    tau_values: Optional[List[int]] = None
    threshold_rates: List[float] = field(
        default_factory=lambda: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    )

    # --- dataset shaping -------------------------------------------------------
    sample_size: Optional[int] = None        # subsample n (None = use all rows)
    max_dims: Optional[int] = None           # cap number of feature columns
    max_cardinality: int = 20                # auto-detect categorical cutoff
    n_bins: Optional[int] = None             # bucketize high-cardinality numeric cols
    random_state: int = 42
    encoding: Optional[str] = None           # CSV encoding; None = try utf-8 then fall back
    dropna: bool = False                     # True: drop rows with missing feature values
    na_fill: str = "missing"                 # sentinel category used when dropna is False

    # --- algorithms ------------------------------------------------------------
    algorithms: List[str] = field(
        default_factory=lambda: ["pattern_breaker", "pattern_combiner", "pattern_diver"]
    )
    repeats: int = 1                         # take the fastest of N timed runs

    # --- plotting / output -----------------------------------------------------
    runtime_log_scale: bool = False          # Figure 12 uses a linear runtime axis
    show: bool = False                       # plt.show() (use on your own machine)
    save: bool = True
    save_dir: Optional[str] = None           # default: <root>/Graphs/figure11


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _read_csv(csv_path: str, config: Experiment11Config) -> pd.DataFrame:
    """Read a CSV, tolerating non-UTF-8 ("found data") encodings.

    If config.encoding is set it is used as-is. Otherwise a sequence of common
    encodings is tried; latin-1 is the guaranteed fallback since it decodes any
    byte (used by the project's existing Online_Retail loader for the same reason).
    """
    encodings = [config.encoding] if config.encoding else list(_ENCODING_FALLBACKS)
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error


def _select_features(df: pd.DataFrame, config: Experiment11Config) -> List[str]:
    if config.feature_cols is not None:
        cols = [c for c in config.feature_cols]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not found in dataset: {missing}")
    else:
        cols = []
        for c in df.columns:
            if config.label_col is not None and c == config.label_col:
                continue
            n_unique = df[c].nunique(dropna=True)
            if 2 <= n_unique <= config.max_cardinality:
                cols.append(c)
            elif (
                config.n_bins
                and pd.api.types.is_numeric_dtype(df[c])
                and n_unique > config.max_cardinality
            ):
                cols.append(c)  # will be bucketized below
        if not cols:
            raise ValueError(
                "No categorical feature columns auto-detected. "
                "Set feature_cols explicitly, raise max_cardinality, or set n_bins."
            )

    if config.max_dims is not None:
        cols = cols[: config.max_dims]
    return cols


def _build_dataset(df: pd.DataFrame, cols: List[str], config: Experiment11Config):
    df = df.copy()

    # Bucketize high-cardinality numeric columns when requested.
    if config.n_bins:
        for c in cols:
            if (
                pd.api.types.is_numeric_dtype(df[c])
                and df[c].nunique(dropna=True) > config.n_bins
            ):
                df[c] = pd.cut(df[c], bins=config.n_bins).astype(str)

    # Handle missing values. By default we treat NaN as its own category (the
    # paper does the same, e.g. COMPAS's "unknown" marital status), which keeps
    # every row. Set dropna=True to drop rows with missing feature values instead.
    if config.dropna:
        before = len(df)
        df = df.dropna(subset=cols)
        if before != len(df):
            print(f"  dropped {before - len(df)} of {before} row(s) with missing "
                  f"feature values")
    else:
        na_total = int(df[cols].isna().to_numpy().sum())
        if na_total:
            df[cols] = df[cols].astype(object).where(df[cols].notna(), config.na_fill)
            print(f"  filled {na_total} missing value(s) with '{config.na_fill}' "
                  f"(treated as a category)")

    if config.sample_size is not None and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.random_state)

    values = df[cols].to_numpy(dtype=object).tolist()
    dataset = [tuple(row) for row in values]

    domains = []
    for c in cols:
        uniq = df[c].dropna().unique().tolist()
        try:
            uniq = sorted(uniq)
        except TypeError:
            pass  # unorderable mixed values -- leave as-is
        domains.append(uniq)

    return dataset, domains, len(dataset)


def _resolve_thresholds(n: int, config: Experiment11Config):
    """Return an ordered list of (rate, tau) pairs; rate is None in tau mode."""
    pairs = []
    if config.tau_values is not None:
        for t in config.tau_values:
            pairs.append((None, int(t)))
    else:
        for rate in config.threshold_rates:
            pairs.append((rate, max(1, round(rate * n))))
    return pairs


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_experiment(
    csv_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    config: Optional[Experiment11Config] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run Experiment 11 and (by default) save the Figure-12-style plot.

    Provide either `csv_path` or a ready-made `df`. Returns a tidy results
    DataFrame with one row per (threshold, algorithm).
    """
    config = config or Experiment11Config()

    if dataset_name is None:
        dataset_name = Path(csv_path).stem if csv_path else "dataset"
    elif dataset_name.lower().endswith(".csv"):
        dataset_name = dataset_name[:-4]

    if df is None:
        if csv_path is None:
            raise ValueError("Provide either csv_path or df.")
        df = _read_csv(csv_path, config)
        df.columns = df.columns.str.strip()

    cols = _select_features(df, config)
    dataset, domains, n = _build_dataset(df, cols, config)
    if n == 0:
        raise ValueError(
            "No rows left to analyze. The CSV may be empty, or (with dropna=True) "
            "every row had a missing value in a selected feature column. "
            "The default dropna=False keeps rows by treating missing as a category; "
            "you can also set feature_cols explicitly to avoid sparse columns."
        )

    thresholds = _resolve_thresholds(n, config)

    print(f"Experiment 11 on '{dataset_name}': n={n}, d={len(cols)}, "
          f"features={cols}")

    records = []
    for rate, tau in thresholds:
        counts_here = {}
        for name in config.algorithms:
            if name not in _ALGORITHMS:
                raise ValueError(f"Unknown algorithm '{name}'.")
            label, func = _ALGORITHMS[name]

            best_t = None
            num_mups = float("nan")
            try:
                for _ in range(max(1, config.repeats)):
                    start = time.perf_counter()
                    mups = func(dataset, domains, tau)
                    elapsed = time.perf_counter() - start
                    best_t = elapsed if best_t is None else min(best_t, elapsed)
                num_mups = len(mups)
                counts_here[label] = num_mups
            except Exception as exc:  # one failing algorithm shouldn't kill the run
                best_t = float("nan")
                print(f"  [warn] {label} failed at tau={tau}: {exc}")

            records.append(
                {
                    "rate": rate,
                    "tau": tau,
                    "algorithm": label,
                    "runtime_s": best_t if best_t is not None else float("nan"),
                    "num_mups": num_mups,
                }
            )

        distinct = set(counts_here.values())
        if len(distinct) > 1:
            print(f"  [warn] algorithms disagree on MUP count at tau={tau}: "
                  f"{counts_here}")
        label_x = f"rate={rate:g}" if rate is not None else f"tau={tau}"
        any_count = next(iter(distinct), "n/a")
        print(f"  {label_x}: tau={tau}, #MUPs={any_count}")

    results = pd.DataFrame.from_records(records)

    save_path = None
    if config.save:
        save_dir = config.save_dir or os.path.join(PROJECT_ROOT, "Graphs", "figure11")
        save_path = os.path.join(save_dir, f"{dataset_name}_figure11.png")

    if config.save or config.show:
        _plot(records, dataset_name, config, save_path)
        if save_path:
            print(f"  saved figure -> {save_path}")

    return results


# ---------------------------------------------------------------------------
# Plotting (Figure 12 style)
# ---------------------------------------------------------------------------

def _fmt_x(value, use_rate: bool) -> str:
    if use_rate:
        return f"{value:g}"
    return str(int(value))


def _plot(records, dataset_name, config, save_path):
    import matplotlib
    if not config.show:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
    import matplotlib.pyplot as plt

    use_rate = records[0]["rate"] is not None
    x_label = "Threshold rates" if use_rate else "Coverage threshold (tau)"

    # ordered unique x-keys (preserve sweep order, evenly spaced like the paper)
    x_keys = []
    for r in records:
        key = r["rate"] if use_rate else r["tau"]
        if key not in x_keys:
            x_keys.append(key)
    positions = list(range(len(x_keys)))
    pos_of = {k: i for i, k in enumerate(x_keys)}

    # one shared MUP-count series (first finite count per x)
    mup_by_x = {}
    for r in records:
        key = r["rate"] if use_rate else r["tau"]
        val = r["num_mups"]
        if key not in mup_by_x and val == val:  # not NaN
            mup_by_x[key] = val
    mup_vals = [mup_by_x.get(k, 0) for k in x_keys]

    algos = []
    for r in records:
        if r["algorithm"] not in algos:
            algos.append(r["algorithm"])

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # bars (MUP count) behind the runtime lines
    ax2.bar(positions, mup_vals, width=0.6, color="lightsteelblue",
            alpha=0.6, zorder=1, label="# of MUPs")

    for i, algo in enumerate(algos):
        ys = [float("nan")] * len(x_keys)
        for r in records:
            if r["algorithm"] == algo:
                key = r["rate"] if use_rate else r["tau"]
                ys[pos_of[key]] = r["runtime_s"]
        marker, linestyle = _LINE_STYLES[i % len(_LINE_STYLES)]
        ax1.plot(positions, ys, marker=marker, linestyle=linestyle,
                 label=algo, zorder=3)

    # draw ax1 (lines) on top of ax2 (bars)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([_fmt_x(k, use_rate) for k in x_keys])
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Runtime (s)")
    ax2.set_ylabel("# of MUPs")
    if config.runtime_log_scale:
        ax1.set_yscale("log")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(f"MUP Identification - varying threshold ({dataset_name})")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if config.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Tiny self-contained smoke test (no external CSV needed).
    toy = pd.DataFrame(
        {
            "A1": [0, 0, 0, 0, 1, 1, 0, 1],
            "A2": [0, 0, 1, 1, 0, 0, 1, 1],
            "A3": [0, 1, 0, 1, 0, 1, 0, 0],
            "label": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    cfg = Experiment11Config(
        feature_cols=["A1", "A2", "A3"],
        tau_values=[1, 2, 3, 4],
        save=False,
        show=False,
    )
    print(run_experiment(df=toy, dataset_name="toy", config=cfg))