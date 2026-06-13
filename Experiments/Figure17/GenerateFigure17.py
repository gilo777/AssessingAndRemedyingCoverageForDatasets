import multiprocessing as mp
import os
import queue
import sys
import time
from itertools import combinations
from math import prod

import matplotlib
matplotlib.use("Agg")  # headless: render to files, no GUI
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from Algorithms.Greedy.GreedyHelper import uncovered_patterns_at_level
from Algorithms.Greedy.Greedy import greedy_coverage_enhancement
from Algorithms.Mups.DeepDiver import pattern_diver

DATASETS_DIR = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = SCRIPT_DIR


# tau = 0.1% of n, matching the paper's standard threshold.
TAU_RATE = 0.001
# Target levels (lambda) to sweep — Figure 17 plots one curve per lambda value.
LAMBDAS = [3, 4, 5, 6]
# Pre-flight safety cap: skip greedy if M_lambda would exceed this many patterns.
# Prevents memory blow-up on large d with high lambda.
MAX_INPUT = 300_000
# Per-call wall-clock limits for the subprocess runs.
MUP_TIMEOUT = 600
GREEDY_TIMEOUT = 1200

DATASET_CONFIGS = [
    {
        "name": "Compas",
        "csv": "CompasDataset.csv",
        "encoding": "utf-8",
        # Columns ordered from most to least discriminative for coverage analysis.
        # The experiment projects onto the first d of these for each d value.
        "ordered_cols": [
            "sex", "c_charge_degree", "age_cat", "score_text", "v_score_text",
            "race", "decile_score", "juv_misd_count", "juv_other_count",
            "juv_fel_count",
        ],
    },
    {
        "name": "AirBnB",
        "csv": "AirBnbListingsDatasets.csv",
        "encoding": "latin-1",
        "ordered_cols": [
            "host_is_superhost", "instant_bookable", "host_identity_verified",
            "room_type", "review_scores_value", "review_scores_cleanliness",
            "review_scores_location", "review_scores_accuracy",
            "review_scores_checkin", "review_scores_communication",
            "host_since", "accommodates",
        ],
    },
]

D_MIN = 5
D_MAX = 10
D_STEP = 2
YLIM = (1e-2, 1e2)   # y-axis limits for the log-scale plot


# ---- data preparation -------------------------------------------------------
def load_prepared(config):
    """Load the CSV and drop rows with NaN in any of the chosen columns once, so
    n is fixed across every (d, lambda)."""
    # Using a single cleaned copy ensures the dataset size n (and therefore tau)
    # is the same for every (d, lambda) pair — making the runtime comparison fair.
    path = os.path.join(DATASETS_DIR, config["csv"])
    df = pd.read_csv(path, encoding=config["encoding"])
    df = df.dropna(subset=config["ordered_cols"]).reset_index(drop=True)
    return df


def project(df, cols):
    """Build the (dataset, domains) projection onto the given columns."""
    domains = [sorted(df[c].dropna().unique().tolist(), key=str) for c in cols]
    dataset = [tuple(row) for row in df[cols].to_numpy()]
    return dataset, domains


# ---- subprocess plumbing (timeout + memory blow-up protection) --------------
def _worker(q, func, args):
    """Target function for the subprocess; puts (status, result) on the queue."""
    try:
        q.put(("ok", func(*args)))
    except MemoryError:
        q.put(("mem", None))
    except Exception as exc:  # noqa: BLE001 - report any failure as a gap
        q.put(("err", repr(exc)))


def run_with_timeout(func, args, timeout):
    """Run func(*args) in a child process; return ("timeout", None) if it exceeds timeout seconds.

    Why a subprocess instead of threading?
    - Python threads share the GIL so a CPU-bound timeout cannot interrupt a
      running computation.  A separate process can be terminated cleanly.
    - "spawn" context (not "fork") avoids inheriting large parent-process state,
      which reduces memory overhead and prevents fork-related deadlocks on macOS.
    """
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(result_q, func, args))
    proc.start()
    try:
        status, value = result_q.get(timeout=timeout)
    except queue.Empty:
        proc.terminate()
        proc.join()
        return ("timeout", None)
    proc.join()
    return (status, value)


def estimate_input_size(mups, domains, target_level, cap):
    """Estimate |M_lambda| without actually materialising it.

    M_lambda can be exponentially large in the number of free attributes.
    This function counts the implied level-lambda patterns by iterating over
    MUPs and their free-position combinations, stopping as soon as the estimate
    exceeds `cap`.  Used as a pre-flight check before committing to greedy.
    """
    total = 0
    for mup in mups:
        lvl = sum(1 for v in mup if v is not None)
        if lvl > target_level:
            continue
        missing = target_level - lvl
        x_positions = [i for i, v in enumerate(mup) if v is None]
        if missing > len(x_positions):
            continue
        # Each combination of positions contributes the Cartesian product of
        # the corresponding domain sizes.
        for combo in combinations(x_positions, missing):
            total += prod(len(domains[i]) for i in combo)
            if total > cap:
                return total
    return total


def greedy_point(mups, domains, target_level, cap):
    """Compute one (d, lambda) data point for Figure 17: run greedy and record runtime.

    Returns a dict with keys:
    - "status": "ok" | "too_large" (input exceeded cap — greedy skipped)
    - "runtime": wall-clock seconds for greedy_coverage_enhancement
    - "input": |M_lambda| (number of patterns fed to greedy)
    - "output": number of suggested tuples returned by greedy
    """
    # Pre-flight: abort before allocating M_lambda if it would be too large.
    estimate = estimate_input_size(mups, domains, target_level, cap)
    if estimate > cap:
        return {"status": "too_large", "input_est": estimate}

    patterns_to_hit = uncovered_patterns_at_level(mups, domains, target_level)
    if not patterns_to_hit:
        # All MUPs are already more specific than target_level, or there are none.
        return {"status": "ok", "runtime": 0.0, "input": 0, "output": 0}

    start = time.perf_counter()
    plan = greedy_coverage_enhancement(patterns_to_hit, domains)
    runtime = time.perf_counter() - start
    return {
        "status": "ok",
        "runtime": runtime,
        "input": len(patterns_to_hit),
        "output": len(plan),
    }


# ---- the sweep --------------------------------------------------------------
def run_for_dataset(config):
    """Sweep all (d, lambda) combinations for one dataset and collect results.

    Structure:
    - Outer loop: d from D_MIN to min(D_MAX, available_cols), step D_STEP.
    - For each d, run DeepDiver in a subprocess (with MUP_TIMEOUT).
    - Inner loop: lambda values in LAMBDAS that are <= d.
    - For each (d, lambda), run greedy_point in a subprocess (GREEDY_TIMEOUT).

    Each cell is recorded with status ("ok", "timeout", "mem", "err",
    "too_large") so gaps in the plot are explicit rather than silent.
    """
    print("=" * 70)
    print(f"DATASET: {config['name']}  ({config['csv']})")
    df = load_prepared(config)
    n = len(df)
    tau = max(1, round(TAU_RATE * n))
    cols = config["ordered_cols"]
    d_max = len(cols)
    print(f"  rows (n): {n}   tau (={TAU_RATE:.3%} of n): {tau}   d: {D_MIN}..{min(d_max, D_MAX)}")
    print("-" * 70)

    rows = []
    for d in range(D_MIN, min(d_max, D_MAX) + 1, D_STEP):
        dataset, domains = project(df, cols[:d])

        # Run DeepDiver in a child process so a timeout or OOM doesn't crash
        # the whole experiment — just marks this d as failed and continues.
        status, mups = run_with_timeout(pattern_diver, (dataset, domains, tau), MUP_TIMEOUT)
        if status != "ok":
            print(f"  d={d:<2}  DEEPDIVER {status} -> all lambdas at this d skipped")
            for lam in LAMBDAS:
                if lam <= d:
                    rows.append({"dataset": config["name"], "d": d, "lambda": lam,
                                 "status": f"mup_{status}", "runtime": None,
                                 "input": None, "output": None})
            continue

        mups = list(mups)
        print(f"  d={d:<2}  cols={cols[:d]}")
        print(f"        MUPs found: {len(mups)}")

        for lam in LAMBDAS:
            if lam > d:
                continue
            status, value = run_with_timeout(
                greedy_point, (mups, domains, lam, MAX_INPUT), GREEDY_TIMEOUT
            )
            if status != "ok":
                print(f"        lambda={lam}: greedy {status}")
                rows.append({"dataset": config["name"], "d": d, "lambda": lam,
                             "status": f"greedy_{status}", "runtime": None,
                             "input": None, "output": None})
            elif value["status"] == "ok":
                rt, isz, osz = value["runtime"], value["input"], value["output"]
                print(f"        lambda={lam}: {rt:8.3f}s   input={isz:>7}  output={osz:>6}")
                rows.append({"dataset": config["name"], "d": d, "lambda": lam,
                             "status": "ok", "runtime": rt, "input": isz, "output": osz})
            else:  # too_large
                print(f"        lambda={lam}: skipped (input ~{value['input_est']} > {MAX_INPUT})")
                rows.append({"dataset": config["name"], "d": d, "lambda": lam,
                             "status": "too_large", "runtime": None,
                             "input": value["input_est"], "output": None})
    print()
    return rows


# ---- plotting ---------------------------------------------------------------
MARKERS = {3: "o", 4: "s", 5: "^", 6: "D"}


def _curve(rows, lam, ykey):
    """Extract (d, y) points for a given lambda, skipping non-"ok" entries."""
    pts = sorted(
        (r["d"], r[ykey]) for r in rows
        if r["lambda"] == lam and r["status"] == "ok" and r[ykey] is not None
    )
    return [d for d, _ in pts], [y for _, y in pts]


def plot_figure17(rows, name):
    """Render Figure 17: greedy runtime vs. dimensions, one curve per lambda.

    Semi-log y-axis (semilogy) matches the paper.  Missing data points (timeout,
    too_large, etc.) simply leave gaps in the curve rather than causing errors.
    """
    plt.figure(figsize=(6, 4))
    for lam in LAMBDAS:
        xs, ys = _curve(rows, lam, "runtime")
        if xs:
            plt.semilogy(xs, ys, marker=MARKERS[lam], label=f"ℓ = {lam}")
    plt.ylim(*YLIM)
    plt.xlabel("Dimensions")
    plt.ylabel("Runtime (s)")
    plt.title(f"Fig. 17 reconstruction: Coverage Enhancement (Greedy) — {name}")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    out = os.path.join(OUTPUT_DIR, f"Figure17_{name}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved: {out}")


def generate(config):
    """Run the full Figure-17 experiment for one dataset and save the plot."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = run_for_dataset(config)
    plot_figure17(rows, config["name"])


def main():
    for config in DATASET_CONFIGS:
        generate(config)


if __name__ == "__main__":
    main()