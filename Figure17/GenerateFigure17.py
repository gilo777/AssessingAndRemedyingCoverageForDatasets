"""Reconstruct the paper's Figure 17 (and, for free, Figure 18).

Paper Fig. 17: "Coverage Enhancement with various dimensions using Greedy"
    (AirBnB, n = 1M, tau = 0.1%). x-axis = number of attributes d; one curve per
    target maximum-covered-level lambda in {3,4,5,6}; y-axis = GREEDY runtime (s).
Paper Fig. 18: same runs, plotting input size (# uncovered patterns to hit at
    level lambda) and output size (# data points GREEDY says to collect).

We cannot match the paper's 5..35 attributes on real data, so we sweep d over
however many low-cardinality columns each dataset actually has (ordered
low-cardinality first, so the pattern graph grows gradually):
    * Compas : d = 3..10
    * AirBnB : d = 3..12

For each (d, lambda):
    1. project the dataset onto the first d columns,
    2. find MUPs with DEEPDIVER at threshold tau   (input -- NOT timed),
    3. time GREEDY enhancing coverage up to level lambda   (this is the y-value).

The greedy step can blow up combinatorially when high-cardinality attributes meet
a high lambda (the number of uncovered patterns explodes -> the bit-mask GREEDY
uses becomes astronomically large). So every point runs in a subprocess with a
wall-clock timeout, and a cheap input-size estimate skips points whose pattern
count would exceed MAX_INPUT. Skipped/timed-out points simply leave a gap in the
curve -- which is itself the paper's message: small lambda is cheap, large lambda
is not.

Run:  python3 Figure17/GenerateFigure17.py
"""

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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from Greedy.GreedyHelper import uncovered_patterns_at_level
from Greedy.Greedy import greedy_coverage_enhancement
from Mups.DeepDiver import pattern_diver

DATASETS_DIR = os.path.join(PROJECT_ROOT, "Datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Graphs", "Figure-17")

# ---- experiment knobs -------------------------------------------------------
TAU_RATE = 0.001          # coverage threshold as a fraction of n (paper: 0.1%)
LAMBDAS = [3, 4, 5, 6]    # target maximum-covered-levels -> one curve each
MAX_INPUT = 300_000       # skip a point if the (upper-bound) # patterns to hit exceeds this
MUP_TIMEOUT = 600         # seconds allowed for DEEPDIVER per dimension
GREEDY_TIMEOUT = 1200     # seconds allowed for GREEDY per (d, lambda)

# Columns ordered low-cardinality first. All chosen columns are ~0% null so a
# single up-front dropna keeps n stable across the whole d sweep.
DATASET_CONFIGS = [
    {
        "name": "Compas",
        "csv": "CompasDataset.csv",
        "encoding": "utf-8",
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

D_MIN = 5    # start the x-axis at 5 (paper Fig. 17); below this runtimes are trivial
D_MAX = 10   # cap the x-axis at 10 so both datasets show the same 5..10 window
D_STEP = 2   # gap between successive dimensions on the x-axis (paper-style bigger jumps)
YLIM = (1e-2, 1e2)  # runtime axis range, matching the paper (10^-2 .. 10^2 s)


# ---- data preparation -------------------------------------------------------
def load_prepared(config):
    """Load the CSV and drop rows with NaN in any of the chosen columns once, so
    n is fixed across every (d, lambda)."""
    path = os.path.join(DATASETS_DIR, config["csv"])
    df = pd.read_csv(path, encoding=config["encoding"])
    df = df.dropna(subset=config["ordered_cols"]).reset_index(drop=True)
    return df


def project(df, cols):
    """Return (dataset, domains) for the given columns, mirroring the convention
    used by AlgoTests/RunOnDatasets.py."""
    domains = [sorted(df[c].dropna().unique().tolist(), key=str) for c in cols]
    dataset = [tuple(row) for row in df[cols].to_numpy()]
    return dataset, domains


# ---- subprocess plumbing (timeout + memory blow-up protection) --------------
def _worker(q, func, args):
    try:
        q.put(("ok", func(*args)))
    except MemoryError:
        q.put(("mem", None))
    except Exception as exc:  # noqa: BLE001 - report any failure as a gap
        q.put(("err", repr(exc)))


def run_with_timeout(func, args, timeout):
    """Run func(*args) in a spawned subprocess. Returns (status, value):
    status in {"ok", "timeout", "mem", "err"}."""
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
    """Upper bound on the number of uncovered patterns to hit at target_level,
    summing each MUP's descendant count without materializing them. Bails out as
    soon as it passes `cap` (the real, de-duplicated set is only smaller)."""
    total = 0
    for mup in mups:
        lvl = sum(1 for v in mup if v is not None)
        if lvl > target_level:
            continue
        missing = target_level - lvl
        x_positions = [i for i, v in enumerate(mup) if v is None]
        if missing > len(x_positions):
            continue
        for combo in combinations(x_positions, missing):
            total += prod(len(domains[i]) for i in combo)
            if total > cap:
                return total
    return total


def greedy_point(mups, domains, target_level, cap):
    """Worker: build the level-`target_level` hitting-set input from the MUPs and
    time GREEDY on it. Returns a dict the parent records directly."""
    estimate = estimate_input_size(mups, domains, target_level, cap)
    if estimate > cap:
        return {"status": "too_large", "input_est": estimate}

    patterns_to_hit = uncovered_patterns_at_level(mups, domains, target_level)
    if not patterns_to_hit:
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

        # MUPs depend only on (dataset_d, tau) -- discover once per d, reuse for
        # all lambdas. Not part of the timed runtime, but still guarded.
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
                continue  # a level-lambda pattern needs at least lambda attributes
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
    pts = sorted(
        (r["d"], r[ykey]) for r in rows
        if r["lambda"] == lam and r["status"] == "ok" and r[ykey] is not None
    )
    return [d for d, _ in pts], [y for _, y in pts]


def plot_figure17(rows, name):
    """Runtime (log) vs. dimensions, one curve per lambda."""
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
    """Run the Figure-17 sweep for a single dataset config and save its plot."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = run_for_dataset(config)
    plot_figure17(rows, config["name"])


def main():
    for config in DATASET_CONFIGS:
        generate(config)


if __name__ == "__main__":
    main()
