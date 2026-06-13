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


TAU_RATE = 0.001
LAMBDAS = [3, 4, 5, 6]
MAX_INPUT = 300_000
MUP_TIMEOUT = 600
GREEDY_TIMEOUT = 1200

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

D_MIN = 5
D_MAX = 10
D_STEP = 2
YLIM = (1e-2, 1e2)


# ---- data preparation -------------------------------------------------------
def load_prepared(config):
    """Load the CSV and drop rows with NaN in any of the chosen columns once, so
    n is fixed across every (d, lambda)."""
    path = os.path.join(DATASETS_DIR, config["csv"])
    df = pd.read_csv(path, encoding=config["encoding"])
    df = df.dropna(subset=config["ordered_cols"]).reset_index(drop=True)
    return df


def project(df, cols):

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
    pts = sorted(
        (r["d"], r[ykey]) for r in rows
        if r["lambda"] == lam and r["status"] == "ok" and r[ykey] is not None
    )
    return [d for d, _ in pts], [y for _, y in pts]


def plot_figure17(rows, name):

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = run_for_dataset(config)
    plot_figure17(rows, config["name"])


def main():
    for config in DATASET_CONFIGS:
        generate(config)


if __name__ == "__main__":
    main()
