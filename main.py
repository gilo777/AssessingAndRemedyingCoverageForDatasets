import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DATASETS_DIR = os.path.join(PROJECT_ROOT, "Datasets")

from Figure10.GenerateFigure10 import generate as generate_figure10
from Figure10.MupConstants import MUP_CONFIGS
from Figure16.Graph16 import generate as generate_figure16

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Each entry: experiment number -> (short description, runner function).
# The runner takes (csv_path, dataset_name). Replace the stub bodies later.

def run_experiment_10(csv_path, dataset_name):
    """Effect of lack of coverage: generate the dataset's Figure-10 graph using
    its hardcoded MUP (see Figure10/MupConstants.py). The PNG is written to
    Graphs/Figure-10/."""
    match = next(
        ((name, cfg) for name, cfg in MUP_CONFIGS.items()
         if cfg["csv"] == dataset_name),
        None,
    )
    if match is None:
        print(f"  Experiment 10 skipped: no hardcoded Figure-10 MUP for "
              f"'{dataset_name}'. Configured: "
              f"{[cfg['csv'] for cfg in MUP_CONFIGS.values()]}")
        return

    name, cfg = match
    generate_figure10(name, cfg)


def run_experiment_11(csv_path, dataset_name):
    from Experiments.experiment_11 import run_experiment
    return run_experiment(csv_path=csv_path, dataset_name=dataset_name)


def run_experiment_16(csv_path, dataset_name):
    """Coverage enhancement runtime, varying the threshold."""
    return generate_figure16(csv_path, dataset_name)


def run_experiment_17(csv_path, dataset_name):
    """Coverage enhancement runtime, varying the number of dimensions: generate
    the dataset's Figure-17 graph. The PNG is written to Graphs/Figure-17/."""
    from Figure17.GenerateFigure17 import DATASET_CONFIGS, generate

    match = next(
        (cfg for cfg in DATASET_CONFIGS if cfg["csv"] == dataset_name),
        None,
    )
    if match is None:
        print(f"  Experiment 17 skipped: no Figure-17 config for "
              f"'{dataset_name}'. Configured: "
              f"{[cfg['csv'] for cfg in DATASET_CONFIGS]}")
        return

    generate(match)


EXPERIMENTS = {
    10: ("Effect of lack of coverage on classification", run_experiment_10),
    11: ("MUP identification - varying threshold", run_experiment_11),
    16: ("Coverage enhancement - varying threshold", run_experiment_16),
    17: ("Coverage enhancement - varying dimensions", run_experiment_17),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_datasets(datasets_dir):
    if not os.path.isdir(datasets_dir):
        return []
    return sorted(
        name for name in os.listdir(datasets_dir)
        if name.lower().endswith(".csv")
    )


def parse_selection(raw, valid_values):
    raw = raw.strip().lower()

    if raw in ("", "all"):
        return list(valid_values)

    tokens = raw.replace(",", " ").split()
    selected = []

    for token in tokens:
        if not token.lstrip("-").isdigit():
            raise ValueError(f"'{token}' is not a number")

        number = int(token)

        if number not in valid_values:
            raise ValueError(f"{number} is not one of the available choices")

        if number not in selected:
            selected.append(number)

    return selected


def prompt_until_valid(prompt, valid_values):
    """Keep asking until the user gives a parseable, valid selection."""
    while True:
        raw = input(prompt)
        try:
            return parse_selection(raw, valid_values)
        except ValueError as error:
            print(f"  Invalid input: {error}. Try again.\n")


# ---------------------------------------------------------------------------
# Menu steps
# ---------------------------------------------------------------------------

def choose_datasets(datasets):
    print("Available datasets:")
    for index, name in enumerate(datasets, start=1):
        print(f"  {index}. {name}")
    print()

    valid_indices = range(1, len(datasets) + 1)
    chosen_indices = prompt_until_valid(
        "Pick dataset number(s) (blank for all): ",
        valid_indices,
    )

    return [datasets[i - 1] for i in chosen_indices]


def choose_experiments():
    print("\nAvailable experiments:")
    for number, (description, _) in EXPERIMENTS.items():
        print(f"  {number}. {description}")
    print()

    return prompt_until_valid(
        "Pick experiment number(s) (blank for all): ",
        list(EXPERIMENTS.keys()),
    )


def run(chosen_datasets, chosen_experiments):
    print("\nRunning...\n")

    for dataset_name in chosen_datasets:
        csv_path = os.path.join(DATASETS_DIR, dataset_name)
        print(f"Dataset: {dataset_name}")

        for number in chosen_experiments:
            _, runner = EXPERIMENTS[number]
            runner(csv_path, dataset_name)

        print()

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    datasets = list_datasets(DATASETS_DIR)

    if not datasets:
        print(f"No CSV datasets found in: {DATASETS_DIR}")
        print("Add some .csv files there and run again.")
        return

    chosen_datasets = choose_datasets(datasets)
    chosen_experiments = choose_experiments()
    run(chosen_datasets, chosen_experiments)


if __name__ == "__main__":
    main()
