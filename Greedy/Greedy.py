from itertools import product
from typing import Any, Callable, Dict, List

from .GreedyHelper import Dataset, Pattern, TupleRow, all_patterns_at_level, coverage, matches_tuple


def greedy_coverage_enhancement(
    dataset: Dataset,
    domains: List[List[Any]],
    tau: int,
    max_level: int,
    validation_oracle: Callable[[TupleRow], bool] = None
) -> List[TupleRow]:
    """
    Greedy algorithm for the Coverage Enhancement Problem.

    Goal:
    Add new tuples so that every pattern at level max_level
    has coverage at least tau.

    Parameters:
    dataset:
        The original dataset.

    domains:
        Possible values for each attribute.
        Example:
        [[0, 1], [0, 1], [0, 1]]

    tau:
        Coverage threshold.

    max_level:
        The level lambda from the paper.
        We want all patterns at this level to become covered.

    validation_oracle:
        Optional function that returns True if a tuple is valid.
        If None, every tuple is considered valid.

    Returns:
        A list of tuples to add to the dataset.
    """

    if validation_oracle is None:
        validation_oracle = lambda row: True

    # Step 1: Find all uncovered patterns at level max_level.
    patterns = all_patterns_at_level(domains, max_level)

    uncovered_deficit: Dict[Pattern, int] = {}

    for pattern in patterns:
        cov = coverage(pattern, dataset)

        if cov < tau:
            uncovered_deficit[pattern] = tau - cov

    # If there is nothing to fix
    if not uncovered_deficit:
        return []

    # Step 2: Generate all possible full tuples.
    possible_tuples = list(product(*domains))

    # Keep only valid tuples.
    possible_tuples = [
        row for row in possible_tuples
        if validation_oracle(row)
    ]

    added_tuples = []

    # Step 3: Greedily choose tuples until all deficits are fixed.
    while uncovered_deficit:
        best_tuple = None
        best_hit_count = 0

        for candidate in possible_tuples:
            hit_count = 0

            for pattern, deficit in uncovered_deficit.items():
                if deficit > 0 and matches_tuple(candidate, pattern):
                    hit_count += 1

            if hit_count > best_hit_count:
                best_hit_count = hit_count
                best_tuple = candidate

        if best_tuple is None or best_hit_count == 0:
            raise ValueError(
                "Could not cover all patterns. "
                "Maybe the validation oracle blocks all useful tuples."
            )

        # Add the best tuple
        added_tuples.append(best_tuple)

        # Update deficits
        new_deficit = {}

        for pattern, deficit in uncovered_deficit.items():
            if matches_tuple(best_tuple, pattern):
                deficit -= 1

            if deficit > 0:
                new_deficit[pattern] = deficit

        uncovered_deficit = new_deficit

    return added_tuples