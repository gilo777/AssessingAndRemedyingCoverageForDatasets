from itertools import product
from typing import List, Tuple, Any, Set
from .MutualFuncs import X, Pattern, Dataset, dominated_by_any_mup, is_parent_covered_mup


def all_patterns_at_level(domains: List[List[Any]], target_level: int) -> List[Pattern]:
    """
    Generate all patterns with exactly target_level deterministic cells.
    """

    d = len(domains)
    result = []

    def backtrack(index: int, current: List[Any], deterministic_count: int):
        if index == d:
            if deterministic_count == target_level:
                result.append(tuple(current))
            return

        # Put X
        current.append(X)
        backtrack(index + 1, current, deterministic_count)
        current.pop()

        # Put real values
        for value in domains[index]:
            current.append(value)
            backtrack(index + 1, current, deterministic_count + 1)
            current.pop()

    backtrack(0, [], 0)
    return result


def pattern_combiner(dataset: Dataset, domains: List[List[Any]], tau: int) -> Set[Pattern]:
    """
    Bottom-up MUP search.
    Starts from the most specific patterns and moves upward.
    """

    d = len(domains)
    mups = set()
    already_checked = set()

    # Start at level d and move upward to level 0
    for current_level in range(d, -1, -1):
        patterns = all_patterns_at_level(domains, current_level)

        for pattern in patterns:
            if pattern in already_checked:
                continue

            already_checked.add(pattern)

            if dominated_by_any_mup(pattern, mups):
                continue

            if is_parent_covered_mup(pattern, dataset, tau):
                mups.add(pattern)

    return mups