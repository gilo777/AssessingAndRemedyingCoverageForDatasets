from itertools import product
from typing import List, Tuple, Any, Set
from .MutualFuncs import X, Pattern, Dataset, dominated_by_any_mup, is_parent_covered_mup, is_uncovered, children


def pattern_breaker(dataset: Dataset, domains: List[List[Any]], tau: int) -> Set[Pattern]:
    """
    Correct top-down MUP search.
    Starts from XXX...X and explores downward.
    A pattern is added only if it is truly a MUP:
    uncovered, and all parents are covered.
    """

    d = len(domains)
    root = tuple([X] * d)

    mups = set()
    stack = [root]
    visited = set()

    while stack:
        pattern = stack.pop()

        if pattern in visited:
            continue

        visited.add(pattern)

        if dominated_by_any_mup(pattern, mups):
            continue

        if is_uncovered(pattern, dataset, tau):
            # Important fix:
            # do NOT automatically add it.
            # First check that all parents are covered.
            if is_parent_covered_mup(pattern, dataset, tau):
                mups.add(pattern)

            # Do not go deeper from an uncovered pattern.
            continue

        # Pattern is covered, so its children may contain MUPs.
        for child in children(pattern, domains):
            if child not in visited:
                stack.append(child)

    return mups