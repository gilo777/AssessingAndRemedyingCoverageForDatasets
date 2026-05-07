from itertools import product
from typing import List, Tuple, Any, Set

X = None  # represents "X" in the paper


Pattern = Tuple[Any, ...]
Dataset = List[Tuple[Any, ...]]


def coverage(pattern: Pattern, dataset: Dataset) -> int:
    """
    Count how many rows match the pattern.
    X/None means the attribute is unspecified.
    """
    count = 0

    for row in dataset:
        match = True

        for p_val, r_val in zip(pattern, row):
            if p_val is not X and p_val != r_val:
                match = False
                break

        if match:
            count += 1

    return count


def is_uncovered(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    return coverage(pattern, dataset) < tau


def level(pattern: Pattern) -> int:
    """
    Number of deterministic cells.
    Example: (X, 1, X, 0) has level 2.
    """
    return sum(1 for value in pattern if value is not X)


def parents(pattern: Pattern) -> List[Pattern]:
    """
    A parent is created by replacing one deterministic value with X.
    """
    result = []

    for i, value in enumerate(pattern):
        if value is not X:
            parent = list(pattern)
            parent[i] = X
            result.append(tuple(parent))

    return result


def children(pattern: Pattern, domains: List[List[Any]]) -> List[Pattern]:
    """
    A child is created by replacing one X with a possible attribute value.
    """
    result = []

    for i, value in enumerate(pattern):
        if value is X:
            for attr_value in domains[i]:
                child = list(pattern)
                child[i] = attr_value
                result.append(tuple(child))

    return result


def is_parent_covered_mup(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    """
    A pattern is MUP if:
    1. it is uncovered
    2. all its parents are covered
    """
    if not is_uncovered(pattern, dataset, tau):
        return False

    for parent in parents(pattern):
        if is_uncovered(parent, dataset, tau):
            return False

    return True


def dominates(general: Pattern, specific: Pattern) -> bool:
    """
    general dominates specific if every deterministic value in general
    agrees with specific.

    Example:
    (1, X, X) dominates (1, 0, 1)
    """
    for g_val, s_val in zip(general, specific):
        if g_val is not X and g_val != s_val:
            return False

    return True


def dominated_by_any_mup(pattern: Pattern, mups: Set[Pattern]) -> bool:
    return any(dominates(mup, pattern) for mup in mups)


def dominates_any_mup(pattern: Pattern, mups: Set[Pattern]) -> bool:
    return any(dominates(pattern, mup) for mup in mups)