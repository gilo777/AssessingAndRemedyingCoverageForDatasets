from itertools import product
from typing import List, Tuple, Any, Set
from .MutualFuncs import X, Pattern, Dataset, dominated_by_any_mup, dominates_any_mup, is_uncovered, children, parents


def find_uncovered_parent(pattern: Pattern, dataset: Dataset, tau: int):
    """
    Return one uncovered parent if it exists.
    Otherwise, return None.
    """
    for parent in parents(pattern):
        if is_uncovered(parent, dataset, tau):
            return parent

    return None


def deepdiver(dataset: Dataset, domains: List[List[Any]], tau: int) -> Set[Pattern]:
    """
    DeepDiver MUP search.
    Goes down until it reaches uncovered region,
    then goes up until it finds a MUP.
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

        # If pattern is already pruned by known MUPs, skip it.
        if dominated_by_any_mup(pattern, mups):
            continue

        # If pattern dominates a known MUP, it cannot be a new MUP.
        if dominates_any_mup(pattern, mups):
            continue

        if is_uncovered(pattern, dataset, tau):
            # Climb up until no uncovered parent exists.
            current = pattern

            while True:
                uncovered_parent = find_uncovered_parent(current, dataset, tau)

                if uncovered_parent is None:
                    break

                current = uncovered_parent

            mups.add(current)

        else:
            # Covered pattern: dive deeper.
            for child in children(pattern, domains):
                if child not in visited:
                    stack.append(child)

    return mups