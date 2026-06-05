from itertools import product
from typing import List, Any, Set, Dict

from .MutualFuncs import X, Pattern, Dataset, coverage, parents


def full_patterns(domains: List[List[Any]]) -> List[Pattern]:
    """
    Generate all fully deterministic patterns at level d.

    Example for 3 binary attributes:
    000, 001, 010, 011, 100, 101, 110, 111
    """
    return [tuple(values) for values in product(*domains)]


def rightmost_x_index(pattern: Pattern) -> int:
    """
    Return the index of the right-most X.
    If there is no X, return -1.

    For a full pattern like 010, this returns -1.
    """
    for i in range(len(pattern) - 1, -1, -1):
        if pattern[i] is X:
            return i

    return -1


def generate_rule2_parents(
    pattern: Pattern,
    zero_values: List[Any]
) -> List[Pattern]:
    """
    Rule 2 from the paper:

    From an uncovered pattern P, generate candidate parents by replacing
    deterministic cells with the mapped-zero value, on the right side of
    the right-most X, with X.

    For binary data, zero_values is usually [0, 0, 0, ...].
    For string/categorical data, we choose one value in each domain to act
    like the paper's '0'.
    """
    result = []

    rx = rightmost_x_index(pattern)

    for i in range(rx + 1, len(pattern)):
        if pattern[i] is not X and pattern[i] == zero_values[i]:
            parent = list(pattern)
            parent[i] = X
            result.append(tuple(parent))

    return result


def partition_children_for_parent(
    parent: Pattern,
    domains: List[List[Any]]
) -> List[Pattern]:
    """
    To compute coverage of a parent P', the paper uses a disjoint partition
    of its children.

    Find the right-most X in P', replace it with every possible value
    of that attribute, and return those children.

    Example:
    parent = 1X0
    domains[1] = [0, 1]

    children:
    100
    110
    """
    i = rightmost_x_index(parent)

    if i == -1:
        return []

    result = []

    for value in domains[i]:
        child = list(parent)
        child[i] = value
        result.append(tuple(child))

    return result


def pattern_combiner(
    dataset: Dataset,
    domains: List[List[Any]],
    tau: int,
    zero_values: List[Any] = None
) -> Set[Pattern]:
    """
    Bottom-up PATTERN-COMBINER algorithm.

    Input:
        dataset - list of tuples
        domains - possible values for each attribute
        tau - coverage threshold
        zero_values - optional mapped-zero value for each attribute

    Output:
        Set of maximal uncovered patterns, MUPs.
    """

    if zero_values is None:
        zero_values = [domain[0] for domain in domains]

    d = len(domains)
    mups: Set[Pattern] = set()

    # Step 1:
    # Start from level d, the fully deterministic patterns.
    # Keep only uncovered patterns in count.
    count: Dict[Pattern, int] = {}

    for pattern in full_patterns(domains):
        cnt = coverage(pattern, dataset)

        if cnt < tau:
            count[pattern] = cnt

    if not count:
        return set()

    current_level = d

    # Step 2:
    # Move upward while there are uncovered candidates.
    while count:
        # If the root itself is uncovered, it is a MUP.
        if current_level == 0:
            mups.update(count.keys())
            break

        next_count: Dict[Pattern, int] = {}

        # Generate uncovered candidates at level current_level - 1
        for pattern in list(count.keys()):
            candidate_parents = generate_rule2_parents(pattern, zero_values)

            for parent in candidate_parents:
                partition_children = partition_children_for_parent(parent, domains)

                total = 0

                for child in partition_children:
                    if child in count:
                        total += count[child]
                    else:
                        # If child is not in count, it is covered.
                        # We only need to know whether total reaches tau.
                        total += tau

                    if total >= tau:
                        break

                if total < tau:
                    next_count[parent] = total

        # Step 3:
        # A pattern in count is a MUP if none of its parents are uncovered.
        next_keys = set(next_count.keys())

        for pattern in count.keys():
            has_uncovered_parent = any(
                parent in next_keys
                for parent in parents(pattern)
            )

            if not has_uncovered_parent:
                mups.add(pattern)

        count = next_count
        current_level -= 1

    return mups
