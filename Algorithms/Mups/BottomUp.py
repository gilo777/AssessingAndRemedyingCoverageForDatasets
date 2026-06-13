from collections import Counter
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .MutualFuncs import X, parents
except ImportError:
    from MutualFuncs import X, parents


Pattern = Tuple[Any, ...]
TupleRow = Tuple[Any, ...]
Dataset = List[TupleRow]
Domains = List[List[Any]]


def _validate_inputs(dataset: Dataset, domains: Domains) -> None:

    d = len(domains)

    for i, domain in enumerate(domains):
        if not domain:
            raise ValueError(f"domains[{i}] is empty")

    for row_index, row in enumerate(dataset):
        if len(row) != d:
            raise ValueError(
                f"dataset row {row_index} has length {len(row)}, "
                f"but domains has length {d}"
            )


def _anchor_values(domains: Domains) -> List[Any]:

    anchors: List[Any] = []

    for i, domain in enumerate(domains):
        if not domain:
            raise ValueError(f"domains[{i}] is empty")

        anchors.append(0 if 0 in domain else domain[0])

    return anchors


def _rightmost_x(pattern: Pattern) -> int:
    """Return the index of the right-most X/None, or -1 if there is no X."""
    for i in range(len(pattern) - 1, -1, -1):
        if pattern[i] is X:
            return i

    return -1


def _rule2_parents(pattern: Pattern, anchors: Sequence[Any]) -> Iterable[Pattern]:

    start = _rightmost_x(pattern) + 1

    for i in range(start, len(pattern)):
        if pattern[i] is not X and pattern[i] == anchors[i]:
            parent = list(pattern)
            parent[i] = X
            yield tuple(parent)


def _coverage_from_partition_children(
    pattern: Pattern,
    domains: Domains,
    child_counts: Dict[Pattern, int],
    tau: int,
) -> Optional[int]:

    split_attr = _rightmost_x(pattern)

    # Should not usually happen for a parent candidate, but safe to keep.
    if split_attr < 0:
        return child_counts.get(pattern, 0)

    total = 0
    child = list(pattern)

    for value in domains[split_attr]:
        child[split_attr] = value
        child_pattern = tuple(child)

        child_count = child_counts.get(child_pattern)

        if child_count is None:
            return None

        total += child_count

        if total >= tau:
            return None

    child[split_attr] = X
    return total


def _has_no_uncovered_parent(
    pattern: Pattern,
    next_count: Dict[Pattern, int],
) -> bool:

    return all(parent not in next_count for parent in parents(pattern))


def pattern_combiner(
    dataset: Dataset,
    domains: Domains,
    tau: int,
) -> Set[Pattern]:

    _validate_inputs(dataset, domains)

    d = len(domains)

    if tau <= 0:
        return set()

    # Degenerate zero-attribute case: the root/leaf is the empty pattern.
    if d == 0:
        return {()} if len(dataset) < tau else set()

    anchors = _anchor_values(domains)

    row_counts = Counter(tuple(row) for row in dataset)

    current_count: Dict[Pattern, int] = {}

    for leaf in product(*domains):
        leaf_pattern = tuple(leaf)
        leaf_count = row_counts.get(leaf_pattern, 0)

        if leaf_count < tau:
            current_count[leaf_pattern] = leaf_count

    if not current_count:
        return set()

    mups: Set[Pattern] = set()
    current_level = d

    while current_count:
        next_count: Dict[Pattern, int] = {}

        if current_level > 0:
            # Generate candidate parents from uncovered nodes.
            for pattern in current_count.keys():
                for parent in _rule2_parents(pattern, anchors):
                    if parent in next_count:
                        continue

                    parent_count = _coverage_from_partition_children(
                        parent,
                        domains,
                        current_count,
                        tau,
                    )

                    if parent_count is not None:
                        next_count[parent] = parent_count

        # An uncovered node is a MUP if it has no uncovered parent.
        for pattern in current_count.keys():
            if _has_no_uncovered_parent(pattern, next_count):
                mups.add(pattern)

        if not next_count:
            break

        current_count = next_count
        current_level -= 1

    return mups


# Optional alias, in case your other files expect this name.
bottom_up = pattern_combiner
pattern_combiner_bottom_up = pattern_combiner
