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
    """Return one canonical "anchor" value per attribute.

    The bottom-up algorithm uses Rule 2 (the paper's Theorem 4) to generate
    exactly one parent per node: the parent is formed by replacing a
    deterministic value with X, but ONLY when that value equals the anchor for
    its column.  This mirrors Rule 1 from the top-down direction — both rules
    impose a canonical parent-child relationship so each node is processed once.

    Anchor choice: prefer 0 (common in binary / integer attributes) so that
    the canonical parent of (0, v, ...) is (X, v, ...) regardless of domain
    ordering.  Falls back to domain[0] when 0 is absent.
    """
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
    """Generate Rule-2 parents of `pattern`.

    Rule 2: replace a deterministic value at position i with X, but only
    when i is strictly to the RIGHT of the rightmost X in the pattern AND
    the value equals anchors[i].

    This constraint makes the relationship injective: every pattern is
    reachable via Rule 2 from exactly one parent (the one with the rightmost
    anchor value removed), which is the bottom-up dual of Rule 1.
    """
    # start is one position right of the rightmost X in the pattern.
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
    """Compute coverage of `pattern` by summing already-known child counts.

    In the bottom-up pass the children of `pattern` (the level below) have
    already been counted.  The coverage of the parent equals the sum of its
    children's counts, because the children partition the rows that match the
    parent (each row's value at split_attr routes it into exactly one child).

    Returns None when:
    - a child's count is not yet known (child was covered → not in child_counts)
    - the running total already reaches tau (pattern is covered → not a MUP)

    Returning None for "covered" avoids storing covered patterns in next_count,
    which keeps the working set small (only uncovered patterns are tracked).
    """
    # The split attribute is the rightmost X — that is exactly the attribute
    # that distinguishes the children of this pattern from one another.
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
            # Child is covered (not in the uncovered working set) → parent is
            # also covered (a covered child contributes >= tau rows by itself, or
            # the parent just inherits coverage from it).
            return None

        total += child_count

        if total >= tau:
            # Already at threshold; parent is covered — stop early.
            return None

    child[split_attr] = X
    return total


def _has_no_uncovered_parent(
    pattern: Pattern,
    next_count: Dict[Pattern, int],
) -> bool:
    """True iff none of pattern's parents appear in the next (more-general) level.

    A pattern is a MUP when it is uncovered AND every parent is covered.
    After the bottom-up sweep, "every parent is covered" is equivalent to
    "no parent is in the uncovered working set (next_count)."
    """
    return all(parent not in next_count for parent in parents(pattern))


def pattern_combiner(
    dataset: Dataset,
    domains: Domains,
    tau: int,
) -> Set[Pattern]:
    """Bottom-up MUP discovery (BottomUp / PatternCombiner algorithm).

    Algorithm outline:
    1. Start at the leaf level (all attributes pinned): count exact-match rows.
       Only leaves with count < tau enter the initial working set.
    2. Each iteration sweeps the current uncovered set to propose parent
       candidates via Rule 2 (single canonical parent per node).
    3. A parent candidate's coverage is computed by summing its children's counts
       (partitioning trick — no oracle scan needed at higher levels).
    4. Parents with coverage >= tau are covered and are NOT added to the next
       working set (they cannot be MUPs).
    5. Any node in the current set with no uncovered parent is a MUP.
    6. Repeat until the working set is empty.

    Key efficiency:
    - No dataset scan after the initial leaf-level count (uses partitioning).
    - Rule 2 ensures each node is generated once (no deduplication needed).
    - Early exit once the working set empties (often before reaching the root).
    """
    _validate_inputs(dataset, domains)

    d = len(domains)

    if tau <= 0:
        return set()

    # Degenerate zero-attribute case: the root/leaf is the empty pattern.
    if d == 0:
        return {()} if len(dataset) < tau else set()

    anchors = _anchor_values(domains)

    # Count exact (leaf-level) row occurrences once — the only dataset scan.
    row_counts = Counter(tuple(row) for row in dataset)

    current_count: Dict[Pattern, int] = {}

    # Seed the working set with uncovered leaf patterns.
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
