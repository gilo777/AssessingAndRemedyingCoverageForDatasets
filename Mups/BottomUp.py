# ---------------------------------------------------------------------------
# Bottom-up MUP identification: PATTERN-COMBINER
# ---------------------------------------------------------------------------
# This implementation follows the document's Algorithm 2 / Rule 2:
#   * start at level d (fully deterministic patterns),
#   * keep only uncovered patterns together with their exact coverage,
#   * generate candidate parents from uncovered nodes using Rule 2,
#   * compute a candidate parent's coverage by summing the disjoint partition
#     coverages of its children at the previous level,
#   * prune as soon as a candidate parent is covered,
#   * report an uncovered node as a MUP exactly when it has no uncovered parent.
#
# Rule 2 needs one distinguished value per attribute (called value 0 in the
# paper).  If 0 is in a domain, we use 0; otherwise we use the first domain value.
# ---------------------------------------------------------------------------

from collections import Counter
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # support both package imports and direct imports from this folder
    from .MutualFuncs import X, parents
except ImportError:  # pragma: no cover - used when running this file directly
    from MutualFuncs import X, parents

Pattern = Tuple[Any, ...]
Dataset = List[Tuple[Any, ...]]
Domains = List[List[Any]]


def all_patterns_at_level(domains: Domains, target_level: int) -> List[Pattern]:
    """Return all patterns whose number of deterministic cells is target_level."""
    d = len(domains)
    result: List[Pattern] = []

    def backtrack(index: int, current: List[Any], deterministic_count: int) -> None:
        if deterministic_count > target_level:
            return
        if index == d:
            if deterministic_count == target_level:
                result.append(tuple(current))
            return

        current.append(X)
        backtrack(index + 1, current, deterministic_count)
        current.pop()

        for value in domains[index]:
            current.append(value)
            backtrack(index + 1, current, deterministic_count + 1)
            current.pop()

    if 0 <= target_level <= d:
        backtrack(0, [], 0)
    return result


def _anchor_values(domains: Domains) -> List[Any]:
    """
    Pick the per-attribute value that plays the paper's value 0 role in Rule 2.

    The paper notes that Rule 2 is not restricted to binary attributes; all that
    is required is mapping one value of every attribute to 0.  For numeric domains
    that contain 0, using real 0 is the least surprising choice.  For arbitrary
    categorical domains, the first listed value is the fixed anchor.
    """
    anchors: List[Any] = []
    for i, domain in enumerate(domains):
        if not domain:
            raise ValueError(f"domains[{i}] is empty")
        anchors.append(0 if 0 in domain else domain[0])
    return anchors


def _rightmost_x(pattern: Pattern) -> int:
    """Index of the right-most X/None, or -1 when the pattern is a leaf."""
    for i in range(len(pattern) - 1, -1, -1):
        if pattern[i] is X:
            return i
    return -1


def _rule2_parents(pattern: Pattern, anchors: Sequence[Any]) -> Iterable[Pattern]:
    """
    Generate candidate parent nodes according to Rule 2.

    A node P with coverage below tau generates level(P)-1 candidates by replacing
    deterministic anchor-value cells on the right-hand side of its right-most X
    with X.  If P has no X, the right-most X is treated as being before index 0,
    so every anchor-valued cell may be replaced.
    """
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
    """
    Return exact coverage for an uncovered candidate parent, otherwise None.

    For the right-most X in the candidate parent, its children partition the
    matches of the parent.  `child_counts` stores exact counts only for uncovered
    children; a missing child is already covered, so the parent is covered too.
    """
    split_attr = _rightmost_x(pattern)
    if split_attr < 0:
        return child_counts.get(pattern, 0)

    total = 0
    child = list(pattern)
    for value in domains[split_attr]:
        child[split_attr] = value
        child_pattern = tuple(child)
        child_count = child_counts.get(child_pattern)

        if child_count is None:
            # This partition child is covered (coverage >= tau), so the parent
            # must be covered.  We do not need the exact parent coverage.
            return None

        total += child_count
        if total >= tau:
            return None

    child[split_attr] = X
    return total


def _is_mup_from_next_level(pattern: Pattern, next_count: Dict[Pattern, int]) -> bool:
    """An uncovered pattern is maximal iff none of its parents is uncovered."""
    return all(parent not in next_count for parent in parents(pattern))


def pattern_combiner(dataset: Dataset, domains: Domains, tau: int) -> Set[Pattern]:
    """
    Find all maximal uncovered patterns using the paper's bottom-up algorithm.

    `dataset` is a list of full value combinations. `domains[i]` contains the
    possible values of attribute i.  A pattern is uncovered when coverage < tau.
    """
    d = len(domains)
    if tau <= 0:
        return set()

    anchors = _anchor_values(domains)
    row_counts = Counter(tuple(row) for row in dataset)

    # Degenerate zero-attribute case: the root/leaf is the empty pattern.
    if d == 0:
        return {()} if len(dataset) < tau else set()

    # Algorithm 2 lines 1-5: count uncovered leaves exactly once.
    count: Dict[Pattern, int] = {}
    for leaf in product(*domains):
        leaf_count = row_counts.get(tuple(leaf), 0)
        if leaf_count < tau:
            count[tuple(leaf)] = leaf_count

    if not count:
        return set()

    mups: Set[Pattern] = set()
    current_level = d

    while count:
        next_count: Dict[Pattern, int] = {}

        if current_level > 0:
            # Algorithm 2 lines 8-17: combine uncovered nodes to get uncovered
            # candidate parents.  Covered parents are pruned immediately.
            for pattern in count.keys():
                for parent in _rule2_parents(pattern, anchors):
                    if parent in next_count:
                        continue

                    parent_count = _coverage_from_partition_children(
                        parent, domains, count, tau
                    )
                    if parent_count is not None:
                        next_count[parent] = parent_count

        # Algorithm 2 lines 18-22: uncovered nodes with no uncovered parent are MUPs.
        for pattern in count.keys():
            if _is_mup_from_next_level(pattern, next_count):
                mups.add(pattern)

        if not next_count:
            break

        count = next_count
        current_level -= 1

    return mups
