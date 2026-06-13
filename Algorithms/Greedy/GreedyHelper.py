from itertools import combinations, product
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from Algorithms.Mups.MutualFuncs import X, Pattern, level


Domains = List[List[Any]]
ValidationOracle = Callable[[Pattern], bool]


def always_valid(pattern: Pattern) -> bool:
    # Default oracle: every value combination is acceptable.
    # Replace this with a domain-constraint checker to forbid impossible tuples.
    return True


def uncovered_patterns_at_level(
    mups: Iterable[Pattern],
    domains: Domains,
    target_level: int,
) -> List[Pattern]:
    """Compute M_lambda: the set of level-lambda patterns implied by the MUPs.

    Each MUP is an uncovered region.  To collect data that will cover it we
    need concrete, fully-or-partially-specified tuples at level `target_level`
    (lambda in the paper).  A MUP at level k < lambda implies multiple
    level-lambda patterns: all the ways to pin (lambda - k) of its free
    attributes to specific domain values.

    MUPs already at or above `target_level` are skipped — they are more
    specific than lambda and their uncoverage manifests at a finer grain than
    we are targeting.

    Result: the de-duplicated union of all implied level-lambda patterns across
    all MUPs.  This set is the input to greedy_coverage_enhancement().
    """
    d = len(domains)

    if target_level < 0 or target_level > d:
        raise ValueError("target_level must be between 0 and len(domains)")

    result: Set[Pattern] = set()

    for mup in mups:
        mup = tuple(mup)

        if len(mup) != d:
            raise ValueError("Every MUP must have the same length as domains")

        current_level = level(mup)

        # Skip MUPs that are already more specific than the target level — their
        # implied patterns would be below lambda and outside our scope.
        if current_level > target_level:
            continue

        missing = target_level - current_level
        x_positions = [i for i, value in enumerate(mup) if value is X]

        # Choose which `missing` free positions to pin, then enumerate all
        # combinations of values for those positions.
        for chosen_positions in combinations(x_positions, missing):
            chosen_domains = [domains[i] for i in chosen_positions]

            for chosen_values in product(*chosen_domains):
                pattern = list(mup)

                for i, value in zip(chosen_positions, chosen_values):
                    pattern[i] = value

                result.add(tuple(pattern))

    return list(result)


def _build_inverted_indices(
    patterns_to_hit: List[Pattern],
    domains: Domains,
) -> List[Dict[Any, int]]:
    """Build a per-attribute inverted index over the patterns-to-hit list.

    index[i][v] is a bitmask where bit j is set iff patterns_to_hit[j] is
    satisfied by value v at attribute i — either because pattern[i] == v or
    because pattern[i] is X (wildcard matches any value).

    This structure lets _best_value_combination compute, in O(1) per attribute
    assignment, how many remaining patterns a partial tuple hits — by ANDing
    bitmasks across attributes.
    """
    index: List[Dict[Any, int]] = []

    for i, domain in enumerate(domains):
        attr_index: Dict[Any, int] = {}

        for value in domain:
            mask = 0

            for pattern_id, pattern in enumerate(patterns_to_hit):
                # X in a pattern means "any value is acceptable here."
                if pattern[i] is X or pattern[i] == value:
                    mask |= 1 << pattern_id

            attr_index[value] = mask

        index.append(attr_index)

    return index


def _patterns_from_mask(patterns: List[Pattern], mask: int) -> List[Pattern]:
    """Extract the patterns corresponding to set bits in `mask`."""
    # Uses the isolate-lowest-bit trick (mask & -mask) to iterate over set bits
    # without scanning every position — O(popcount) instead of O(len(patterns)).
    result: List[Pattern] = []

    while mask:
        lowest_bit = mask & -mask
        pattern_id = lowest_bit.bit_length() - 1
        result.append(patterns[pattern_id])
        mask ^= lowest_bit

    return result


def _generalize_value_combination(
    value_combination: Pattern,
    hit_patterns: List[Pattern],
) -> Pattern:
    """Generalize a fully-pinned tuple by restoring X for non-essential attributes.

    After the greedy step picks `value_combination`, some of its pinned
    attributes may be irrelevant to the patterns it actually hit — i.e., every
    hit pattern had X at that position (they don't care about that attribute).
    Replacing those values with X makes the output tuple more general: it still
    covers the same set of dataset rows (and therefore still "hits" the same
    uncovered patterns) but is less restrictive, which is preferable when the
    suggested tuple will be collected / added to the dataset.
    """
    generalized = []

    for i, value in enumerate(value_combination):
        # Keep the value only if at least one hit pattern cares about attribute i.
        needed = any(pattern[i] is not X for pattern in hit_patterns)
        generalized.append(value if needed else X)

    return tuple(generalized)


def _best_value_combination(
    remaining_mask: int,
    domains: Domains,
    inverted_indices: List[Dict[Any, int]],
    validation_oracle: ValidationOracle,
) -> Tuple[int, Pattern, int]:
    """Find the value combination that hits the most remaining uncovered patterns.

    This is the core of the greedy set-cover step.  A brute-force search over
    all |D_1| x |D_2| x ... x |D_d| combinations would be too slow.  Instead
    we use a branch-and-bound DFS over attributes:

    - At each attribute we try all domain values, compute the partial hit-mask
      (patterns satisfied by the values chosen so far), and recurse.
    - Pruning: if the number of patterns that *could* still be hit by this
      branch (= popcount of child_mask) is no greater than the current best,
      the branch cannot improve on the best and is cut.
    - Ordering: candidates at each attribute are sorted by descending hit-count,
      so the best branch is explored first, making the pruning bound tight early.

    The validation oracle is consulted before expanding a branch; invalid partial
    tuples are skipped entirely (useful for real-world constraint enforcement).
    """
    d = len(domains)

    if d == 0:
        raise ValueError("domains must contain at least one attribute")

    best_count = 0
    best_combination: Optional[Pattern] = None
    best_hit_mask = 0

    # `current` is built in-place as we recurse; avoids allocation per call.
    current: List[Any] = [X] * d

    def dfs(attr_index: int, current_mask: int) -> None:
        nonlocal best_count, best_combination, best_hit_mask

        candidates = []

        for domain_order, value in enumerate(domains[attr_index]):
            current[attr_index] = value
            partial_pattern = tuple(current)

            # Validation oracle is called before expanding this branch.
            if not validation_oracle(partial_pattern):
                current[attr_index] = X
                continue

            # AND the current hit-mask with the precomputed mask for (attr, value).
            child_mask = current_mask & inverted_indices[attr_index].get(value, 0)
            child_count = child_mask.bit_count()

            if child_count > 0:
                candidates.append((child_count, domain_order, value, child_mask))

            current[attr_index] = X

        # Stronger branches first, so pruning becomes more effective.
        candidates.sort(key=lambda item: (-item[0], item[1]))

        for child_count, _, value, child_mask in candidates:
            # No descendant can hit more patterns than this partial branch.
            if child_count <= best_count:
                break

            current[attr_index] = value

            if attr_index == d - 1:
                # Leaf: this is a fully-specified combination — update the best.
                best_count = child_count
                best_combination = tuple(current)
                best_hit_mask = child_mask
            else:
                dfs(attr_index + 1, child_mask)

            current[attr_index] = X

    dfs(0, remaining_mask)

    if best_combination is None:
        raise ValueError(
            "No valid value combination can hit the remaining uncovered patterns. "
            "Check your domains or validation oracle."
        )

    return best_count, best_combination, best_hit_mask
