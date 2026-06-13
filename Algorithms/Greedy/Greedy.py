from typing import Iterable, List

from Algorithms.Mups.MutualFuncs import Pattern

from .GreedyHelper import (
    Domains,
    ValidationOracle,
    always_valid,
    uncovered_patterns_at_level,
    _build_inverted_indices,
    _best_value_combination,
    _patterns_from_mask,
    _generalize_value_combination,
)


def greedy_coverage_enhancement(
    patterns_to_hit: Iterable[Pattern],
    domains: Domains,
    validation_oracle: ValidationOracle = always_valid,
    generalize_output: bool = False,
) -> List[Pattern]:
    """Greedy set-cover algorithm for coverage enhancement.

    Goal: given a set of uncovered patterns (M_lambda), return the smallest
    list of value combinations (tuples to add to the dataset) such that every
    pattern in M_lambda is "hit" by at least one returned tuple.

    This is a weighted set-cover problem, which is NP-hard in general.  The
    greedy (1 - 1/e)-approximation repeatedly picks the tuple that covers the
    most currently-uncovered patterns.

    Implementation uses bit-parallel set arithmetic:
    - `remaining_mask` is an integer bitmask: bit j is set iff patterns[j] has
      not yet been hit.
    - _best_value_combination() finds the best tuple via branch-and-bound DFS
      over attribute assignments, intersecting bitmasks at each step.
    - After picking a tuple we clear the bits it hits from remaining_mask with
      a bitwise AND of the complement.

    Parameters:
        patterns_to_hit   — the M_lambda set (output of uncovered_patterns_at_level)
        domains           — possible values per attribute
        validation_oracle — optional constraint: returns False for forbidden tuples
        generalize_output — if True, redundant attributes are replaced with X
                            in the returned tuples (smaller, more readable output)
    """
    # De-duplicate while preserving order (dict.fromkeys trick).
    patterns = list(dict.fromkeys(tuple(pattern) for pattern in patterns_to_hit))

    if not patterns:
        return []

    d = len(domains)

    for pattern in patterns:
        if len(pattern) != d:
            raise ValueError("Every pattern must have the same length as domains")

    # Build once; reused in every iteration of the greedy loop.
    inverted_indices = _build_inverted_indices(patterns, domains)

    # All patterns start uncovered: every bit set.
    remaining_mask = (1 << len(patterns)) - 1
    selected: List[Pattern] = []

    while remaining_mask:
        # Pick the value combination that hits the most remaining patterns.
        _, value_combination, hit_mask = _best_value_combination(
            remaining_mask=remaining_mask,
            domains=domains,
            inverted_indices=inverted_indices,
            validation_oracle=validation_oracle,
        )

        # Intersect with remaining_mask to count only newly-hit patterns.
        hit_now = remaining_mask & hit_mask

        if hit_now == 0:
            raise ValueError(
                "The selected value combination did not hit any remaining "
                "patterns. Check the domains or validation oracle."
            )

        if generalize_output:
            # Replace non-essential attributes with X to produce a more general
            # suggestion (see _generalize_value_combination for rationale).
            hit_patterns = _patterns_from_mask(patterns, hit_now)
            selected.append(
                _generalize_value_combination(
                    value_combination,
                    hit_patterns,
                )
            )
        else:
            selected.append(value_combination)

        # Clear the bits for patterns now covered — they won't be re-hit.
        remaining_mask &= ~hit_now

    return selected


def greedy_coverage_enhancement_from_mups(
    mups: Iterable[Pattern],
    domains: Domains,
    target_level: int,
    validation_oracle: ValidationOracle = always_valid,
    generalize_output: bool = False,
) -> List[Pattern]:
    """Convenience wrapper: compute M_lambda from MUPs, then run the greedy algorithm.

    This bridges the MUP-discovery step (TopDown / BottomUp / DeepDiver) and
    the coverage-enhancement step in a single call.  target_level (lambda) is
    the specificity of the tuples to suggest — lower lambda means more general
    suggestions, higher lambda means more targeted ones.
    """
    patterns_to_hit = uncovered_patterns_at_level(
        mups=mups,
        domains=domains,
        target_level=target_level,
    )

    return greedy_coverage_enhancement(
        patterns_to_hit=patterns_to_hit,
        domains=domains,
        validation_oracle=validation_oracle,
        generalize_output=generalize_output,
    )
