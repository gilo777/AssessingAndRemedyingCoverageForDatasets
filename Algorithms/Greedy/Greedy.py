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

    patterns = list(dict.fromkeys(tuple(pattern) for pattern in patterns_to_hit))

    if not patterns:
        return []

    d = len(domains)

    for pattern in patterns:
        if len(pattern) != d:
            raise ValueError("Every pattern must have the same length as domains")

    inverted_indices = _build_inverted_indices(patterns, domains)

    remaining_mask = (1 << len(patterns)) - 1
    selected: List[Pattern] = []

    while remaining_mask:
        _, value_combination, hit_mask = _best_value_combination(
            remaining_mask=remaining_mask,
            domains=domains,
            inverted_indices=inverted_indices,
            validation_oracle=validation_oracle,
        )

        hit_now = remaining_mask & hit_mask

        if hit_now == 0:
            raise ValueError(
                "The selected value combination did not hit any remaining "
                "patterns. Check the domains or validation oracle."
            )

        if generalize_output:
            hit_patterns = _patterns_from_mask(patterns, hit_now)
            selected.append(
                _generalize_value_combination(
                    value_combination,
                    hit_patterns,
                )
            )
        else:
            selected.append(value_combination)

        remaining_mask &= ~hit_now

    return selected


def greedy_coverage_enhancement_from_mups(
    mups: Iterable[Pattern],
    domains: Domains,
    target_level: int,
    validation_oracle: ValidationOracle = always_valid,
    generalize_output: bool = False,
) -> List[Pattern]:

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
