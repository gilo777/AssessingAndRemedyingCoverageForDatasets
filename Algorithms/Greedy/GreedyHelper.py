from itertools import combinations, product
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from Algorithms.Mups.MutualFuncs import X, Pattern, level


Domains = List[List[Any]]
ValidationOracle = Callable[[Pattern], bool]


def always_valid(pattern: Pattern) -> bool:

    return True


def uncovered_patterns_at_level(
    mups: Iterable[Pattern],
    domains: Domains,
    target_level: int,
) -> List[Pattern]:

    d = len(domains)

    if target_level < 0 or target_level > d:
        raise ValueError("target_level must be between 0 and len(domains)")

    result: Set[Pattern] = set()

    for mup in mups:
        mup = tuple(mup)

        if len(mup) != d:
            raise ValueError("Every MUP must have the same length as domains")

        current_level = level(mup)

        if current_level > target_level:
            continue

        missing = target_level - current_level
        x_positions = [i for i, value in enumerate(mup) if value is X]

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

    index: List[Dict[Any, int]] = []

    for i, domain in enumerate(domains):
        attr_index: Dict[Any, int] = {}

        for value in domain:
            mask = 0

            for pattern_id, pattern in enumerate(patterns_to_hit):
                if pattern[i] is X or pattern[i] == value:
                    mask |= 1 << pattern_id

            attr_index[value] = mask

        index.append(attr_index)

    return index


def _patterns_from_mask(patterns: List[Pattern], mask: int) -> List[Pattern]:

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

    generalized = []

    for i, value in enumerate(value_combination):
        needed = any(pattern[i] is not X for pattern in hit_patterns)
        generalized.append(value if needed else X)

    return tuple(generalized)


def _best_value_combination(
    remaining_mask: int,
    domains: Domains,
    inverted_indices: List[Dict[Any, int]],
    validation_oracle: ValidationOracle,
) -> Tuple[int, Pattern, int]:

    d = len(domains)

    if d == 0:
        raise ValueError("domains must contain at least one attribute")

    best_count = 0
    best_combination: Optional[Pattern] = None
    best_hit_mask = 0

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
