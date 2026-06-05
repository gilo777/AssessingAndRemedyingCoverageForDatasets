from typing import List, Tuple, Any

# Sentinel for a non-deterministic ("X") cell in a pattern (paper Definition 1).
X = None

Pattern = Tuple[Any, ...]
TupleRow = Tuple[Any, ...]
Dataset = List[TupleRow]


def matches_tuple(row: TupleRow, pattern: Pattern) -> bool:
    """
    Check if a full tuple matches a pattern.
    Example:
    row     = (1, 0, 1)
    pattern = (1, None, 1)
    result  = True
    """
    for r_val, p_val in zip(row, pattern):
        if p_val is not X and r_val != p_val:
            return False

    return True


def coverage(pattern: Pattern, dataset: Dataset) -> int:
    """
    Count how many rows in the dataset match this pattern.
    """
    return sum(1 for row in dataset if matches_tuple(row, pattern))


def pattern_level(pattern: Pattern) -> int:
    """
    Number of deterministic values in the pattern.
    Example:
    (1, None, 0) has level 2.
    """
    return sum(1 for value in pattern if value is not X)


def all_patterns_at_level(domains: List[List[Any]], target_level: int) -> List[Pattern]:
    """
    Generate all patterns with exactly target_level deterministic cells.
    """

    d = len(domains)
    result = []

    def backtrack(index: int, current: List[Any], deterministic_count: int):
        if index == d:
            if deterministic_count == target_level:
                result.append(tuple(current))
            return

        # Option 1: put X
        current.append(X)
        backtrack(index + 1, current, deterministic_count)
        current.pop()

        # Option 2: put an actual value
        for value in domains[index]:
            current.append(value)
            backtrack(index + 1, current, deterministic_count + 1)
            current.pop()

    backtrack(0, [], 0)
    return result