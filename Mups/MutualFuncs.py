from typing import List, Tuple, Any, Set

# In the paper, X means "unspecified".
# In our code, we represent X using None.
X = None

Pattern = Tuple[Any, ...]
Dataset = List[Tuple[Any, ...]]


def matches(row: Tuple[Any, ...], pattern: Pattern) -> bool:
    """
    Return True if a row matches a pattern.

    Example:
        row     = (1, 0, 1)
        pattern = (1, None, 1)

    The row matches because:
        A1 = 1
        A2 = anything
        A3 = 1
    """
    for p_val, r_val in zip(pattern, row):
        if p_val is not X and p_val != r_val:
            return False

    return True


def coverage(pattern: Pattern, dataset: Dataset) -> int:
    """
    Count how many rows in the dataset match the pattern.

    A pattern is covered if:
        coverage(pattern) >= tau

    A pattern is uncovered if:
        coverage(pattern) < tau
    """
    count = 0

    for row in dataset:
        if matches(row, pattern):
            count += 1

    return count


def is_uncovered(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    """
    Return True if the pattern has coverage smaller than tau.
    """
    return coverage(pattern, dataset) < tau


def is_covered(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    """
    Return True if the pattern has coverage at least tau.
    """
    return coverage(pattern, dataset) >= tau


def level(pattern: Pattern) -> int:
    """
    Return the level of a pattern.

    The level is the number of deterministic cells.

    Example:
        (None, 1, None, 0) has level 2.
    """
    return sum(1 for value in pattern if value is not X)


def parents(pattern: Pattern) -> List[Pattern]:
    """
    Generate all parents of a pattern.

    A parent is created by replacing one deterministic value with X.

    Example:
        pattern = (1, 1, 0)

        parents:
            (None, 1, 0)
            (1, None, 0)
            (1, 1, None)
    """
    result = []

    for i, value in enumerate(pattern):
        if value is not X:
            parent = list(pattern)
            parent[i] = X
            result.append(tuple(parent))

    return result


def children(pattern: Pattern, domains: List[List[Any]]) -> List[Pattern]:
    """
    Generate all children of a pattern.

    A child is created by replacing one X with a real attribute value.

    Example:
        pattern = (1, None, 0)
        domains[1] = [0, 1]

        children:
            (1, 0, 0)
            (1, 1, 0)
    """
    result = []

    for i, value in enumerate(pattern):
        if value is X:
            for attr_value in domains[i]:
                child = list(pattern)
                child[i] = attr_value
                result.append(tuple(child))

    return result


def is_mup(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    """
    A pattern is a MUP if:
        1. The pattern itself is uncovered.
        2. All of its parents are covered.
    """
    if not is_uncovered(pattern, dataset, tau):
        return False

    for parent in parents(pattern):
        if is_uncovered(parent, dataset, tau):
            return False

    return True


def is_parent_covered_mup(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    """
    Same as is_mup.

    Kept because some of your other files already import this name.
    """
    return is_mup(pattern, dataset, tau)


def dominates(general: Pattern, specific: Pattern) -> bool:
    """
    Return True if 'general' dominates 'specific'.

    general dominates specific if every deterministic value in general
    agrees with specific.

    Example:
        general  = (1, None, None)
        specific = (1, 0, 1)

        True, because 1XX contains 101.
    """
    for g_val, s_val in zip(general, specific):
        if g_val is not X and g_val != s_val:
            return False

    return True


def dominated_by_any_mup(pattern: Pattern, mups: Set[Pattern]) -> bool:
    """
    Return True if pattern is dominated by one of the discovered MUPs.

    If this is True, pattern cannot be a new MUP.
    """
    return any(dominates(mup, pattern) for mup in mups)


def dominates_any_mup(pattern: Pattern, mups: Set[Pattern]) -> bool:
    """
    Return True if pattern dominates one of the discovered MUPs.

    If this is True, pattern cannot be a new MUP.
    """
    return any(dominates(pattern, mup) for mup in mups)


def pattern_to_string(pattern: Pattern) -> str:
    """
    Pretty-print a pattern.

    Example:
        (1, None, 0) -> "1X0"
    """
    return "".join("X" if value is X else str(value) for value in pattern)


def print_patterns(patterns: Set[Pattern]) -> None:
    """
    Print patterns nicely.
    """
    for pattern in patterns:
        print(pattern_to_string(pattern), pattern)
