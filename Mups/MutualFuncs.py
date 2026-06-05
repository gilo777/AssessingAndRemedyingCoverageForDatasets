from typing import List, Tuple, Any, Set

X = None

Pattern = Tuple[Any, ...]
Dataset = List[Tuple[Any, ...]]


def coverage(pattern: Pattern, dataset: Dataset) -> int:
    count = 0

    for row in dataset:
        match = True

        for p_val, r_val in zip(pattern, row):
            if p_val is not X and p_val != r_val:
                match = False
                break

        if match:
            count += 1

    return count


def parents(pattern: Pattern) -> List[Pattern]:
    result = []

    for i, value in enumerate(pattern):
        if value is not X:
            parent = list(pattern)
            parent[i] = X
            result.append(tuple(parent))

    return result
