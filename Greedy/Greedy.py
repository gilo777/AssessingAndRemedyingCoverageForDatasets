from itertools import product
from typing import Any, Callable, List, Set

from .GreedyHelper import (
    Dataset,
    Pattern,
    TupleRow,
    all_patterns_at_level,
    coverage,
    matches_tuple,
)


def greedy_data_collection_plan(
    dataset: Dataset,
    domains: List[List[Any]],
    tau: int,
    max_level: int,
    validation_oracle: Callable[[TupleRow], bool] = None,
) -> List[TupleRow]:
    """
    Greedy approximation for the Coverage Enhancement Problem (paper Section IV).

    Goal:
    Raise the maximum covered level of the dataset to ``max_level`` (the paper's
    lambda). Per Appendix C, this is guaranteed by hitting every *uncovered*
    pattern at exactly level ``max_level``: once those are covered, every more
    general pattern (level <= max_level) is covered too.

    The problem is a hitting set instance (paper Section IV-A): each uncovered
    pattern is a set, and a full value combination "hits" it if it matches the
    pattern. We greedily pick, at each step, the value combination that hits the
    most still-un-hit patterns (Algorithm 5, using the selection idea of
    Algorithm 4). This is the standard log-approximation for hitting set.

    Each returned value combination is a *specification* for data collection: the
    data owner then gathers enough real tuples (up to tau) matching it. A single
    combination can hit many patterns, which is why the output is far smaller than
    collecting data for each pattern separately.

    Parameters:
    dataset:
        The original dataset.

    domains:
        Possible values for each attribute, e.g. [[0, 1], [0, 1], [0, 1]].

    tau:
        Coverage threshold. A pattern is uncovered if fewer than tau tuples
        match it.

    max_level:
        The level lambda from the paper. After collection, every pattern at this
        level (and below) is covered.

    validation_oracle:
        Optional function returning True if a value combination is semantically
        valid (paper Definition 11). If None, every combination is valid.

    Returns:
        The list of value combinations (full tuples) to collect.
    """

    if validation_oracle is None:
        validation_oracle = lambda row: True

    # Step 1: the patterns to hit are the uncovered patterns at level max_level
    # (paper Appendix C). all_patterns_at_level enumerates every level-lambda
    # pattern; we keep the ones whose coverage is below the threshold.
    patterns_to_hit: Set[Pattern] = {
        pattern
        for pattern in all_patterns_at_level(domains, max_level)
        if coverage(pattern, dataset) < tau
    }

    if not patterns_to_hit:
        return []

    # Step 2: the universe of items U is every valid full value combination
    # (paper Section IV-A).
    candidate_tuples = [row for row in product(*domains) if validation_oracle(row)]

    # Step 3: greedy hitting set. Each iteration adds the value combination that
    # hits the most remaining patterns, then drops those patterns (Algorithm 5).
    tuples_to_collect: List[TupleRow] = []

    while patterns_to_hit:
        best_candidate_tuple = None
        best_hit_patterns: Set[Pattern] = set()

        for candidate_tuple in candidate_tuples:
            hit_patterns = {
                pattern for pattern in patterns_to_hit
                if matches_tuple(candidate_tuple, pattern)
            }
            if len(hit_patterns) > len(best_hit_patterns):
                best_candidate_tuple = candidate_tuple
                best_hit_patterns = hit_patterns

        if best_candidate_tuple is None:
            # Some pattern has no valid value combination matching it, so the
            # requested coverage level cannot be reached under the oracle.
            raise ValueError(
                "Cannot cover all patterns: the validation oracle blocks every "
                "value combination that would hit some remaining pattern."
            )

        tuples_to_collect.append(best_candidate_tuple)
        patterns_to_hit -= best_hit_patterns

    return tuples_to_collect