"""PATTERN-BREAKER (paper §III-C, Algorithm 1): the top-down MUP algorithm.

Walks the pattern graph from the root (all-X) downward, level by level. The
"monotonicity" of coverage drives the pruning: if a pattern is uncovered, all
of its descendants are uncovered too, so a *covered* pattern is the only thing
worth breaking down further.

Two devices from the paper keep the traversal cheap:

  * Rule 1 / Theorem 3 -- a covered pattern only specialises the X's to the
    right of its right-most deterministic element. Every node then has exactly
    one generating parent, so the graph collapses to a tree and each node is
    reached exactly once (no `visited` set required). See `children_rule1`.

  * Parent pruning (Algorithm 1, lines 7-11) -- a candidate that has an
    uncovered parent is itself uncovered (monotonicity) and is dominated by
    that parent, so it can never be a MUP. We drop it without ever computing
    its coverage. Because every covered node is generated as a candidate
    (Rule 1, by induction from the root), "this parent is covered" is exactly
    "this parent was one of the covered nodes one level up".

Coverage is computed by the inverted-index CoverageOracle (Appendix A).

A MUP is an uncovered pattern all of whose parents are covered (Definition 5).
Once parent pruning has let a pattern through, *all* of its parents are known
to be covered -- so an uncovered survivor is, by definition, a MUP.

NOTE: Algorithm 1 as printed tracks the whole previous level (`Qp`) and tries
to recover "parent covered" from `Qp` plus the MUP set. That pseudocode lets
uncovered-but-non-maximal candidates slip through and over-reports MUPs. We
instead track the covered nodes of the previous level directly, which is what
the parent check is really asking for.
"""

from .MutualFuncs import X, parents, children_rule1, CoverageOracle


def pattern_breaker(dataset, domains, tau):
    if tau <= 0:
        return set()

    root = tuple([X] * len(domains))
    oracle = CoverageOracle(dataset)

    mups = set()
    current_level = [root]
    covered_above = set()

    while current_level:
        next_level = []
        covered_here = set()

        for pattern in current_level:
            if any(parent not in covered_above for parent in parents(pattern)):
                continue

            if oracle.is_uncovered(pattern, tau):
                mups.add(pattern)
            else:
                covered_here.add(pattern)
                next_level.extend(children_rule1(pattern, domains))

        current_level = next_level
        covered_above = covered_here

    return mups
