"""DEEPDIVER (paper §III-E, Algorithm 3): fast search-space pruner for MUPs.

Strategy: dive down the pattern graph (DFS) until reaching an uncovered node,
then climb back up to the maximal uncovered pattern (MUP) it belongs to. Each
discovered MUP shrinks the search space in both directions:
  * its descendants are uncovered but never maximal  -> pruned;
  * its ancestors are always covered                 -> coverage test skipped.

Coverage (Appendix A) and MUP dominance (Appendix B) both go through the
inverted-index oracles in MutualFuncs. Children are generated with Rule 1, so
every node is reached exactly once and no visited set is needed (Theorem 3).

LEVEL-LIMITED SEARCH (paper §V-C-3, Figure 16)
----------------------------------------------
`max_level` caps the discovery to MUPs of level <= max_level (None = unlimited,
the original behaviour). The paper motivates this: low-level MUPs (combinations
of one or two attributes) are the "risky" ones worth finding, while high-level
MUPs are too specific to be interesting, and capping the level is what lets MUP
identification scale to tens of attributes.

Correctness of the cap: a MUP M of level k <= max_level is always generated as
the Rule-1 child of its (covered) generating parent at level k-1 < max_level
(Theorem 3), so the dive never has to pass level max_level to reach it. The
climb only moves to more general patterns (lower level), so it never violates
the cap. Hence stopping the dive at max_level finds exactly the MUPs with
level <= max_level. We implement the cap by not expanding the children of a
pattern once it sits at level max_level.
"""

from .MutualFuncs import (
    X,
    level,
    parents,
    children_rule1,
    CoverageOracle,
    MupDominanceIndex,
)


def pattern_diver(dataset, domains, tau, max_level=None):
    num_attributes = len(domains)
    root_pattern = tuple([X] * num_attributes)

    coverage_oracle = CoverageOracle(dataset)            # Appendix A: coverage via inverted index
    mup_dominance_index = MupDominanceIndex(num_attributes)  # Appendix B: dominance via inverted index

    # When max_level is None there is no cap; otherwise we only expand a node's
    # children while it is still strictly below the cap.
    def can_expand(pattern):
        return max_level is None or level(pattern) < max_level

    discovered_mups = set()
    patterns_to_explore = [root_pattern]

    while patterns_to_explore:
        pattern = patterns_to_explore.pop()

        # Descendant of a known MUP: uncovered, but not maximal -> prune.
        if mup_dominance_index.is_dominated_by_any(pattern):
            continue

        # Ancestor of a known MUP: guaranteed covered, so skip the coverage test
        # and keep diving (still needed to reach other MUPs).
        if mup_dominance_index.dominates_any(pattern):
            if can_expand(pattern):
                patterns_to_explore.extend(children_rule1(pattern, domains))
            continue

        if coverage_oracle.is_uncovered(pattern, tau):
            # Climb to the maximal uncovered ancestor; that node is the MUP.
            mup_candidate = pattern
            while True:
                uncovered_parent = next(
                    (parent for parent in parents(mup_candidate)
                     if coverage_oracle.is_uncovered(parent, tau)),
                    None,
                )
                if uncovered_parent is None:
                    break
                mup_candidate = uncovered_parent

            if mup_candidate not in discovered_mups:
                discovered_mups.add(mup_candidate)
                mup_dominance_index.add(mup_candidate)
        else:
            # Covered: dive into its children (unless we've hit the level cap).
            if can_expand(pattern):
                patterns_to_explore.extend(children_rule1(pattern, domains))

    return discovered_mups