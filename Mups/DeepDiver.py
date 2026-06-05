"""DEEPDIVER (paper §III-E, Algorithm 3): fast search-space pruner for MUPs.

Strategy: dive down the pattern graph (DFS) until reaching an uncovered node,
then climb back up to the maximal uncovered pattern (MUP) it belongs to. Each
discovered MUP shrinks the search space in both directions:
  * its descendants are uncovered but never maximal  -> pruned;
  * its ancestors are always covered                 -> coverage test skipped.

Coverage (Appendix A) and MUP dominance (Appendix B) both go through the
inverted-index oracles in MutualFuncs. Children are generated with Rule 1, so
every node is reached exactly once and no visited set is needed (Theorem 3).
"""

from .MutualFuncs import X, parents, children_rule1, CoverageOracle, MupDominanceIndex


def pattern_diver(dataset, domains, tau):
    num_attributes = len(domains)
    root_pattern = tuple([X] * num_attributes)

    coverage_oracle = CoverageOracle(dataset)            # Appendix A: coverage via inverted index
    mup_dominance_index = MupDominanceIndex(num_attributes)  # Appendix B: dominance via inverted index

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
            # Covered: dive into its children.
            patterns_to_explore.extend(children_rule1(pattern, domains))

    return discovered_mups