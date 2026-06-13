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

    coverage_oracle = CoverageOracle(dataset)
    mup_dominance_index = MupDominanceIndex(num_attributes)

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