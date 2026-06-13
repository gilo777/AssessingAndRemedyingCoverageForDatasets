from .MutualFuncs import (
    X,
    level,
    parents,
    children_rule1,
    CoverageOracle,
    MupDominanceIndex,
)


def pattern_diver(dataset, domains, tau, max_level=None):
    """DFS-based MUP discovery with dominance pruning (DeepDiver algorithm).

    Why DFS instead of BFS?
    - BFS (TopDown) must hold an entire frontier in memory; DFS uses O(depth)
      stack space, which matters for large d.
    - DFS reaches deep uncovered regions quickly and populates the dominance
      index early, enabling aggressive pruning before most of the lattice is
      touched.

    Two pruning rules powered by MupDominanceIndex:

    Rule A — is_dominated_by_any(pattern):
        Some already-found MUP M is an ancestor of `pattern`.  Because M is
        uncovered and is more general than `pattern`, `pattern` is also
        uncovered but it is NOT maximal (M is strictly more general).  Safe
        to prune the entire subtree rooted at `pattern`.

    Rule B — dominates_any(pattern):
        `pattern` is an ancestor of some already-found MUP M.  Ancestors of
        an uncovered pattern are necessarily covered (by MUP maximality), so
        we skip the oracle call and skip the "climb" step.  We still expand
        children because there may be OTHER MUPs in the subtree that are not
        dominated by M.

    Climb-to-MUP step:
        When `pattern` is found to be uncovered, it might not itself be a MUP
        — there could be more-general uncovered ancestors.  We climb greedily
        upward (via parents()) until no uncovered parent exists; that node is
        the MUP.  This avoids issuing DFS from all ancestors of the true MUP
        and discovering the same MUP multiple times.

    max_level cap:
        Limits the level of MUPs we look for.  A node at level == max_level
        will not have its children expanded, so MUPs strictly deeper than the
        cap are missed.  Useful for the Figure-15 runtime experiment.
    """
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

        # Rule A: Descendant of a known MUP: uncovered, but not maximal -> prune.
        if mup_dominance_index.is_dominated_by_any(pattern):
            continue

        # Rule B: Ancestor of a known MUP: guaranteed covered, so skip the
        # coverage test and keep diving (still needed to reach other MUPs).
        if mup_dominance_index.dominates_any(pattern):
            if can_expand(pattern):
                patterns_to_explore.extend(children_rule1(pattern, domains))
            continue

        if coverage_oracle.is_uncovered(pattern, tau):
            # Climb to the maximal uncovered ancestor; that node is the MUP.
            # We keep moving to a parent as long as that parent is also uncovered.
            # The first node with no uncovered parent satisfies Definition 2.
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
                # Register in the index so future DFS nodes benefit from both
                # pruning rules immediately.
                mup_dominance_index.add(mup_candidate)
        else:
            # Covered: dive into its children (unless we've hit the level cap).
            if can_expand(pattern):
                patterns_to_explore.extend(children_rule1(pattern, domains))

    return discovered_mups
