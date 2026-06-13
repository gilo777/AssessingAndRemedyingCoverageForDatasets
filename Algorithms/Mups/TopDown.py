from .MutualFuncs import X, parents, children_rule1, CoverageOracle


def pattern_breaker(dataset, domains, tau):
    """Top-down BFS traversal of the pattern lattice to find all MUPs.

    Strategy: start at the root (all X, level 0) and expand level by level.
    A node is expanded only if it is *covered* — uncovered nodes are MUPs
    (since their parent was covered, satisfying the maximality condition).

    Why BFS (level by level)?
    - We need to know which nodes at level k are covered BEFORE we can correctly
      check their children at level k+1 for the "all parents covered" guard.
    - covered_above tracks covered nodes from the previous level so that the
      guard below can be checked cheaply without re-querying the oracle.

    Complexity advantage over naive: thanks to children_rule1(), each node has
    exactly one parent, so no visited-set is needed and each node is enqueued
    at most once.
    """
    if tau <= 0:
        return set()

    # Root pattern: all X means no constraints → matches every row.
    root = tuple([X] * len(domains))
    oracle = CoverageOracle(dataset)

    mups = set()
    current_level = [root]
    covered_above = set()          # covered nodes from the previous BFS level

    while current_level:
        next_level = []
        covered_here = set()       # covered nodes at this level (for the next iteration)

        for pattern in current_level:
            # Guard: only process a node if ALL of its parents were covered.
            # Under Rule 1 each node has exactly one parent, so this is a single
            # membership check.  A node whose parent was uncovered is already
            # dominated by that parent-MUP and does not need to be evaluated.
            if any(parent not in covered_above for parent in parents(pattern)):
                continue

            if oracle.is_uncovered(pattern, tau):
                # Uncovered + all parents covered → this IS a MUP (Definition 2).
                mups.add(pattern)
            else:
                # Covered: record it and schedule its children for the next level.
                covered_here.add(pattern)
                next_level.extend(children_rule1(pattern, domains))

        current_level = next_level
        covered_above = covered_here

    return mups