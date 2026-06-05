# ---------------------------------------------------------------------------
# CHANGES vs original BottomUp.py:
#   [EDIT]    imports: CoverageOracle + MupDominanceIndex from .MutualFuncs;
#             drop naive helpers.
#   [NEW]     _is_mup(): oracle-backed replacement for is_parent_covered_mup.
#   [NEW]     build CoverageOracle + MupDominanceIndex once per call.
#   [EDIT]    dominance / MUP tests now go through them.
#   [NEW]     mup_index.add(pattern) when a MUP is found.
#   [REMOVED] the `already_checked` set (every pattern is generated at exactly
#             one level, so it never actually deduped anything).
#   NOTE:     this still ENUMERATES every level; it is faster only because the
#             per-pattern checks are now cheap. It does NOT yet use the paper's
#             "sum children coverages + prune at covered nodes" bottom-up walk.
#   all_patterns_at_level() is unchanged.
# ---------------------------------------------------------------------------

from .MutualFuncs import X, parents, CoverageOracle, MupDominanceIndex  # [EDIT] was: ... dominated_by_any_mup, is_parent_covered_mup


def all_patterns_at_level(domains, target_level):                  # (unchanged)
    d = len(domains)
    result = []

    def backtrack(index, current, deterministic_count):
        if index == d:
            if deterministic_count == target_level:
                result.append(tuple(current))
            return
        current.append(X)
        backtrack(index + 1, current, deterministic_count)
        current.pop()
        for value in domains[index]:
            current.append(value)
            backtrack(index + 1, current, deterministic_count + 1)
            current.pop()

    backtrack(0, [], 0)
    return result


def _is_mup(oracle, pattern, tau):                                 # [NEW] replaces is_parent_covered_mup (uses the oracle)
    if not oracle.is_uncovered(pattern, tau):
        return False
    for parent in parents(pattern):
        if oracle.is_uncovered(parent, tau):
            return False
    return True


def pattern_combiner(dataset, domains, tau):
    d = len(domains)

    oracle = CoverageOracle(dataset)                               # [NEW] Appendix A index
    mup_index = MupDominanceIndex(d)                               # [NEW] Appendix B index

    mups = set()
    # [REMOVED] already_checked = set()   # redundant: each pattern lives at one level only

    for current_level in range(d, -1, -1):
        for pattern in all_patterns_at_level(domains, current_level):
            # [REMOVED] if pattern in already_checked: continue
            # [REMOVED] already_checked.add(pattern)

            if mup_index.is_dominated_by_any(pattern):            # [EDIT] was: if dominated_by_any_mup(pattern, mups):
                continue

            if _is_mup(oracle, pattern, tau):                     # [EDIT] was: if is_parent_covered_mup(pattern, dataset, tau):
                mups.add(pattern)
                mup_index.add(pattern)                            # [NEW] keep the dominance index in sync

    return mups