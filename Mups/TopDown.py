# ---------------------------------------------------------------------------
# CHANGES vs original TopDown.py:
#   [EDIT]    imports: now pull CoverageOracle + MupDominanceIndex from
#             .MutualFuncs (where the add-in lives); drop the naive
#             coverage/dominance helpers.
#   [NEW]     _is_mup(): replaces is_parent_covered_mup, checking coverage via
#             the oracle (Appendix A) instead of scanning the dataset.
#   [NEW]     build a CoverageOracle + MupDominanceIndex once per call.
#   [EDIT]    every is_uncovered / dominance / MUP test goes through them.
#   [NEW]     mup_index.add(pattern) whenever a MUP is recorded.
#   (Traversal logic is otherwise unchanged.)
# ---------------------------------------------------------------------------

from .MutualFuncs import X, children, parents, CoverageOracle, MupDominanceIndex  # [EDIT] was: ... dominated_by_any_mup, is_parent_covered_mup, is_uncovered, children


def _is_mup(oracle, pattern, tau):                                 # [NEW] replaces is_parent_covered_mup (uses the oracle)
    if not oracle.is_uncovered(pattern, tau):
        return False
    for parent in parents(pattern):
        if oracle.is_uncovered(parent, tau):
            return False
    return True


def pattern_breaker(dataset, domains, tau):
    d = len(domains)
    root = tuple([X] * d)

    oracle = CoverageOracle(dataset)                               # [NEW] Appendix A index
    mup_index = MupDominanceIndex(d)                               # [NEW] Appendix B index

    mups = set()
    stack = [root]
    visited = set()

    while stack:
        pattern = stack.pop()

        if pattern in visited:
            continue
        visited.add(pattern)

        if mup_index.is_dominated_by_any(pattern):                # [EDIT] was: if dominated_by_any_mup(pattern, mups):
            continue

        if oracle.is_uncovered(pattern, tau):                     # [EDIT] was: if is_uncovered(pattern, dataset, tau):
            if _is_mup(oracle, pattern, tau):                     # [EDIT] was: if is_parent_covered_mup(pattern, dataset, tau):
                mups.add(pattern)
                mup_index.add(pattern)                            # [NEW] keep the dominance index in sync
            continue

        for child in children(pattern, domains):
            if child not in visited:
                stack.append(child)

    return mups