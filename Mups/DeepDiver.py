# ---------------------------------------------------------------------------
# CHANGES vs original DeepDiver.py:
#   [REMOVED] *** THE BUG ***  the `if dominates_any_mup(pattern, mups): continue`
#             branch. It skipped EXPANDING covered ancestors of known MUPs, which
#             orphaned other still-undiscovered MUPs (missed ~27% of MUPs in
#             random tests). Ancestors of a MUP are always covered, so they must
#             still be expanded; they just can't be MUPs themselves -- which the
#             uncovered/climb logic already guarantees.
#   [EDIT]    imports: CoverageOracle + MupDominanceIndex from .MutualFuncs;
#             drop naive helpers.
#   [REMOVED] standalone find_uncovered_parent(); the climb is inlined so it can
#             use the oracle instead of re-scanning the dataset.
#   [NEW]     build CoverageOracle + MupDominanceIndex once per call.
#   [EDIT]    dominance / coverage tests now go through them.
#   [NEW]     `if current not in mups` dedup guard + mup_index.add(current).
# ---------------------------------------------------------------------------

from .MutualFuncs import X, children, parents, CoverageOracle, MupDominanceIndex  # [EDIT] was: ... dominated_by_any_mup, dominates_any_mup, is_uncovered, children, parents

# [REMOVED] def find_uncovered_parent(pattern, dataset, tau): ...  -> inlined below using the oracle


def deepdiver(dataset, domains, tau):
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

        # Descendants of a known MUP cannot be MUPs -> prune. (kept; correct)
        if mup_index.is_dominated_by_any(pattern):                # [EDIT] was: if dominated_by_any_mup(pattern, mups):
            continue

        # [REMOVED] if dominates_any_mup(pattern, mups): continue   <-- the bug; deleted

        if oracle.is_uncovered(pattern, tau):                     # [EDIT] was: if is_uncovered(pattern, dataset, tau):
            # Climb up while an uncovered parent exists; the stopping node is a MUP.
            current = pattern
            while True:
                up = None                                         # [EDIT] inlined former find_uncovered_parent(), now oracle-backed
                for parent in parents(current):
                    if oracle.is_uncovered(parent, tau):
                        up = parent
                        break
                if up is None:
                    break
                current = up

            if current not in mups:                               # [NEW] dedup (same MUP can be reached via different descendants)
                mups.add(current)
                mup_index.add(current)                            # [NEW] keep the dominance index in sync
        else:
            for child in children(pattern, domains):
                if child not in visited:
                    stack.append(child)

    return mups