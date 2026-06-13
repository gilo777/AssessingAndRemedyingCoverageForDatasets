from itertools import product
from typing import List, Tuple, Any, Set

# X (None) is the wildcard / "unspecified" sentinel used throughout the paper.
# A pattern position holding X matches any value in that attribute's domain.
X = None


Pattern = Tuple[Any, ...]
Dataset = List[Tuple[Any, ...]]


def coverage(pattern: Pattern, dataset: Dataset) -> int:
    """
    Count how many rows match the pattern.
    X/None means the attribute is unspecified.
    """
    # Naive O(n * d) scan — correct for small datasets; CoverageOracle (below)
    # is the production-speed replacement for large ones.
    count = 0

    for row in dataset:
        match = True

        for p_val, r_val in zip(pattern, row):
            if p_val is not X and p_val != r_val:
                match = False
                break

        if match:
            count += 1

    return count


def is_uncovered(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    return coverage(pattern, dataset) < tau


def level(pattern: Pattern) -> int:
    """
    Number of deterministic cells.
    Example: (X, 1, X, 0) has level 2.
    """
    # Level is also the depth in the pattern lattice: the root (all X) has
    # level 0, a fully specified tuple has level d (number of attributes).
    return sum(1 for value in pattern if value is not X)


def parents(pattern: Pattern) -> List[Pattern]:
    """
    A parent is created by replacing one deterministic value with X.
    """
    # Moving from a pattern to its parents means removing one constraint →
    # the parent matches a superset of rows (is more general).
    # A level-k pattern has exactly k parents (one per deterministic attribute).
    result = []

    for i, value in enumerate(pattern):
        if value is not X:
            parent = list(pattern)
            parent[i] = X
            result.append(tuple(parent))

    return result


def children(pattern: Pattern, domains: List[List[Any]]) -> List[Pattern]:
    """
    A child is created by replacing one X with a possible attribute value.
    """
    # This unrestricted children() is used only for reference / testing.
    # The algorithms use children_rule1() to enforce the unique-parent property.
    result = []

    for i, value in enumerate(pattern):
        if value is X:
            for attr_value in domains[i]:
                child = list(pattern)
                child[i] = attr_value
                result.append(tuple(child))

    return result


def children_rule1(pattern: Pattern, domains: List[List[Any]]) -> List[Pattern]:
    """
    Children under the paper's Rule 1: only specialise the X's that lie to the
    right of the right-most deterministic element. Every node then has exactly
    one parent that generates it (Theorem 3), so a top-down traversal reaches
    each node once -- no visited set required.
    """
    # The unique-parent property is the key efficiency win: without it a BFS/DFS
    # would have to track visited nodes explicitly (expensive for large lattices).
    # Rule 1 imposes a canonical order: deterministic values always occupy a
    # prefix, so the single parent is always the one that placed the last value.
    rightmost = -1
    for i, value in enumerate(pattern):
        if value is not X:
            rightmost = i

    result = []
    for i in range(rightmost + 1, len(pattern)):
        # every position past `rightmost` is X by construction
        for attr_value in domains[i]:
            child = list(pattern)
            child[i] = attr_value
            result.append(tuple(child))

    return result


def is_parent_covered_mup(pattern: Pattern, dataset: Dataset, tau: int) -> bool:
    """
    A pattern is MUP if:
    1. it is uncovered
    2. all its parents are covered
    """
    # MUP = Maximal Uncovered Pattern. "Maximal" means no strictly more general
    # pattern is also uncovered — i.e., every parent (one step more general) IS
    # covered.  This is the formal definition from the paper (Definition 2).
    if not is_uncovered(pattern, dataset, tau):
        return False

    for parent in parents(pattern):
        if is_uncovered(parent, dataset, tau):
            return False

    return True


def dominates(general: Pattern, specific: Pattern) -> bool:
    """
    general dominates specific if every deterministic value in general
    agrees with specific.

    Example:
    (1, X, X) dominates (1, 0, 1)
    """
    # Dominance = "general is an ancestor of specific in the lattice."
    # If a MUP M dominates pattern P, then P is a descendant of M:
    #   - P is more specific (higher level) than M
    #   - P is definitely uncovered (inherits M's under-coverage)
    #   - P cannot itself be a MUP (it is not maximal)
    for g_val, s_val in zip(general, specific):
        if g_val is not X and g_val != s_val:
            return False

    return True


def dominated_by_any_mup(pattern: Pattern, mups: Set[Pattern]) -> bool:
    return any(dominates(mup, pattern) for mup in mups)


def dominates_any_mup(pattern: Pattern, mups: Set[Pattern]) -> bool:
    return any(dominates(pattern, mup) for mup in mups)



# ===========================================================================
# EFFICIENCY ADD-IN  —  paste this at the END of MutualFuncs.py
# ---------------------------------------------------------------------------
# Implements the paper's appendices:
#   Appendix A  ->  CoverageOracle      (inverted-index coverage; no per-call
#                                         scan of the dataset)
#   Appendix B  ->  MupDominanceIndex   (inverted-index MUP dominance; no
#                                         pattern-by-pattern comparison)
#
# Both use Python ints as arbitrary-width bit vectors, so ANDs are word-parallel
# and "stop as soon as the mask is empty" comes for free.
#
# Reuses `X` already defined at the top of MutualFuncs.py. The typing import
# below only adds `Dict`; re-importing the others is harmless.
# ===========================================================================
from typing import Any, Dict, List, Tuple


class CoverageOracle:
    """Appendix A: inverted-index coverage oracle.

    Instead of scanning the full dataset for every pattern query (O(n*d) each),
    we build a per-attribute inverted index at construction time: for each
    (attribute i, value v) pair we store a bitmask where bit k is set iff the
    k-th unique row has value v at attribute i.

    Querying a pattern ANDs together the masks for each pinned attribute — a
    word-parallel operation — then sums the counts of the matching unique rows.
    This reduces per-query cost from O(n*d) to O(m*d/W) where m is the number
    of unique rows and W is the machine word size (64).
    """

    def __init__(self, dataset: List[Tuple[Any, ...]]):
        # Aggregate duplicate rows: unique value combinations + a count vector.
        # Deduplication is the key: a dataset with n rows but only m << n unique
        # combinations shrinks the bitmask width from n to m.
        combo_count: Dict[Tuple[Any, ...], int] = {}
        for row in dataset:
            key = tuple(row)
            combo_count[key] = combo_count.get(key, 0) + 1

        self.combos: List[Tuple[Any, ...]] = list(combo_count.keys())
        self.counts: List[int] = [combo_count[c] for c in self.combos]
        self.m = len(self.combos)                       # number of unique combos
        self.full_mask = (1 << self.m) - 1
        self.d = len(self.combos[0]) if self.combos else 0

        # index[i][v] = bitmask over unique combos whose i-th value is v.
        self.index: List[Dict[Any, int]] = [dict() for _ in range(self.d)]
        for k, combo in enumerate(self.combos):
            bit = 1 << k
            for i, v in enumerate(combo):
                col = self.index[i]
                col[v] = col.get(v, 0) | bit

    def _match_mask(self, pattern: Pattern) -> int:
        """Bitmask of unique combos matching the pattern (deterministic cells AND-ed)."""
        # Start with all bits set (full_mask) and narrow down attribute by attribute.
        # Early exit on an empty mask avoids unnecessary AND operations.
        mask = self.full_mask
        for i, v in enumerate(pattern):
            if v is not X:
                mask &= self.index[i].get(v, 0)
                if mask == 0:
                    return 0
        return mask

    def coverage(self, pattern: Pattern) -> int:
        if self.m == 0:
            return 0
        mask = self._match_mask(pattern)
        total = 0
        while mask:                          # weighted popcount over set bits
            low = mask & (-mask)
            total += self.counts[low.bit_length() - 1]
            mask ^= low
        return total

    def is_uncovered(self, pattern: Pattern, tau: int) -> bool:
        """True iff coverage < tau, with an early exit once tau is reached."""
        # We stop accumulating as soon as total >= tau so covered patterns
        # that clearly pass the threshold are recognized quickly.
        if self.m == 0:
            return 0 < tau
        mask = self._match_mask(pattern)
        total = 0
        while mask:
            low = mask & (-mask)
            total += self.counts[low.bit_length() - 1]
            if total >= tau:
                return False
            mask ^= low
        return total < tau


class MupDominanceIndex:
    """Appendix B: inverted-index dominance checking over the discovered MUPs.

    Two operations are needed during DeepDiver's DFS (see DeepDiver.py):

    1. is_dominated_by_any(p): is p a descendant of some known MUP M?
       If yes, p is uncovered but NOT maximal — prune it immediately.

    2. dominates_any(p): is p an ancestor of some known MUP M?
       If yes, p is covered (all its parents are more general and therefore also
       covered), but we must still expand its children because they could reach
       other, unrelated MUPs.

    Naive implementations compare p against every known MUP (O(k*d)).
    The inverted index reduces this to O(d) per query regardless of how many
    MUPs have been discovered so far.
    """

    def __init__(self, d: int):
        self.d = d
        self.size = 0
        # value -> bitmask of MUPs carrying that (deterministic) value at attr i
        self.index: List[Dict[Any, int]] = [dict() for _ in range(d)]
        # bitmask of MUPs that are non-deterministic (X) at attr i
        self.x_mask: List[int] = [0] * d

    def add(self, mup: Pattern) -> None:
        # Assign this MUP bit position self.size and record its structure.
        bit = 1 << self.size
        for i, v in enumerate(mup):
            if v is X:
                self.x_mask[i] |= bit
            else:
                self.index[i][v] = self.index[i].get(v, 0) | bit
        self.size += 1

    def dominates_any(self, p: Pattern) -> bool:
        """True iff p dominates some MUP M (every deterministic p[i]=v has M[i]=v)."""
        # We need M[i] == p[i] for every pinned attribute of p.
        # A MUP M satisfies this iff it is in index[i][v] for every pinned (i,v).
        # Progressively AND the masks; return True if any MUP survives all filters.
        if self.size == 0:
            return False
        mask = (1 << self.size) - 1
        for i, v in enumerate(p):
            if v is not X:
                mask &= self.index[i].get(v, 0)   # M[i] must equal v (NOT X)
                if mask == 0:
                    return False
        return mask != 0

    def is_dominated_by_any(self, p: Pattern) -> bool:
        """True iff some MUP M dominates p (every deterministic M[i]=v has p[i]=v)."""
        # For M to dominate p, every pinned position of M must match p.
        # At attribute i:
        #   - if p[i] == X  → M[i] must be X (only x_mask[i] qualifies)
        #   - if p[i] == v  → M[i] == v or X (either index[i][v] or x_mask[i])
        if self.size == 0:
            return False
        mask = (1 << self.size) - 1
        for i, v in enumerate(p):
            if v is X:
                allowed = self.x_mask[i]                              # M[i] must be X
            else:
                allowed = self.index[i].get(v, 0) | self.x_mask[i]   # M[i] == v or X
            mask &= allowed
            if mask == 0:
                return False
        return mask != 0
