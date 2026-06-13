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
