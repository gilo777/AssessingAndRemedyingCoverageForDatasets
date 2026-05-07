import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from Greedy.Greedy import greedy_coverage_enhancement
from Mups.BottomUp import pattern_combiner
from Mups.DeepDiver import deepdiver
from Mups.TopDown import pattern_breaker




def main():
    # dataset = [
    #     (0, 1, 0),
    #     (0, 0, 1),
    #     (0, 0, 0),
    #     (0, 1, 1),
    #     (0, 0, 1),
    # ]

    # domains = [
    #     [0, 1],
    #     [0, 1],
    #     [0, 1],
    # ]

    # tau = 1
    
    # dataset = [
    #     (0, 0, 0),
    #     (0, 0, 1),
    #     (0, 1, 0),
    #     (0, 1, 1),
    # ]

    # domains = [
    #     [0, 1],
    #     [0, 1],
    #     [0, 1],
    # ]

    # tau = 1
    
    # dataset = [
    #     (0, 0, 0),
    #     (0, 0, 1),
    #     (0, 1, 0),
    #     (1, 0, 0),
    # ]

    # domains = [
    #     [0, 1],
    #     [0, 1],
    #     [0, 1],
    # ]

    # tau = 1
    
    dataset = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ]

    domains = [
        [0, 1],
        [0, 1],
        [0, 1],
    ]

    tau = 1
    max_level = 2

    extra_rows = greedy_coverage_enhancement(
        dataset=dataset,
        domains=domains,
        tau=tau,
        max_level=max_level
    )

    print(extra_rows)

    print("Top-down Pattern-Breaker:")
    print(pattern_breaker(dataset, domains, tau))

    print("Bottom-up Pattern-Combiner:")
    print(pattern_combiner(dataset, domains, tau))

    print("DeepDiver:")
    print(deepdiver(dataset, domains, tau))


if __name__ == "__main__":
    main()