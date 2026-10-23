"""Selection-aware MMD tests for independent, exchangeable observations."""

from .core import (
    TestResult, exact_assignments, mmd_scores, permutation_test,
    pooled_bandwidths, random_assignments, rbf_kernels, upper_tail_ranks,
)

__all__ = [
    "TestResult", "exact_assignments", "mmd_scores", "permutation_test",
    "pooled_bandwidths", "random_assignments", "rbf_kernels", "upper_tail_ranks",
]
