"""MMD squared in unbiased-estimator form and whole-search permutation tests.

The Gaussian bandwidth grid is label-invariant: pooled positive-distance median
times fixed multipliers. The max over that grid is recomputed for every labeling.
A deliberately invalid frozen-winner p-value is exposed only as a diagnostic.
Unbiasedness concerns a fixed kernel, not a data-adaptive bandwidth or maximum.
"""

from dataclasses import dataclass
from itertools import combinations
import math
import operator

import numpy as np

DEFAULT_FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TIE_TOLERANCE = 1e-12


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _real_array(values, name):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric data")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _samples(values):
    values = _real_array(values, "samples")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 2 or values.shape[1] < 1:
        raise ValueError("samples must have at least two rows and one feature")
    return values


def _distances(samples):
    # Accumulated hypot avoids squaring huge/tiny feature differences first.
    distances = np.zeros((len(samples), len(samples)))
    with np.errstate(over="ignore", invalid="ignore"):
        for column in samples.T:
            difference = column[:, None] - column[None, :]
            np.hypot(distances, difference, out=distances)
    if not np.all(np.isfinite(distances)):
        raise ValueError("pairwise distances overflow; rescale features")
    return distances


def pooled_bandwidths(samples, factors=DEFAULT_FACTORS):
    """Median of strictly positive pooled pair distances times fixed factors.

    Zero distances are excluded. Identical rows use a base bandwidth of 1;
    all resulting Gram matrices are constant, so every test returns p=1.
    No group labels enter this computation.
    """
    samples = _samples(samples)
    factors = _real_array(factors, "factors")
    if factors.ndim != 1 or not len(factors) or np.any(factors <= 0):
        raise ValueError("factors must be a nonempty positive vector")
    distances = _distances(samples)
    positive = distances[np.triu_indices(len(samples), 1)]
    positive = np.sort(positive[positive > 0])
    if not len(positive):
        base = 1.0
    else:
        high = len(positive) // 2
        low = (len(positive) - 1) // 2
        base = positive[low] + (positive[high] - positive[low]) / 2
    with np.errstate(over="ignore", under="ignore"):
        widths = base * factors
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("bandwidth grid exceeds numeric range; rescale features")
    return widths


def rbf_kernels(samples, bandwidths):
    """Return K matrices with k(x,y)=exp(-||x-y||^2/(2*h^2))."""
    samples = _samples(samples)
    bandwidths = _real_array(bandwidths, "bandwidths")
    if bandwidths.ndim != 1 or not len(bandwidths) or np.any(bandwidths <= 0):
        raise ValueError("bandwidths must be a nonempty positive vector")
    distances = _distances(samples)
    with np.errstate(over="ignore", under="ignore"):
        ratios = distances[None, :, :] / bandwidths[:, None, None]
        kernels = np.exp(-0.5 * ratios * ratios)
    return kernels


def _sizes(total, group_size):
    total = _integer(total, "total", 4)
    group_size = _integer(group_size, "group_size", 2)
    if total - group_size < 2:
        raise ValueError("both groups need at least two observations")
    return total, group_size


def exact_assignments(total, group_size, max_assignments=50_000):
    """Enumerate each size-m subset once, not N! row permutations."""
    total, group_size = _sizes(total, group_size)
    cap = _integer(max_assignments, "max_assignments", 1)
    count = math.comb(total, group_size)
    if count > cap:
        raise ValueError(f"exact orbit has {count} assignments, above cap {cap}")
    result = np.zeros((count, total), dtype=bool)
    for row, indices in enumerate(combinations(range(total), group_size)):
        result[row, list(indices)] = True
    return result


def random_assignments(total, group_size, draws=199, seed=0):
    """Observed first-m labeling plus B independent uniform size-m subsets.

    Subsets are sampled with replacement across draws; repeats (including the
    observed labeling) remain in the reference sample. Each subset internally
    contains no repeated index. The observed row contributes the p-value's +1.
    """
    total, group_size = _sizes(total, group_size)
    draws = _integer(draws, "draws", 1)
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    masks = np.zeros((draws + 1, total), dtype=bool)
    masks[0, :group_size] = True
    for row in range(1, draws + 1):
        masks[row, rng.choice(total, group_size, replace=False)] = True
    return masks


def mmd_scores(kernels, assignments, batch_size=512):
    """Two-sample MMD^2 in unbiased-estimator form, for R labelings and K kernels.

    Diagonals are excluded, including for arbitrary supplied symmetric matrices.
    PSD is not checked: the MMD interpretation needs a valid kernel, while the
    permutation argument only needs a label-equivariant statistic. Negative
    estimates are retained. All rows must have the same group sizes. Unbiasedness
    for a population MMD target requires a fixed, not data-adaptive, kernel.
    """
    kernels = _real_array(kernels, "kernels")
    if kernels.ndim == 2:
        kernels = kernels[None, :, :]
    if kernels.ndim != 3 or not len(kernels) or kernels.shape[1] != kernels.shape[2]:
        raise ValueError("kernels must have shape (K,N,N) or (N,N)")
    if not np.allclose(kernels, kernels.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("kernel matrices must be symmetric")
    masks = np.asarray(assignments)
    if masks.dtype != np.bool_ or masks.ndim != 2 or not len(masks) or masks.shape[1] != kernels.shape[1]:
        raise ValueError("assignments must be a nonempty boolean R-by-N matrix")
    sizes = masks.sum(axis=1)
    total, m = _sizes(kernels.shape[1], int(sizes[0]))
    if np.any(sizes != m):
        raise ValueError("assignment rows must have the same group size")
    n = total - m
    batch_size = _integer(batch_size, "batch_size", 1)
    kernels = kernels / 2 + kernels.swapaxes(1, 2) / 2
    kernels[:, np.arange(total), np.arange(total)] = 0.0
    result = np.empty((len(masks), len(kernels)))
    with np.errstate(over="ignore", invalid="ignore"):
        for j, kernel in enumerate(kernels):
            row_sums = kernel.sum(axis=1)
            whole = row_sums.sum()
            for start in range(0, len(masks), batch_size):
                w = masks[start:start + batch_size].astype(float)
                xx = np.sum((w @ kernel) * w, axis=1)
                x_all = w @ row_sums
                yy = whole - 2 * x_all + xx
                xy = x_all - xx
                result[start:start + len(w), j] = xx / (m * (m - 1)) + yy / (n * (n - 1)) - 2 * xy / (m * n)
    if not np.all(np.isfinite(result)):
        raise ValueError("kernel arithmetic overflow; rescale the kernel")
    return result


def upper_tail_ranks(values, tolerance=TIE_TOLERANCE):
    """Inclusive tail proportions for each row, within each statistic column.

    Thresholds are lowered by tolerance*max(1,max_abs_column) to count numerical
    near-ties conservatively. Rank denominators include each row itself. This
    is an exact-orbit tail fraction, or the +1 Monte Carlo p-value for row zero
    when that row is observed and all other rows are random assignments.
    """
    values = _real_array(values, "statistics")
    if values.ndim not in (1, 2) or not len(values) or (values.ndim == 2 and not values.shape[1]):
        raise ValueError("statistics must be a nonempty vector or matrix")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite nonnegative scalar")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tolerance must be a finite nonnegative scalar") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative scalar")
    matrix = values[:, None] if values.ndim == 1 else values
    result = np.empty_like(matrix)
    for j, column in enumerate(matrix.T):
        slack = tolerance * max(1.0, float(np.max(np.abs(column))))
        sorted_values = np.sort(column)
        with np.errstate(over="ignore"):
            thresholds = column - slack
        result[:, j] = (len(column) - np.searchsorted(sorted_values, thresholds, side="left")) / len(column)
    return result[:, 0] if values.ndim == 1 else result


@dataclass(frozen=True)
class TestResult:
    p_fixed: float
    p_max: float
    p_frozen_winner_invalid: float
    selected_index: int
    fixed_index: int
    bandwidths: tuple[float, ...]
    observed_scores: tuple[float, ...]
    reference_rows: int
    exact: bool


def permutation_test(x, y, *, bandwidths=None, fixed_index=None, draws=199, seed=0, exact=False):
    """Compare a fixed kernel, recalibrated kernel maximum, and invalid freezing.

    User-specified bandwidths must be fixed in advance or label-invariant. This
    function cannot detect a kernel grid tuned using the observed group labels.
    The null is equality of distributions for independent observations; this is
    not a test valid under merely equal means or unrestricted serial dependence.
    """
    x, y = _samples(x), _samples(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("groups must have equal feature dimensions")
    if not isinstance(exact, (bool, np.bool_)):
        raise ValueError("exact must be boolean")
    pooled = np.concatenate((x, y))
    widths = pooled_bandwidths(pooled) if bandwidths is None else _real_array(bandwidths, "bandwidths")
    kernels = rbf_kernels(pooled, widths)
    fixed = len(widths) // 2 if fixed_index is None else _integer(fixed_index, "fixed_index", 0)
    if fixed >= len(widths):
        raise ValueError("fixed_index is outside the kernel grid")
    masks = exact_assignments(len(pooled), len(x)) if exact else random_assignments(len(pooled), len(x), draws, seed)
    scores = mmd_scores(kernels, masks)
    per_kernel = upper_tail_ranks(scores)
    winner = int(np.argmax(scores[0]))
    return TestResult(
        p_fixed=float(per_kernel[0, fixed]),
        p_max=float(upper_tail_ranks(scores.max(axis=1))[0]),
        p_frozen_winner_invalid=float(per_kernel[0, winner]),
        selected_index=winner, fixed_index=fixed,
        bandwidths=tuple(float(v) for v in widths),
        observed_scores=tuple(float(v) for v in scores[0]),
        reference_rows=len(masks), exact=bool(exact),
    )
