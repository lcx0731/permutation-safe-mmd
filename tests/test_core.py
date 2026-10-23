from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest

from safe_mmd import (
    exact_assignments, mmd_scores, permutation_test, pooled_bandwidths,
    random_assignments, rbf_kernels, upper_tail_ranks,
)


def fraction_score(kernel, mask):
    x = [i for i, label in enumerate(mask) if label]
    y = [i for i, label in enumerate(mask) if not label]
    xx = sum(int(kernel[i, j]) for i in x for j in x if i != j)
    yy = sum(int(kernel[i, j]) for i in y for j in y if i != j)
    xy = sum(int(kernel[i, j]) for i in x for j in y)
    return Fraction(xx, len(x) * (len(x) - 1)) + Fraction(yy, len(y) * (len(y) - 1)) - Fraction(2 * xy, len(x) * len(y))


@pytest.mark.parametrize("group_size", [2, 3, 4])
def test_scores_match_independent_fraction_oracle(group_size):
    features = np.array([[-2, 1], [-1, -2], [0, 0], [1, 3], [2, -1], [4, 2]])
    kernel = features @ features.T
    assignments = exact_assignments(6, group_size)
    expected = [float(fraction_score(kernel, row)) for row in assignments]
    np.testing.assert_allclose(mmd_scores(kernel, assignments)[:, 0], expected, atol=1e-13, rtol=1e-13)


def test_unbiased_mmd_can_be_negative_even_for_a_valid_rbf_kernel():
    kernel = rbf_kernels([0, 1, 0, 1], [1])
    observed = np.array([[True, True, False, False]])
    score = mmd_scores(kernel, observed)[0, 0]
    assert score == pytest.approx(math.exp(-0.5) - 1)
    assert score < 0


def test_diagonal_is_excluded_and_inputs_not_modified():
    kernels = rbf_kernels(np.arange(6), [1, 2])
    original = kernels.copy()
    assignments = exact_assignments(6, 2)
    expected = mmd_scores(kernels, assignments)
    np.testing.assert_array_equal(kernels, original)
    kernels[:, np.arange(6), np.arange(6)] = 1e300
    np.testing.assert_allclose(mmd_scores(kernels, assignments), expected, atol=1e-13)


def test_row_permutation_and_group_complement_invariance():
    z = np.array([-3, -1, 0, 2, 3, 4])
    kernels = rbf_kernels(z, [0.5, 2])
    masks = exact_assignments(6, 2)
    expected = mmd_scores(kernels, masks)
    np.testing.assert_allclose(mmd_scores(kernels, ~masks), expected, atol=1e-13)
    order = [3, 0, 4, 1, 5, 2]
    permuted = kernels[:, order, :][:, :, order]
    np.testing.assert_allclose(mmd_scores(permuted, masks[:, order]), expected, atol=1e-13)


def test_batch_sizes_and_duplicate_kernels_do_not_change_scores():
    kernels = rbf_kernels(np.arange(8), [0.1, 1, 3])
    masks = exact_assignments(8, 3)
    np.testing.assert_allclose(mmd_scores(kernels, masks, batch_size=1),
                               mmd_scores(kernels, masks, batch_size=11), atol=1e-13)
    repeated = np.concatenate([kernels, kernels])
    np.testing.assert_array_equal(mmd_scores(repeated, masks)[:, :3], mmd_scores(repeated, masks)[:, 3:])


def test_pooled_bandwidth_is_label_order_invariant_and_scale_equivariant():
    z = np.array([-2, -1, 0, 1, 4, 4])
    widths = pooled_bandwidths(z)
    np.testing.assert_array_equal(pooled_bandwidths(z[::-1]), widths)
    np.testing.assert_allclose(pooled_bandwidths(7 * z + 100), widths * 7, atol=1e-13)
    np.testing.assert_allclose(rbf_kernels(7 * z + 100, widths * 7), rbf_kernels(z, widths), atol=1e-13)


def test_bandwidth_excludes_zero_distances_and_constant_sample_fallback():
    # Strictly positive distances: 2, 6, 2, 6, 4; median=4.
    assert pooled_bandwidths([0, 0, 2, 6], factors=[1])[0] == 4
    np.testing.assert_array_equal(pooled_bandwidths([3, 3, 3], factors=[1, 2]), [1, 2])
    result = permutation_test([3, 3, 3], [3, 3, 3], exact=True)
    assert result.p_fixed == result.p_max == result.p_frozen_winner_invalid == 1


def test_extreme_distance_scales_do_not_square_before_scaling():
    for scale in (1e-200, 1e200):
        kernel = rbf_kernels([0, scale], [scale])
        assert kernel[0, 0, 1] == pytest.approx(math.exp(-0.5))
    with pytest.raises(ValueError, match="overflow"):
        rbf_kernels([-1e308, 1e308], [1])


def test_exact_assignment_counts_unique_and_observed_first():
    for n, m in ((4, 2), (7, 2), (8, 4)):
        masks = exact_assignments(n, m)
        assert len(masks) == math.comb(n, m)
        assert np.all(masks.sum(axis=1) == m)
        assert len({tuple(row) for row in masks}) == len(masks)
        np.testing.assert_array_equal(masks[0], [True] * m + [False] * (n - m))


def test_exhaustive_cap_checked_before_allocation():
    with pytest.raises(ValueError, match="above cap"):
        exact_assignments(1000, 500)


def test_random_draws_keep_duplicates_and_repeat_for_a_seed():
    masks = random_assignments(4, 2, draws=100, seed=91)
    np.testing.assert_array_equal(masks, random_assignments(4, 2, 100, 91))
    assert len(masks) == 101
    assert len({tuple(row) for row in masks}) < len(masks)
    assert np.all(masks.sum(axis=1) == 2)


def test_inclusive_tail_ranks_and_numerical_ties():
    np.testing.assert_array_equal(upper_tail_ranks([3, 1, 3]), [2 / 3, 1, 2 / 3])
    near = [1.0, np.nextafter(1.0, 0.0), 0.0]
    np.testing.assert_array_equal(upper_tail_ranks(near), [2 / 3, 2 / 3, 1])
    assert upper_tail_ranks(near, tolerance=0)[0] == 1 / 3
    np.testing.assert_array_equal(upper_tail_ranks(np.ones((5, 3))), np.ones((5, 3)))


def test_monte_carlo_plus_one_never_returns_zero():
    result = permutation_test(np.zeros(15), np.ones(15), bandwidths=[0.1], draws=19, seed=42)
    assert result.reference_rows == 20
    assert result.p_fixed == result.p_max == result.p_frozen_winner_invalid == 1 / 20


def test_one_kernel_has_no_selection_penalty_and_swapped_groups_agree():
    x, y = [-2, -1, 0], [0.1, 0.5, 1, 2]
    result = permutation_test(x, y, bandwidths=[1], exact=True)
    swapped = permutation_test(y, x, bandwidths=[1], exact=True)
    assert result.p_fixed == result.p_max == result.p_frozen_winner_invalid
    assert result.p_max == swapped.p_max


def test_complete_orbit_max_is_superuniform_but_frozen_winner_is_not():
    z = np.array([-2.5, -2, -0.4, 0, 0.2, 0.8, 2, 4])
    scores = mmd_scores(rbf_kernels(z, pooled_bandwidths(z)), exact_assignments(8, 4))
    fixed = upper_tail_ranks(scores)[:, 3]
    maximum = upper_tail_ranks(scores.max(axis=1))
    frozen = upper_tail_ranks(scores)[np.arange(70), scores.argmax(axis=1)]
    for pvalues in (fixed, maximum):
        for alpha in np.unique(pvalues):
            assert np.mean(pvalues <= alpha) <= alpha + 1e-14
    assert np.count_nonzero(maximum <= 0.1) == 6
    assert np.count_nonzero(frozen <= 0.1) == 12


def test_monte_carlo_rank_calibration_by_enumerating_all_reference_draws():
    # Six equally likely observed labelings and two IID reference labelings:
    # enumerate 6**3 cases, not a stochastic assertion about a sampled size.
    kernel = np.outer([-2, 0, 1, 4], [-2, 0, 1, 4])
    masks = exact_assignments(4, 2)
    stats = [fraction_score(kernel, row) for row in masks]
    rejections = 0
    for a, b, c in product(stats, repeat=3):
        p = Fraction(1 + int(b >= a) + int(c >= a), 3)
        rejections += p <= Fraction(1, 3)
    assert rejections <= len(stats) ** 3 // 3


@pytest.mark.parametrize("samples,widths", [
    ([0], [1]), ([0, np.nan], [1]), ([0, 1], [0]), ([0, 1], [-1]),
    ([0, 1], [np.inf]), ([0, 1], []), ([0, 1], [[1]]),
    ([True, False], [1]), ([0j, 1j], [1]), (["0", "1"], [1]),
])
def test_bad_kernel_inputs(samples, widths):
    with pytest.raises(ValueError):
        rbf_kernels(samples, widths)


@pytest.mark.parametrize("total,m", [(3, 2), (5, 1), (5, 4), (6.0, 2), (6, True)])
def test_bad_group_sizes(total, m):
    with pytest.raises(ValueError):
        exact_assignments(total, m)


@pytest.mark.parametrize("values,tolerance", [([], 0), ([np.nan], 0), ([1], -1),
                                              ([1], np.inf), ([1], True), ([1], 1j)])
def test_bad_rank_inputs(values, tolerance):
    with pytest.raises(ValueError):
        upper_tail_ranks(values, tolerance)


def test_bad_gram_shapes_asymmetry_assignments_and_overflow():
    mask = np.array([[True, True, False, False]])
    bad_grams = [np.ones((2, 3)), np.ones((4,)), np.ones((0, 4, 4)), np.eye(4) * np.nan]
    asymmetric = np.eye(4)
    asymmetric[0, 1] = 1
    bad_grams.append(asymmetric)
    for gram in bad_grams:
        with pytest.raises(ValueError):
            mmd_scores(gram, mask)
    for masks in (mask.astype(int), mask[0], mask[:, :3], np.zeros((0, 4), dtype=bool),
                  np.array([[True, True, False, False], [True, False, False, False]])):
        with pytest.raises(ValueError):
            mmd_scores(np.eye(4), masks)
    with pytest.raises(ValueError, match="overflow"):
        mmd_scores(np.full((4, 4), 1e308), mask)


@pytest.mark.parametrize("kwargs", [{"fixed_index": 7}, {"fixed_index": -1}, {"draws": 0},
                                   {"seed": -1}, {"exact": 1}, {"fixed_index": 1.2}])
def test_bad_test_options(kwargs):
    with pytest.raises(ValueError):
        permutation_test([0, 1, 2], [1, 2, 3], **kwargs)


def test_incompatible_feature_dimensions_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        permutation_test([[0, 1], [1, 2]], [1, 2])
