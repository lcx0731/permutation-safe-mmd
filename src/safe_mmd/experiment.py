"""Exact conditional label-orbit audit and independent synthetic replications."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    exact_assignments, mmd_scores, permutation_test, pooled_bandwidths,
    rbf_kernels, upper_tail_ranks,
)

METHODS = ("fixed_pooled_median", "selection_aware_max", "frozen_winner_invalid")
LABELS = ("Fixed pooled median", "Selection-aware max", "Frozen winner (invalid)")
COLORS = ("#376b8a", "#488168", "#bb604b")


def exact_orbit():
    pooled = np.array([-2.5, -2, -0.4, 0, 0.2, 0.8, 2, 4])
    widths = pooled_bandwidths(pooled)
    masks = exact_assignments(8, 4)
    scores = mmd_scores(rbf_kernels(pooled, widths), masks)
    ranks = upper_tail_ranks(scores)
    winners = scores.argmax(axis=1)
    pvalues = {
        METHODS[0]: ranks[:, 3],
        METHODS[1]: upper_tail_ranks(scores.max(axis=1)),
        METHODS[2]: ranks[np.arange(len(masks)), winners],
    }
    for key in METHODS[:2]:
        for alpha in np.unique(pvalues[key]):
            if np.mean(pvalues[key] <= alpha) > alpha + 1e-14:
                raise ArithmeticError("valid method failed the complete conditional-orbit size check")
    return {
        "pooled_points": pooled.tolist(), "group_sizes": [4, 4],
        "bandwidths": widths.tolist(), "labelings": len(masks),
        "pvalues_by_labeling": {key: values.tolist() for key, values in pvalues.items()},
        "rejection_counts": {str(alpha): {key: int(np.count_nonzero(values <= alpha))
                                            for key, values in pvalues.items()}
                             for alpha in (0.05, 0.1)},
    }


def _binomial_summary(pvalues, alpha):
    count = int(np.count_nonzero(np.asarray(pvalues) <= alpha))
    n = len(pvalues)
    rate = count / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return {"rejections": count, "rate": rate, "mcse": math.sqrt(rate * (1 - rate) / n),
            "wilson_95": [max(0.0, center - half), min(1.0, center + half)]}


def run_experiment(replicates=200, draws=199, seed=20261023):
    if type(replicates) is not int or replicates < 2:
        raise ValueError("replicates must be an integer >= 2")
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    results = []
    alpha = 0.05
    for scenario_id, scenario in enumerate(("equal_distributions", "mean_shift", "scale_shift")):
        pvalues = {method: [] for method in METHODS}
        winner_counts = [0] * 7
        for replicate in range(replicates):
            streams = np.random.SeedSequence([seed, scenario_id, replicate]).spawn(2)
            rng = np.random.default_rng(streams[0])
            permutation_seed = int(streams[1].generate_state(1, dtype=np.uint64)[0])
            x = rng.normal(size=(20, 3))
            y = rng.normal(size=(20, 3))
            if scenario == "mean_shift":
                y[:, 0] += 0.9
            elif scenario == "scale_shift":
                y *= 1.65
            result = permutation_test(x, y, draws=draws, seed=permutation_seed)
            for name, value in zip(METHODS, (result.p_fixed, result.p_max, result.p_frozen_winner_invalid)):
                if not 1 / (draws + 1) <= value <= 1:
                    raise ArithmeticError("Monte Carlo p-value outside its inclusive-rank range")
                pvalues[name].append(value)
            winner_counts[result.selected_index] += 1
        results.append({
            "scenario": scenario, "group_sizes": [20, 20], "features": 3,
            "summary": {name: _binomial_summary(values, alpha) for name, values in pvalues.items()},
            "pvalues": pvalues, "winner_counts": winner_counts,
        })
    return {
        "data": "Seeded synthetic IID observations; no external or personal data.",
        "seed": seed, "replicates_per_scenario": replicates, "random_assignments_per_test": draws,
        "alpha": alpha, "null": "Equality of the complete X and Y distributions, not merely their means.",
        "bandwidth_factors": [0.125, 0.25, 0.5, 1, 2, 4, 8],
        "intervals": "Pointwise Wilson 95% intervals across independent simulation replicates; approximate coverage.",
        "exact_orbit": exact_orbit(), "replicated_results": results,
    }


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    orbit = result["exact_orbit"]
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.7), layout="constrained")
    grid = np.unique(np.concatenate(([0, 1], *[v for v in orbit["pvalues_by_labeling"].values()])))
    for ax in axes:
        for name, label, color in zip(METHODS, LABELS, COLORS):
            values = np.array(orbit["pvalues_by_labeling"][name])
            cdf = np.array([np.mean(values <= alpha) for alpha in grid])
            ax.step(grid, cdf, where="post", color=color, label=label, linewidth=2)
        ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="Nominal level")
        ax.set(xlabel="Nominal rejection level α", ylabel="Fraction of 70 labelings rejected")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), title="Entire conditional label orbit")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set(xlim=(0, .2), ylim=(0, .32), title="Zoom: small rejection levels")
    axes[1].axvline(.1, color="#bbbbbb", linestyle=":")
    axes[1].text(.012, .285, "At α = 0.10: valid max 6/70; frozen winner 12/70", fontsize=9)
    fig.suptitle("Kernel selection must also be repeated under label permutation", fontsize=12)
    fig.savefig(output / "conditional-orbit.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.4), layout="constrained")
    for ax, record in zip(axes, result["replicated_results"]):
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            row = record["summary"][method]
            lower, upper = row["wilson_95"]
            ax.bar(i, row["rate"], color=color, width=.65,
                   yerr=np.array([[max(0, row["rate"] - lower)], [max(0, upper - row["rate"])]]),
                   capsize=4)
            ax.text(i, upper + .012, str(row["rejections"]), ha="center", fontsize=9)
        ax.set(xticks=range(3), xticklabels=["Fixed", "Max", "Frozen\n(invalid)"],
               ylabel="Rejection proportion", title=record["scenario"].replace("_", " "))
        if record["scenario"] == "equal_distributions":
            ax.set_ylim(0, max(.20, max(v["wilson_95"][1] for v in record["summary"].values()) + .05))
            ax.axhline(result["alpha"], color="#666666", linestyle="--", linewidth=1)
        else:
            ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Synthetic experiments · {result['replicates_per_scenario']} independent runs per scenario · "
                 "α = 0.05\nNumbers are rejection counts; intervals are pointwise Wilson 95%", fontsize=11)
    fig.savefig(output / "replicated-tests.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--draws", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    try:
        result = run_experiment(args.replicates, args.draws, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_outputs(result, args.output)
    print(json.dumps({"output": str(args.output), "replicates_per_scenario": args.replicates,
                      "orbit_rejections_at_0.1": result["exact_orbit"]["rejection_counts"]["0.1"]}, indent=2))


if __name__ == "__main__":
    main()
