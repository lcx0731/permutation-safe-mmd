import json
import subprocess
import sys

import pytest

from safe_mmd.experiment import METHODS, _binomial_summary, exact_orbit, run_experiment, write_outputs


def test_exact_orbit_counts():
    orbit = exact_orbit()
    assert orbit["labelings"] == 70
    assert orbit["rejection_counts"]["0.1"] == dict(zip(METHODS, (6, 6, 12)))
    assert orbit["rejection_counts"]["0.05"] == dict(zip(METHODS, (2, 2, 4)))


def test_experiment_repeats_and_counts_match_pvalues():
    result = run_experiment(replicates=4, draws=19, seed=42)
    assert result == run_experiment(replicates=4, draws=19, seed=42)
    for row in result["replicated_results"]:
        assert sum(row["winner_counts"]) == 4
        for name, values in row["pvalues"].items():
            assert row["summary"][name]["rejections"] == sum(p <= .05 for p in values)
            assert min(values) >= 1 / 20


def test_json_and_figures(tmp_path):
    result = run_experiment(replicates=2, draws=9, seed=5)
    write_outputs(result, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    for name in ("conditional-orbit.png", "replicated-tests.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_wilson_intervals_remain_in_range_at_zero_and_all_rejections():
    for pvalues in ([1] * 20, [0.01] * 20):
        result = _binomial_summary(pvalues, .05)
        assert 0 <= result["wilson_95"][0] <= result["rate"] <= result["wilson_95"][1] <= 1


@pytest.mark.parametrize("reps,draws,seed", [(1, 1, 1), (True, 1, 1), (2, 0, 1), (2, 1, -1)])
def test_bad_arguments(reps, draws, seed):
    with pytest.raises(ValueError):
        run_experiment(reps, draws, seed)


def test_cli(tmp_path):
    result = subprocess.run([sys.executable, "-m", "safe_mmd.experiment", "--replicates", "2",
                             "--draws", "9", "--output", str(tmp_path)],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["orbit_rejections_at_0.1"][METHODS[1]] == 6
