import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mnist_lipschitz.distance import euclidean_distance_fn
from mnist_lipschitz.run_experiment import class_separation_ratio, run_class_separation_check


def test_class_separation_ratio_matches_hand_computed_means():
    """4 points on a line, labels [0, 0, 1, 1], plain Euclidean distance -- both within-class
    pairs and both between-class-pair distances are exactly computable by hand, so this checks
    class_separation_ratio's within/between split and ratio directly against a scalar oracle
    rather than trusting the vectorized triu_indices/boolean-mask implementation blindly."""
    x = torch.tensor([[0.0], [1.0], [4.0], [6.0]])
    y = torch.tensor([0, 0, 1, 1])

    result = class_separation_ratio(x, y, euclidean_distance_fn)

    # within-class pairs: (0,1) dist=1, (2,3) dist=2 -> mean 1.5
    # between-class pairs: (0,2)=4, (0,3)=6, (1,2)=3, (1,3)=5 -> mean 4.5
    assert abs(result["within_mean"] - 1.5) < 1e-10
    assert abs(result["between_mean"] - 4.5) < 1e-10
    assert abs(result["ratio"] - 3.0) < 1e-10
    assert result["n_within"] == 2
    assert result["n_between"] == 4


def test_class_separation_ratio_within_and_between_means_computed_over_correct_pair_sets():
    """5 points, labels [0, 0, 0, 1, 1] -- checks the within/between split scales correctly to
    more than 2 pairs per class (3 within-class pairs among the label-0 points, 6 between-class
    pairs), not just the minimal 2-vs-2 case above."""
    x = torch.tensor([[0.0], [1.0], [3.0], [10.0], [12.0]])
    y = torch.tensor([0, 0, 0, 1, 1])

    result = class_separation_ratio(x, y, euclidean_distance_fn)

    # within: (0,1)=1, (0,2)=3, (1,2)=2, (3,4)=2 -> mean 2.0
    # between: (0,3)=10, (0,4)=12, (1,3)=9, (1,4)=11, (2,3)=7, (2,4)=9 -> mean 9.6666...
    assert abs(result["within_mean"] - 2.0) < 1e-10
    assert abs(result["between_mean"] - 58.0 / 6.0) < 1e-10
    assert result["n_within"] == 4
    assert result["n_between"] == 6


def test_run_class_separation_check_runs_all_four_metrics_and_ranks_as_expected():
    """Light real-data smoke test (small n_points, not the full n_points=300 production run):
    checks the driver wires up all 4 metrics, returns finite positive ratios for each, and
    reproduces the qualitative ranking already verified in a full n_points=300 run (recorded in
    README.md/notebook_smoothing.ipynb) -- smoothed cross-terms + Euclidean beats raw-pixel
    Euclidean, and Mahalanobis is the weakest of the four. Not a bit-exact check (small subsample,
    different n_points), just a same-ballpark sanity check on the actual computation."""
    torch.manual_seed(0)
    out = run_class_separation_check(n_points=60, seed=0, verbose=False)
    results = out["results"]

    expected_metrics = {
        "euclidean_raw_pixels", "cross_terms_unsmoothed_euclidean",
        "cross_terms_sigma1_euclidean", "cross_terms_sigma1_mahalanobis",
    }
    assert set(results.keys()) == expected_metrics

    for name, r in results.items():
        assert torch.isfinite(torch.tensor(r["ratio"])), f"{name}: non-finite ratio"
        assert r["ratio"] > 0, f"{name}: non-positive ratio"
        assert r["within_mean"] > 0, f"{name}: non-positive within_mean"

    # Mahalanobis is the weakest class-separator of the four (README.md's class-separation table).
    mahalanobis_ratio = results["cross_terms_sigma1_mahalanobis"]["ratio"]
    for name, r in results.items():
        if name != "cross_terms_sigma1_mahalanobis":
            assert mahalanobis_ratio < r["ratio"], (
                f"expected mahalanobis ({mahalanobis_ratio}) to be the weakest separator, "
                f"but {name} was lower ({r['ratio']})")
