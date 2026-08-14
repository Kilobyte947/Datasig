import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from toy_lipschitz.run_experiment import build_tier_b_1d, sweep_over_N, sweep_over_N_seed_averaged, DOMAIN, SEED

# Small/fast settings throughout -- these tests check the seed-averaging
# machinery's shapes and its equivalence to sweep_over_N at n_seeds=1, not
# the real (slow) N_values=(50..5000) sweep used for the actual finding.
N_VALUES = (50, 200)
HIDDEN_SIZES = (16, 16)
EPOCHS = 300
HELD_OUT_GRID = torch.linspace(DOMAIN[0], DOMAIN[1], 300).unsqueeze(-1)


def _setup():
    components, L_star, x_star = build_tier_b_1d()
    return components, L_star, x_star


def test_returns_results_for_every_N_and_every_seed():
    components, L_star, x_star = _setup()
    n_seeds = 3
    result = sweep_over_N_seed_averaged(
        "gap", components, L_star, x_star, HELD_OUT_GRID, N_values=N_VALUES, n_seeds=n_seeds,
        hidden_sizes=HIDDEN_SIZES, epochs=EPOCHS, verbose=False)

    assert result["seeds"] == [SEED, SEED + 1, SEED + 2]
    assert len(result["seeds"]) == n_seeds
    assert result["N_values"].tolist() == list(N_VALUES)
    assert result["L_hat_data_per_seed"].shape == (n_seeds, len(N_VALUES))
    assert result["L_hat_model_per_seed"].shape == (n_seeds, len(N_VALUES))
    # every entry should be a finite, positive Lipschitz estimate
    assert np.isfinite(result["L_hat_model_per_seed"]).all()
    assert (result["L_hat_model_per_seed"] > 0).all()


def test_aggregated_arrays_have_expected_shape():
    components, L_star, x_star = _setup()
    n_seeds = 3
    result = sweep_over_N_seed_averaged(
        "gap", components, L_star, x_star, HELD_OUT_GRID, N_values=N_VALUES, n_seeds=n_seeds,
        hidden_sizes=HIDDEN_SIZES, epochs=EPOCHS, verbose=False)

    for key in ["L_hat_data_mean", "L_hat_data_std", "L_hat_data_min", "L_hat_data_max",
                "L_hat_model_mean", "L_hat_model_std", "L_hat_model_min", "L_hat_model_max"]:
        assert result[key].shape == (len(N_VALUES),), f"{key} has shape {result[key].shape}"

    # sanity: mean must lie within [min, max] at every N
    assert (result["L_hat_model_mean"] >= result["L_hat_model_min"] - 1e-9).all()
    assert (result["L_hat_model_mean"] <= result["L_hat_model_max"] + 1e-9).all()
    # std must be non-negative
    assert (result["L_hat_model_std"] >= 0).all()


def test_n_seeds_1_reproduces_sweep_over_N_exactly():
    """The seed-parameterized sweep_over_N and the seed-averaging wrapper
    around it must agree exactly at n_seeds=1, base_seed=SEED -- both paths
    are fully deterministic given a fixed seed, so this is an exact-equality
    regression check that threading `seed` through sweep_over_N didn't
    change its own single-seed behavior."""
    components, L_star, x_star = _setup()

    direct_data, direct_model = sweep_over_N(
        "gap", components, L_star, x_star, HELD_OUT_GRID, N_values=N_VALUES,
        hidden_sizes=HIDDEN_SIZES, epochs=EPOCHS, seed=SEED, verbose=False)

    result = sweep_over_N_seed_averaged(
        "gap", components, L_star, x_star, HELD_OUT_GRID, N_values=N_VALUES, n_seeds=1, base_seed=SEED,
        hidden_sizes=HIDDEN_SIZES, epochs=EPOCHS, verbose=False)

    assert result["seeds"] == [SEED]
    np.testing.assert_array_equal(result["L_hat_data_per_seed"][0], direct_data)
    np.testing.assert_array_equal(result["L_hat_model_per_seed"][0], direct_model)
    np.testing.assert_array_equal(result["L_hat_data_mean"], direct_data)
    np.testing.assert_array_equal(result["L_hat_model_mean"], direct_model)
    assert (result["L_hat_model_std"] == 0).all()  # single seed -> zero spread
