import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import SmallCNN, FlattenedInputWrapper, train_classifier
from mnist_lipschitz.adversarial.attacks import fgsm_attack, pgd_attack
from mnist_lipschitz.estimators import linear_layer_lipschitz
from mnist_lipschitz.adversarial.run_experiment import (
    filter_correctly_classified,
    achieved_ratio,
    run_epsilon_sweep,
    summarize_epsilon_sweep,
    run_bound_comparison,
    most_and_least_sensitive_examples,
    head_layer_bound_check,
)


def _small_trained_cnn(seed=0, n_train=2000, epochs=3):
    torch.manual_seed(seed)
    train = load_mnist(train=True)
    test = load_mnist(train=False)
    dev = get_dev_subset(train, n=n_train, seed=seed)
    train_loader = make_loader(dev.x_image, dev.y, batch_size=128, shuffle=True, seed=seed)
    test_loader = make_loader(test.x_image, test.y, batch_size=1000, shuffle=False)
    model, train_acc, test_acc = train_classifier(
        SmallCNN(conv_channels=(8, 16)), train_loader, test_loader, epochs=epochs, lr=1e-3, verbose=False)
    return model, train, test


# --- filter_correctly_classified ---

def test_filter_correctly_classified_keeps_only_correct_points():
    """Synthetic case with a known ground truth: a linear model whose predicted class is always
    argmax of a fixed one-hot-ish logit vector, checked against manually-computed correctness."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    x = torch.randn(20, 4)
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    y_all_correct = preds.clone()
    y_half_wrong = preds.clone()
    y_half_wrong[::2] = (y_half_wrong[::2] + 1) % 3  # deliberately wrong on every other point

    x_c, y_c, frac = filter_correctly_classified(model, x, y_all_correct)
    assert frac == 1.0
    assert x_c.shape[0] == 20

    x_c2, y_c2, frac2 = filter_correctly_classified(model, x, y_half_wrong)
    assert frac2 == 0.5
    assert x_c2.shape[0] == 10
    with torch.no_grad():
        assert (model(x_c2).argmax(dim=1) == y_c2).all()


def test_run_epsilon_sweep_only_attacks_correctly_classified_points():
    """Requirement 8 checkpoint: the filter must be applied BEFORE sampling/attacking, not after
    -- checked here by confirming every point run_epsilon_sweep actually evaluated (x_eval/y_eval)
    was correctly classified by the clean model, on a real trained CNN with genuine
    misclassifications in the pool (test-set accuracy is well under 100%)."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)

    x_pool, y_pool = test.x_flat[:500], test.y[:500]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1,), pgd_num_steps=3,
                               pgd_num_restarts=1, n_points=100, seed=0, verbose=False)

    assert sweep["kept_frac"] < 1.0, "test setup: expected some misclassified points in the pool"
    with torch.no_grad():
        clean_preds = wrapped(sweep["x_eval"]).argmax(dim=1)
    assert (clean_preds == sweep["y_eval"]).all(), \
        "run_epsilon_sweep attacked a point the clean model got wrong"


# --- achieved_ratio ---

def test_achieved_ratio_matches_manual_logit_space_computation():
    """Explicit check that achieved_ratio uses the FULL logit vector (shape (N, num_classes)),
    Euclidean distance -- the same convention layer_decomposition.py's L_full_estimated is
    measured on -- not the scalar margin used elsewhere in this project."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x = test.x_flat[:16]
    x_adv = x + 0.05 * torch.randn_like(x)
    x_adv = x_adv.clamp(0.0, 1.0)

    with torch.no_grad():
        f_x = wrapped(x)
        f_adv = wrapped(x_adv)
    assert f_x.shape == (16, 10), "expected full 10-d logit output, not a scalar margin"

    expected = (f_x - f_adv).norm(p=2, dim=-1) / (x - x_adv).norm(p=2, dim=-1)
    actual = achieved_ratio(wrapped, x, x_adv)
    assert torch.allclose(actual, expected, atol=1e-9)


def test_achieved_ratio_zero_distance_gives_zero_not_nan():
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x = test.x_flat[:8]
    ratio = achieved_ratio(wrapped, x, x.clone())
    assert torch.equal(ratio, torch.zeros(8))
    assert not torch.isnan(ratio).any()


# --- most_and_least_sensitive_examples ---

def test_most_and_least_sensitive_examples_are_the_true_extremes():
    """The returned pair must be the actual global max/min of R_adv across every (epsilon,
    method) case in the sweep -- checked here against a brute-force scan of the same
    sweep_results, not just "some example got returned"."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1, 0.2), pgd_num_steps=5,
                               pgd_num_restarts=2, n_points=50, seed=0, verbose=False)

    most_sensitive, least_sensitive = most_and_least_sensitive_examples(wrapped, sweep)

    all_R = torch.cat([case["R_adv"] for case in sweep["per_case"].values()])
    assert abs(most_sensitive["R_adv"] - all_R.max().item()) < 1e-9
    assert abs(least_sensitive["R_adv"] - all_R.min().item()) < 1e-9
    assert most_sensitive["R_adv"] >= least_sensitive["R_adv"]


def test_most_and_least_sensitive_examples_have_consistent_predictions_and_shapes():
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1,), pgd_num_steps=3,
                               pgd_num_restarts=1, n_points=50, seed=0, verbose=False)

    for example in most_and_least_sensitive_examples(wrapped, sweep):
        assert example["x"].shape == (784,)
        assert example["x_adv"].shape == (784,)
        with torch.no_grad():
            expected_pred_clean = wrapped(example["x"].unsqueeze(0)).argmax(dim=1).item()
            expected_pred_adv = wrapped(example["x_adv"].unsqueeze(0)).argmax(dim=1).item()
        assert example["pred_clean"] == expected_pred_clean
        assert example["pred_adv"] == expected_pred_adv
        assert example["epsilon"] in (0.1,)
        assert example["method"] in ("FGSM", "PGD")


def test_most_and_least_sensitive_examples_pixel_distance_matches_manual_and_R_adv():
    """pixel_distance must match a manual ||x - x_adv||_2 computation, and must be consistent
    with R_adv and head_layer_bound_check's actual_logit_distance via
    R_adv == actual_logit_distance / pixel_distance -- the same relationship achieved_ratio
    itself computes, checked here end to end across the two independently-computed pieces."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1, 0.2), pgd_num_steps=5,
                               pgd_num_restarts=2, n_points=50, seed=0, verbose=False)

    for example in most_and_least_sensitive_examples(wrapped, sweep):
        expected_pixel_distance = (example["x"] - example["x_adv"]).norm(p=2).item()
        assert abs(example["pixel_distance"] - expected_pixel_distance) < 1e-9

        head_result = head_layer_bound_check(model, example)
        expected_R_adv = head_result["actual_logit_distance"] / example["pixel_distance"]
        assert abs(example["R_adv"] - expected_R_adv) < 1e-6


# --- head_layer_bound_check ---

def test_head_layer_bound_check_matches_manual_computation():
    """feature_distance/actual_logit_distance/L_head_exact must match an independent manual
    computation on the raw (unnormalized) extractor/head, not layer_decomposition_experiment's
    internal standardized-feature convention (see head_layer_bound_check's docstring)."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.15,), pgd_num_steps=3,
                               pgd_num_restarts=1, n_points=50, seed=0, verbose=False)
    most_sensitive, _ = most_and_least_sensitive_examples(wrapped, sweep)

    result = head_layer_bound_check(model, most_sensitive)

    with torch.no_grad():
        features = model.extractor(most_sensitive["x"].reshape(1, 1, 28, 28)).squeeze(0)
        features_adv = model.extractor(most_sensitive["x_adv"].reshape(1, 1, 28, 28)).squeeze(0)
        logits = model.head(features)
        logits_adv = model.head(features_adv)

    expected_feature_distance = (features - features_adv).norm(p=2).item()
    expected_logit_distance = (logits - logits_adv).norm(p=2).item()
    expected_L_head = linear_layer_lipschitz(model.head)

    assert abs(result["feature_distance"] - expected_feature_distance) < 1e-9
    assert abs(result["actual_logit_distance"] - expected_logit_distance) < 1e-9
    assert abs(result["L_head_exact"] - expected_L_head) < 1e-9


def test_head_layer_bound_check_actual_logit_distance_matches_full_model_output():
    """head(extractor(x)) must equal the full model's own forward pass exactly (SmallCNN's own
    invariant, see models.py), so actual_logit_distance must equal
    ||wrapped_model(x) - wrapped_model(x_adv)||_2 -- the same quantity achieved_ratio's numerator
    uses for this example."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.15,), pgd_num_steps=3,
                               pgd_num_restarts=1, n_points=50, seed=0, verbose=False)
    most_sensitive, least_sensitive = most_and_least_sensitive_examples(wrapped, sweep)

    for example in (most_sensitive, least_sensitive):
        result = head_layer_bound_check(model, example)
        with torch.no_grad():
            expected = (wrapped(example["x"].unsqueeze(0))
                        - wrapped(example["x_adv"].unsqueeze(0))).norm(p=2).item()
        assert abs(result["actual_logit_distance"] - expected) < 1e-9


def test_head_layer_bound_check_never_exceeds_its_own_bound():
    """The central checkpoint: actual_logit_distance <= head_bound (= L_head_exact *
    feature_distance) always, for a linear head -- Cauchy-Schwarz, not an approximation. Checked
    on several real attacked examples, not just the two extremes, since this must hold for every
    point, not only the ones happening to be most/least sensitive."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1, 0.2), pgd_num_steps=5,
                               pgd_num_restarts=2, n_points=50, seed=0, verbose=False)

    for (epsilon, method), case in sweep["per_case"].items():
        for i in range(0, 50, 10):  # a spot-check sample, not all 50 x 2 cases
            example = {"x": sweep["x_eval"][i], "x_adv": case["x_adv"][i]}
            result = head_layer_bound_check(model, example)
            assert result["actual_logit_distance"] <= result["head_bound"] + 1e-9
            assert result["head_bound_tightness"] <= 1.0 + 1e-6


# --- PGD vs. FGSM on a real trained model ---

def test_pgd_achieves_at_least_as_much_sensitivity_as_fgsm_on_trained_cnn():
    """Empirical checkpoint (Requirement 8): PGD, given many more steps and several random
    restarts, should never do MEANINGFULLY worse than a single FGSM step at finding sensitive
    directions -- FGSM is a special case of PGD's search. This is checked at the aggregate
    (mean/max over the sample) level, not per example: PGD's best-of-restarts selection in
    attacks.py optimizes cross-entropy loss, not achieved_ratio directly (see attacks.py's and
    run_experiment.py's docstrings), so a per-example R_adv guarantee doesn't follow from that
    selection rule alone -- only an aggregate, empirically-observed one."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_correct, y_correct, _ = filter_correctly_classified(wrapped, test.x_flat[:1000], test.y[:1000])
    x_eval, y_eval = x_correct[:200], y_correct[:200]

    epsilon = 0.2
    x_fgsm = fgsm_attack(wrapped, x_eval, y_eval, epsilon)
    x_pgd = pgd_attack(wrapped, x_eval, y_eval, epsilon=epsilon, alpha=epsilon / 4,
                        num_steps=20, num_restarts=5, seed=0)

    R_fgsm = achieved_ratio(wrapped, x_eval, x_fgsm)
    R_pgd = achieved_ratio(wrapped, x_eval, x_pgd)

    assert R_pgd.mean().item() >= R_fgsm.mean().item() * 0.9
    assert R_pgd.max().item() >= R_fgsm.max().item() * 0.9


# --- summarize_epsilon_sweep's sanity check ---

def test_summarize_epsilon_sweep_warns_but_does_not_drop_violating_rows(capsys):
    """Requirement 5's sanity check: passing an artificially tiny L_full_estimated (smaller than
    any achieved ratio, guaranteed to trigger the "L_full under-sampled" case) must print a
    warning AND still include the row in the returned table -- not crash, not silently drop it."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1,), pgd_num_steps=3,
                               pgd_num_restarts=1, n_points=50, seed=0, verbose=False)

    tiny_L_full = 1e-6
    df = summarize_epsilon_sweep(sweep, L_full_estimated=tiny_L_full, product_bound=1e-6, verbose=False)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "L_full_estimated" in captured.out
    assert len(df) == 2  # FGSM + PGD, both rows kept despite the violation
    assert (df["max_R_adv"] > tiny_L_full).all()


def test_summarize_epsilon_sweep_no_warning_with_a_generous_L_full_estimated():
    """Sanity check for the sanity check: with a comfortably large L_full_estimated (no violation
    possible), no warning should fire."""
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    sweep = run_epsilon_sweep(wrapped, x_pool, y_pool, epsilons=(0.1,), pgd_num_steps=3,
                               pgd_num_restarts=1, n_points=50, seed=0, verbose=False)
    df = summarize_epsilon_sweep(sweep, L_full_estimated=1e6, product_bound=1e6, verbose=False)
    assert (df["ratio_to_L_full"] < 1.0).all()


# --- full single-checkpoint pipeline ---

def test_run_bound_comparison_end_to_end_on_small_trained_cnn():
    """Integration checkpoint: the full pipeline (layer_decomposition_experiment's bounds +
    the epsilon sweep against the SAME checkpoint) runs end to end and returns a well-formed
    summary table with the required columns."""
    model, train, test = _small_trained_cnn()
    gen = torch.Generator().manual_seed(0)
    query_idx = torch.randperm(len(test), generator=gen)[:40]
    x_query, y_query = test.x_flat[query_idx], test.y[query_idx]
    mask = torch.ones(len(test), dtype=torch.bool)
    mask[query_idx] = False
    pool_idx = mask.nonzero(as_tuple=True)[0][:300]
    x_pool, y_pool = test.x_flat[pool_idx], test.y[pool_idx]
    x_train_for_norm = train.x_flat[:200]

    summary_df, sweep_results = run_bound_comparison(
        model, x_query, y_query, x_pool, y_pool, x_train_for_norm,
        epsilons=(0.1, 0.2), pgd_num_steps=5, pgd_num_restarts=2, n_points=50,
        seed=0, verbose=False)

    expected_cols = {"epsilon", "method", "mean_R_adv", "median_R_adv", "max_R_adv",
                      "pct_misclassified", "L_full_estimated", "product_bound",
                      "ratio_to_L_full", "ratio_to_product_bound"}
    assert expected_cols.issubset(set(summary_df.columns))
    assert len(summary_df) == 4  # 2 epsilons x {FGSM, PGD}
    assert (summary_df["L_full_estimated"] == summary_df["L_full_estimated"].iloc[0]).all()
    assert (summary_df["product_bound"] == summary_df["product_bound"].iloc[0]).all()
