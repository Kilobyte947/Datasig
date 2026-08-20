import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.data import load_mnist, get_dev_subset, make_loader
from mnist_lipschitz.models import SmallCNN, FlattenedInputWrapper, train_classifier, LogisticRegressionModel
from mnist_lipschitz.estimators import euclidean_distance_fn, pairwise_lipschitz
from mnist_lipschitz.adversarial.attacks import fgsm_attack
from mnist_lipschitz.adversarial.run_experiment import (
    filter_correctly_classified,
    adversarial_accuracy,
    clean_logit_stats,
    margin_lipschitz_estimate,
    flip_direction_alignment,
)

TOLERANCE = 0.10  # matches mnist_lipschitz/tests/test_estimators.py's checkpoint tolerance


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


# --- adversarial_accuracy ---

def test_adversarial_accuracy_epsilon_zero_gives_zero_misclassification_rate():
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    x_correct, y_correct, _ = filter_correctly_classified(wrapped, x_pool, y_pool)
    x_eval, y_eval = x_correct[:50], y_correct[:50]

    x_adv = fgsm_attack(wrapped, x_eval, y_eval, epsilon=0.0)
    result = adversarial_accuracy(wrapped, x_eval, x_adv, y_eval)

    assert result["clean_acc"] == 1.0
    assert result["misclassification_rate"] == 0.0
    assert result["n_flipped"] == 0
    assert result["n_evaluated"] == 50
    assert result["adv_acc"] == 1.0


def test_adversarial_accuracy_counts_match_manual_computation():
    model, train, test = _small_trained_cnn()
    wrapped = FlattenedInputWrapper(model)
    x_pool, y_pool = test.x_flat[:300], test.y[:300]
    x_correct, y_correct, _ = filter_correctly_classified(wrapped, x_pool, y_pool)
    x_eval, y_eval = x_correct[:50], y_correct[:50]

    x_adv = fgsm_attack(wrapped, x_eval, y_eval, epsilon=0.2)
    result = adversarial_accuracy(wrapped, x_eval, x_adv, y_eval)

    with torch.no_grad():
        pred_adv = wrapped(x_adv).argmax(dim=1)
    expected_flipped = (pred_adv != y_eval).sum().item()
    assert result["n_flipped"] == expected_flipped
    assert abs(result["misclassification_rate"] - expected_flipped / 50) < 1e-12


# --- clean_logit_stats ---

def test_clean_logit_stats_hand_constructed_2class_logits():
    logits = torch.tensor([
        [1.0, 4.0],   # top2 = [4,1], margin = 3
        [0.0, 0.0],   # margin = 0
        [-2.0, 3.0],  # margin = 5
    ])

    def fake_model(x):
        return logits

    x_dummy = torch.zeros(3, 2)
    result = clean_logit_stats(fake_model, x_dummy)

    expected_margins = torch.tensor([3.0, 0.0, 5.0])
    expected_norms = logits.norm(p=2, dim=-1)

    assert abs(result["mean_top2_margin"] - expected_margins.mean().item()) < 1e-9
    assert abs(result["std_top2_margin"] - expected_margins.std().item()) < 1e-9
    assert abs(result["mean_logit_norm"] - expected_norms.mean().item()) < 1e-9
    assert abs(result["std_logit_norm"] - expected_norms.std().item()) < 1e-9
    assert abs(result["p5_top2_margin"] - expected_margins.quantile(0.05).item()) < 1e-9
    assert abs(result["p10_top2_margin"] - expected_margins.quantile(0.10).item()) < 1e-9


# --- margin_lipschitz_estimate ---

def test_margin_lipschitz_estimate_matches_closed_form_on_linear_model():
    """On a 2-class logistic regression, the margin is exactly linear in x, with closed-form
    Lipschitz constant ||w_0 - w_1||_2 -- same checkpoint pattern as
    mnist_lipschitz/tests/test_estimators.py's own closed-form check, applied to this module's
    wrapper around the same underlying estimator machinery."""
    d = 5
    torch.manual_seed(0)
    model = LogisticRegressionModel(input_dim=d, num_classes=2)
    w_diff = (model.linear.weight[0] - model.linear.weight[1]).detach()
    L_star = w_diff.norm(p=2).item()

    torch.manual_seed(1)
    x_batch = torch.randn(200, d)
    y_batch = torch.zeros(200, dtype=torch.int64)

    L_hat = margin_lipschitz_estimate(model, x_batch, y_batch, estimator="pairwise",
                                       distance_fn=euclidean_distance_fn, seed=0)

    assert L_hat <= L_star * 1.01
    rel_err = abs(L_hat - L_star) / L_star
    assert rel_err < TOLERANCE, f"margin_lipschitz_estimate rel_err={rel_err:.4f}"


def test_margin_lipschitz_estimate_unknown_estimator_raises():
    d = 5
    torch.manual_seed(0)
    model = LogisticRegressionModel(input_dim=d, num_classes=2)
    x_batch = torch.randn(10, d)
    y_batch = torch.zeros(10, dtype=torch.int64)
    try:
        margin_lipschitz_estimate(model, x_batch, y_batch, estimator="not_a_method")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- flip_direction_alignment ---

def test_flip_direction_alignment_perfectly_aligned_gives_plus_one():
    """dz constructed exactly along e_k - e_y (k=1, y=0, num_classes=3): cosine must be +1.0."""
    logits_clean = torch.tensor([[1.0, 0.5, 0.2]])  # y=0's runner-up is class 1 (0.5 > 0.2)
    logits_adv = logits_clean + torch.tensor([[-1.0, 1.0, 0.0]])  # dz = e_1 - e_0 exactly

    def fake_model_factory(logits_to_return):
        def _fn(x):
            return logits_to_return
        return _fn

    calls = {"n": 0}
    outputs = [logits_clean, logits_adv]

    def fake_model(x):
        out = outputs[calls["n"]]
        calls["n"] += 1
        return out

    x = torch.zeros(1, 2)
    x_adv = torch.zeros(1, 2)
    y = torch.tensor([0])

    result = flip_direction_alignment(fake_model, x, x_adv, y)
    assert abs(result["mean_cosine_alignment"] - 1.0) < 1e-9


def test_flip_direction_alignment_orthogonal_gives_zero():
    """dz orthogonal to e_k - e_y (k=1, y=0): cosine must be 0.0. e_k - e_y = (-1, 1, 0); an
    orthogonal dz that also has nonzero norm is e.g. (1, 1, 0) (dot product with (-1,1,0) is 0)."""
    logits_clean = torch.tensor([[1.0, 0.5, 0.2]])
    logits_adv = logits_clean + torch.tensor([[1.0, 1.0, 0.0]])

    outputs = [logits_clean, logits_adv]
    calls = {"n": 0}

    def fake_model(x):
        out = outputs[calls["n"]]
        calls["n"] += 1
        return out

    x = torch.zeros(1, 2)
    x_adv = torch.zeros(1, 2)
    y = torch.tensor([0])

    result = flip_direction_alignment(fake_model, x, x_adv, y)
    assert abs(result["mean_cosine_alignment"] - 0.0) < 1e-9


def test_flip_direction_alignment_zero_movement_gives_zero_not_nan():
    logits = torch.tensor([[1.0, 0.5, 0.2]])
    y = torch.tensor([0])

    def fake_model(x):
        return logits

    result = flip_direction_alignment(fake_model, torch.zeros(1, 2), torch.zeros(1, 2), y)
    assert result["mean_cosine_alignment"] == 0.0
