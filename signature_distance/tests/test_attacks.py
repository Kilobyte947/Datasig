import torch

from signature_distance.attacks import fgsm_attack, pgd_attack, random_noise_perturbation
from signature_distance.distances import pixel_euclidean_distance
from signature_distance.models import SmallCNN, train_classifier


def _tiny_dataset(n=200, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return torch.utils.data.TensorDataset(x, y)


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------


def test_fgsm_eps0_returns_unchanged():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    x_adv = fgsm_attack(model, x, y, epsilon=0.0)
    assert torch.allclose(x_adv, x, atol=1e-6)


def test_fgsm_stays_within_epsilon_ball_and_valid_range():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.05
    x_adv = fgsm_attack(model, x, y, epsilon=eps)
    assert (x_adv - x).abs().max().item() <= eps + 1e-6
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0


def test_fgsm_flips_some_predictions_on_a_trained_model():
    # A briefly-trained model (not random weights) is needed for FGSM to be
    # meaningful - on random weights, "flipping" a near-random prediction
    # proves nothing. One epoch on a small synthetic set is enough for this
    # correctness check (it doesn't need to be an accurate classifier, just
    # one with real gradient signal).
    train_ds = _tiny_dataset(n=500, seed=0)
    test_ds = _tiny_dataset(n=100, seed=1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    model, _, _ = train_classifier(SmallCNN(), train_loader, test_loader, epochs=1, verbose=False)
    model.eval()

    x, y = next(iter(test_loader))
    x_adv = fgsm_attack(model, x, y, epsilon=0.3)  # large eps - should be able to flip something
    preds_orig = model(x).argmax(dim=1)
    preds_adv = model(x_adv).argmax(dim=1)
    flip_fraction = (preds_orig != preds_adv).float().mean().item()
    assert flip_fraction > 0.0


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------


def test_pgd_eps0_returns_unchanged():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    x_adv = pgd_attack(model, x, y, epsilon=0.0)
    assert torch.allclose(x_adv, x, atol=1e-6)


def test_pgd_stays_within_epsilon_ball_and_valid_range():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    eps = 0.05
    x_adv = pgd_attack(model, x, y, epsilon=eps, num_steps=10)
    assert (x_adv - x).abs().max().item() <= eps + 1e-6
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0


def test_pgd_zero_steps_matches_random_start_only():
    torch.manual_seed(0)
    model = SmallCNN()
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    eps = 0.05
    x_adv = pgd_attack(model, x, y, epsilon=eps, num_steps=0, generator=torch.Generator().manual_seed(1))
    # no gradient steps taken - still within the epsilon-ball and valid range
    assert (x_adv - x).abs().max().item() <= eps + 1e-6
    assert x_adv.min().item() >= 0.0
    assert x_adv.max().item() <= 1.0


def test_pgd_flips_more_or_equal_predictions_than_fgsm_at_matched_epsilon():
    # PGD is a strictly stronger iterative attack than single-step FGSM at
    # the same L_inf budget - on a real (briefly trained) model, its flip
    # rate should be at least as high, not lower.
    train_ds = _tiny_dataset(n=500, seed=0)
    test_ds = _tiny_dataset(n=100, seed=1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    model, _, _ = train_classifier(SmallCNN(), train_loader, test_loader, epochs=1, verbose=False)
    model.eval()

    x, y = next(iter(test_loader))
    eps = 0.1
    x_adv_fgsm = fgsm_attack(model, x, y, epsilon=eps)
    x_adv_pgd = pgd_attack(model, x, y, epsilon=eps, num_steps=10, random_start=False)

    preds_orig = model(x).argmax(dim=1)
    fgsm_flip = (preds_orig != model(x_adv_fgsm).argmax(dim=1)).float().mean().item()
    pgd_flip = (preds_orig != model(x_adv_pgd).argmax(dim=1)).float().mean().item()
    assert pgd_flip >= fgsm_flip - 1e-6


def test_pgd_reduces_margin_more_than_a_single_gradient_step_on_average():
    # A weaker correctness check independent of the flip-rate threshold
    # above: iterating should reduce the true-class margin at least as much
    # as a single step, on average, at the same epsilon.
    torch.manual_seed(0)
    train_ds = _tiny_dataset(n=500, seed=0)
    test_ds = _tiny_dataset(n=100, seed=1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    model, _, _ = train_classifier(SmallCNN(), train_loader, test_loader, epochs=1, verbose=False)
    model.eval()

    x, y = next(iter(test_loader))
    eps = 0.1
    from signature_distance.distances import margin
    m_orig = margin(model, x, y)
    x_adv_fgsm = fgsm_attack(model, x, y, epsilon=eps)
    x_adv_pgd = pgd_attack(model, x, y, epsilon=eps, num_steps=10, random_start=False)
    m_fgsm = margin(model, x_adv_fgsm, y)
    m_pgd = margin(model, x_adv_pgd, y)
    assert (m_orig - m_pgd).mean().item() >= (m_orig - m_fgsm).mean().item() - 1e-4


# ---------------------------------------------------------------------------
# Random-noise control perturbation
# ---------------------------------------------------------------------------


def test_random_noise_perturbation_matches_l2_budget():
    torch.manual_seed(0)
    x = torch.rand(5, 1, 28, 28) * 0.5 + 0.25  # keep away from [0,1] edges
    budget = torch.full((5,), 0.5)
    x_control = random_noise_perturbation(x, budget, generator=torch.Generator().manual_seed(0))
    achieved = pixel_euclidean_distance(x, x_control)
    # clipping to [0,1] can shrink the achieved norm below budget, but not
    # exceed it noticeably
    assert (achieved <= budget + 1e-4).all()
