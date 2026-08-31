import torch
import torch.nn as nn

from signature_distance.method_b_adversarial_eval import (
    SmallCNN,
    fgsm_attack,
    margin,
    method_b_signature_distance,
    pixel_euclidean_distance,
    random_noise_perturbation,
    train_classifier,
)


def _tiny_dataset(n=200, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return torch.utils.data.TensorDataset(x, y)


def test_margin_matches_hand_computed_logits():
    class FixedLogits(nn.Module):
        def forward(self, x):
            n = x.shape[0]
            return torch.tensor([[1.0, 5.0, 3.0]]).expand(n, 3).clone()

    model = FixedLogits()
    x = torch.rand(2, 1, 28, 28)
    y = torch.tensor([1, 2])  # true class is the max for sample 0, not for sample 1
    m = margin(model, x, y)
    # sample 0: true logit 5.0, runner-up 3.0 -> margin 2.0
    # sample 1: true logit 3.0, runner-up 5.0 -> margin -2.0
    assert torch.allclose(m, torch.tensor([2.0, -2.0]), atol=1e-5)


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


def test_random_noise_perturbation_matches_l2_budget():
    torch.manual_seed(0)
    x = torch.rand(5, 1, 28, 28) * 0.5 + 0.25  # keep away from [0,1] edges
    budget = torch.full((5,), 0.5)
    x_control = random_noise_perturbation(x, budget, generator=torch.Generator().manual_seed(0))
    achieved = pixel_euclidean_distance(x, x_control)
    # clipping to [0,1] can shrink the achieved norm below budget, but not
    # exceed it noticeably
    assert (achieved <= budget + 1e-4).all()


def test_pixel_euclidean_distance_zero_for_identical():
    x = torch.rand(3, 1, 28, 28)
    d = pixel_euclidean_distance(x, x)
    assert torch.allclose(d, torch.zeros(3), atol=1e-6)


def test_method_b_signature_distance_zero_for_identical_positive_otherwise():
    torch.manual_seed(0)
    images = torch.rand(3, 28, 28)
    d_self = method_b_signature_distance(images, images)
    assert torch.allclose(d_self, torch.zeros(3), atol=1e-4)

    other = torch.rand(3, 28, 28)
    d_other = method_b_signature_distance(images, other)
    assert (d_other > 0).all()
