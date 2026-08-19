import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnist_lipschitz.models import LogisticRegressionModel
from mnist_lipschitz.adversarial.attacks import fgsm_attack, pgd_attack


def _synthetic_batch(seed=0, n=16):
    """Random-weight LogisticRegressionModel + random [0,1] pixel-like input -- no MNIST download
    or training needed, since these tests only check the attacks' geometric/gradient-step
    properties, not attack strength on a real classifier."""
    torch.manual_seed(seed)
    model = LogisticRegressionModel()
    x = torch.rand(n, 784)
    y = torch.randint(0, 10, (n,))
    return model, x, y


def test_fgsm_stays_within_epsilon_ball_and_pixel_range():
    model, x, y = _synthetic_batch()
    epsilon = 0.1
    x_adv = fgsm_attack(model, x, y, epsilon)
    assert (x_adv >= x - epsilon - 1e-9).all()
    assert (x_adv <= x + epsilon + 1e-9).all()
    assert (x_adv >= 0.0).all()
    assert (x_adv <= 1.0).all()


def test_fgsm_epsilon_zero_returns_x_unchanged():
    model, x, y = _synthetic_batch()
    x_adv = fgsm_attack(model, x, y, epsilon=0.0)
    assert torch.equal(x_adv, x)


def test_pgd_stays_within_epsilon_ball_and_pixel_range():
    model, x, y = _synthetic_batch()
    epsilon = 0.15
    x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=epsilon / 4, num_steps=10,
                        num_restarts=3, seed=0)
    assert (x_adv >= x - epsilon - 1e-9).all()
    assert (x_adv <= x + epsilon + 1e-9).all()
    assert (x_adv >= 0.0).all()
    assert (x_adv <= 1.0).all()


def test_pgd_single_step_matches_fgsm():
    """The central checkpoint from attacks.py's docstring: with num_restarts=1 (restart 0 always
    starts from the clean x) and alpha=epsilon, a single PGD step performs the exact same update
    as fgsm_attack -- FGSM is a special case of PGD's search, not a separate code path."""
    model, x, y = _synthetic_batch()
    epsilon = 0.2
    x_fgsm = fgsm_attack(model, x, y, epsilon)
    x_pgd = pgd_attack(model, x, y, epsilon=epsilon, alpha=epsilon, num_steps=1, num_restarts=1)
    assert torch.allclose(x_fgsm, x_pgd, atol=1e-10)


def test_pgd_more_restarts_never_reduces_per_example_loss():
    """Best-of-restarts selection is per-example and monotonic in num_restarts by construction:
    adding restart 0 (the deterministic clean start) as a fixed baseline candidate to any set of
    additional random restarts can only ever match or improve each example's achieved loss, never
    regress it."""
    import torch.nn.functional as F
    model, x, y = _synthetic_batch(n=32)
    epsilon = 0.15
    x_1 = pgd_attack(model, x, y, epsilon=epsilon, alpha=epsilon / 4, num_steps=5,
                      num_restarts=1, seed=0)
    x_5 = pgd_attack(model, x, y, epsilon=epsilon, alpha=epsilon / 4, num_steps=5,
                      num_restarts=5, seed=0)
    with torch.no_grad():
        loss_1 = F.cross_entropy(model(x_1), y, reduction="none")
        loss_5 = F.cross_entropy(model(x_5), y, reduction="none")
    assert (loss_5 >= loss_1 - 1e-9).all()
