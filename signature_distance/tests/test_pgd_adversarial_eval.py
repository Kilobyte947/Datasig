import torch

from signature_distance.hilbert_stream import NUM_SEGMENTS
from signature_distance.method_b_adversarial_eval import SmallCNN, fgsm_attack, train_classifier
from signature_distance.pgd_adversarial_eval import (
    METHOD_B_WINNER_LINES,
    pgd_attack,
    pgd_fold_summary,
    run_pgd_comparison,
)


def _tiny_dataset(n=200, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return torch.utils.data.TensorDataset(x, y)


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
    from signature_distance.method_b_adversarial_eval import margin
    m_orig = margin(model, x, y)
    x_adv_fgsm = fgsm_attack(model, x, y, epsilon=eps)
    x_adv_pgd = pgd_attack(model, x, y, epsilon=eps, num_steps=10, random_start=False)
    m_fgsm = margin(model, x_adv_fgsm, y)
    m_pgd = margin(model, x_adv_pgd, y)
    assert (m_orig - m_pgd).mean().item() >= (m_orig - m_fgsm).mean().item() - 1e-4


def test_run_pgd_comparison_smoke():
    # Tiny/fast smoke test - real training (1 epoch, small sample, few PGD
    # steps) just to confirm the combined Method B/C driver runs end-to-end
    # and produces sane shapes for both methods.
    out = run_pgd_comparison(
        n_per_class=2, epsilons=(0.05,), seed=0,
        cnn_epochs=1, strong_epochs=1, pgd_steps=2, verbose=False,
    )
    assert out["n_images"] == 20
    n_lines = METHOD_B_WINNER_LINES.shape[0]
    for mname in ("SmallCNN", "StrongCNN"):
        e_b = out["method_b"]["models"][mname]["eps"][0.05]
        e_c = out["method_c"]["models"][mname]["eps"][0.05]
        assert e_b["ratio_adv"].shape == (20, n_lines)
        assert e_c["ratio_adv"].shape == (20, NUM_SEGMENTS)
        assert not torch.isnan(e_b["ratio_adv"]).any()
        assert not torch.isnan(e_c["ratio_adv"]).any()
        # both methods evaluated the SAME perturbation - same flip mask
        assert torch.equal(e_b["flip_mask"], e_c["flip_mask"])
        assert e_b["flip_fraction"] == e_c["flip_fraction"]


def test_pgd_fold_summary_structure():
    torch.manual_seed(0)
    n = 20
    flip_mask = torch.zeros(n, dtype=torch.bool)
    flip_mask[:5] = True
    fake_results = {
        "n_images": n, "epsilons": [0.03], "pgd_steps": 10,
        "method_b": {"r": 2.5, "depth": 2, "n_lines": 16, "models": {
            "FakeModel": {"test_acc": 0.99, "eps": {0.03: {
                "flip_mask": flip_mask, "flip_fraction": 0.25, "fgsm_flip_fraction": 0.2,
                "ratio_adv": torch.rand(n, 16) + 0.1, "ratio_control": torch.rand(n, 16) * 0.2,
                "dist_adv": torch.rand(n, 16) + 0.5, "dist_control": torch.rand(n, 16) + 0.5,
            }}},
        }},
        "method_c": {"r": 2.5, "depth": 3, "n_segments": 16, "models": {
            "FakeModel": {"test_acc": 0.99, "eps": {0.03: {
                "flip_mask": flip_mask, "flip_fraction": 0.25, "fgsm_flip_fraction": 0.2,
                "ratio_adv": torch.rand(n, 16) + 0.1, "ratio_control": torch.rand(n, 16) * 0.2,
                "dist_adv": torch.rand(n, 16) + 0.5, "dist_control": torch.rand(n, 16) + 0.5,
            }}},
        }},
    }
    summary = pgd_fold_summary(fake_results)
    for key in ("method_b", "method_c"):
        assert "overall_mean_fold" in summary[key]
        assert summary[key]["overall_total"] == 16
        entry = summary[key]["by_model_eps"]["FakeModel"][0.03]
        assert entry["n_flipped"] == 5
        assert entry["fgsm_flip_fraction"] == 0.2
