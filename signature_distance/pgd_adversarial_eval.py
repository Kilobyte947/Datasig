"""PGD adversarial evaluation, applied to both Method B (winning
configuration: 12h+4v, depth=2) and Method C (Hilbert, depth=3) - a second
attack beyond FGSM, so the head-to-head comparison between the two methods
isn't validated more thoroughly for the winner than for the method it beat.

ISOLATION: `pgd_attack` below is a FRESH REIMPLEMENTATION (standard L_inf
PGD, Madry et al. 2018: random start within the epsilon-ball, `num_steps`
gradient-ascent steps of size `alpha` on the cross-entropy loss, each
followed by a projection back onto the epsilon-ball and a clamp to the
valid [0, 1] pixel range) - not imported from `mnist_lipschitz/adversarial`,
per this project's isolation convention (see `method_b_adversarial_eval.py`'s
own docstring). Small perturbation budgets (the same epsilons already used
for FGSM: 0.02, 0.03, 0.05), imperceptible by construction, same as FGSM.

Reuses, unmodified: `SmallCNN`/`StrongCNN`/`train_classifier`/
`load_mnist_train_test`/`margin`/`fgsm_attack`/`pixel_euclidean_distance`/
`random_noise_perturbation` (`method_b_adversarial_eval.py`);
`make_reference_lines`/`line_stream` (`streams.py`); `signature_of_stream`
(`signatures.py`); `choose_rescale_factor`/`rescale_signature`/
`per_line_distances` (`distances.py`); `make_hilbert_curve`/`hilbert_stream`/
`NUM_SEGMENTS` (`hilbert_stream.py`). No changes to Method B or Method C's
construction/hyperparameters - both configs below are fixed, settled values,
not reswept.

SmallCNN/StrongCNN are trained ONCE and PGD/control/FGSM perturbations are
generated ONCE per model/epsilon, then reused for BOTH Method B and Method
C's evaluation - the two methods see literally the same perturbed images,
not just a matched sample size, which is what makes the head-to-head
comparison meaningful rather than coincidental (same discipline already
used for FGSM's finalist comparison in `method_b_sweep.run_stage_b_validation`,
which shares perturbations across finalists the same way). FGSM is also run
here, on the same freshly-trained models, purely to report a same-run
flip-rate comparison against PGD - it does not feed either method's PGD
evaluation.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import choose_rescale_factor, per_line_distances, rescale_signature
from signature_distance.hilbert_stream import NUM_SEGMENTS, hilbert_stream, make_hilbert_curve
from signature_distance.method_b_adversarial_eval import (
    SmallCNN,
    StrongCNN,
    fgsm_attack,
    load_mnist_train_test,
    margin,
    pixel_euclidean_distance,
    random_noise_perturbation,
    train_classifier,
)
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream, make_reference_lines

RESULTS_DIR = Path(__file__).parent / "results"

# Method B's current winning configuration (Stage 8 sweep, see README.md's
# "Method B: Reference-Line Signature Distance" section) - settled, not
# reswept here (out of scope for this task).
METHOD_B_WINNER_LINES = make_reference_lines(angles_deg=(0, 90), counts=(12, 4), points_per_line=32)
METHOD_B_WINNER_DEPTH = 2
# Method C's settled depth (see README.md's "Method C: Hilbert-Curve
# Signature Distance" section).
METHOD_C_DEPTH = 3


def pgd_attack(model, x, y, epsilon: float, num_steps: int = 10, alpha: float = None,
               random_start: bool = True, generator=None) -> torch.Tensor:
    """Standard L_inf PGD (Madry et al. 2018): `num_steps` steps of
    gradient ascent on cross-entropy loss with step size `alpha` (defaults
    to the common `2.5 * epsilon / num_steps` heuristic), each step
    projected back onto the L_inf epsilon-ball around `x` and clamped to
    [0, 1]. Random start within the epsilon-ball by default
    (`random_start=True`) - standard practice, avoids every attack starting
    from the same point on the loss surface. `epsilon=0` returns `x`
    unchanged (the epsilon-ball projection pins every step back to `x`
    itself, and the random-start delta range collapses to a point)."""
    if alpha is None:
        alpha = 2.5 * epsilon / num_steps if num_steps > 0 else 0.0

    x = x.detach()
    if random_start and epsilon > 0:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon, generator=generator)
        x_adv = (x + delta).clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        (grad,) = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()


def run_pgd_comparison(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                        cnn_epochs: int = 3, strong_epochs: int = 3, pgd_steps: int = 10,
                        verbose: bool = True) -> dict:
    """Trains SmallCNN/StrongCNN once, generates PGD (+ magnitude-matched
    random control, + FGSM for the flip-rate comparison only) perturbations
    once per model/epsilon, then evaluates BOTH Method B's winning
    configuration and Method C against the SAME PGD-perturbed images -
    per-line/per-segment ratios, never merged, same framework as
    `method_b_sweep.run_stage_b_validation` / `hilbert_stream.run_hilbert_adversarial_eval`,
    with `pgd_attack` in place of `fgsm_attack`."""
    torch.manual_seed(seed)
    train_loader, test_loader = load_mnist_train_test()

    models = {}
    for name, model, epochs in [("SmallCNN", SmallCNN(), cnn_epochs), ("StrongCNN", StrongCNN(), strong_epochs)]:
        if verbose:
            print(f"Training {name} ({epochs} epochs)...")
        trained, _, test_acc = train_classifier(model, train_loader, test_loader, epochs=epochs, verbose=verbose)
        trained.eval()
        models[name] = {"model": trained, "test_acc": test_acc}
        if verbose:
            print(f"  {name}: test_acc={test_acc:.4f}")

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    num_lines = METHOD_B_WINNER_LINES.shape[0]
    stream_b_orig = line_stream(images, METHOD_B_WINNER_LINES)
    sig_b_orig_raw = torch.stack(
        [signature_of_stream(stream_b_orig[:, i], depth=METHOD_B_WINNER_DEPTH) for i in range(num_lines)], dim=1
    )
    r_b = choose_rescale_factor(sig_b_orig_raw, depth=METHOD_B_WINNER_DEPTH)
    sig_b_orig = rescale_signature(sig_b_orig_raw, r=r_b, depth=METHOD_B_WINNER_DEPTH)

    curve_c = make_hilbert_curve()
    sig_c_orig_raw = torch.stack(
        [signature_of_stream(hilbert_stream(images, curve_c)[:, i], depth=METHOD_C_DEPTH) for i in range(NUM_SEGMENTS)],
        dim=1,
    )
    r_c = choose_rescale_factor(sig_c_orig_raw, depth=METHOD_C_DEPTH)
    sig_c_orig = rescale_signature(sig_c_orig_raw, r=r_c, depth=METHOD_C_DEPTH)

    results = {
        "n_images": images.shape[0], "epsilons": list(epsilons), "pgd_steps": pgd_steps,
        "method_b": {"r": r_b, "depth": METHOD_B_WINNER_DEPTH, "n_lines": num_lines, "models": {}},
        "method_c": {"r": r_c, "depth": METHOD_C_DEPTH, "n_segments": NUM_SEGMENTS, "models": {}},
    }

    for name, info in models.items():
        model = info["model"]
        b_model_result = {"test_acc": info["test_acc"], "eps": {}}
        c_model_result = {"test_acc": info["test_acc"], "eps": {}}

        for eps in epsilons:
            x_adv_c = pgd_attack(model, images_c, labels, eps, num_steps=pgd_steps, generator=generator)
            x_adv = x_adv_c.squeeze(1)

            pgd_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, pgd_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)
                preds_adv = model(x_adv_c).argmax(dim=1)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()
            flip_mask = preds_adv != labels
            flip_fraction = flip_mask.float().mean().item()

            # FGSM on the same model/epsilon - flip-rate comparison only,
            # does not feed either method's PGD evaluation below.
            x_adv_fgsm_c = fgsm_attack(model, images_c, labels, eps)
            with torch.no_grad():
                preds_fgsm = model(x_adv_fgsm_c).argmax(dim=1)
            fgsm_flip_fraction = (preds_fgsm != labels).float().mean().item()

            stream_b_adv = line_stream(x_adv, METHOD_B_WINNER_LINES)
            stream_b_control = line_stream(x_control, METHOD_B_WINNER_LINES)
            sig_b_adv = rescale_signature(
                torch.stack([signature_of_stream(stream_b_adv[:, i], depth=METHOD_B_WINNER_DEPTH)
                             for i in range(num_lines)], dim=1),
                r=r_b, depth=METHOD_B_WINNER_DEPTH,
            )
            sig_b_control = rescale_signature(
                torch.stack([signature_of_stream(stream_b_control[:, i], depth=METHOD_B_WINNER_DEPTH)
                             for i in range(num_lines)], dim=1),
                r=r_b, depth=METHOD_B_WINNER_DEPTH,
            )
            dist_b_adv = per_line_distances(sig_b_orig, sig_b_adv)
            dist_b_control = per_line_distances(sig_b_orig, sig_b_control)

            sig_c_adv = rescale_signature(
                torch.stack([signature_of_stream(hilbert_stream(x_adv, curve_c)[:, i], depth=METHOD_C_DEPTH)
                             for i in range(NUM_SEGMENTS)], dim=1),
                r=r_c, depth=METHOD_C_DEPTH,
            )
            sig_c_control = rescale_signature(
                torch.stack([signature_of_stream(hilbert_stream(x_control, curve_c)[:, i], depth=METHOD_C_DEPTH)
                             for i in range(NUM_SEGMENTS)], dim=1),
                r=r_c, depth=METHOD_C_DEPTH,
            )
            dist_c_adv = per_line_distances(sig_c_orig, sig_c_adv)
            dist_c_control = per_line_distances(sig_c_orig, sig_c_control)

            b_model_result["eps"][eps] = {
                "flip_mask": flip_mask, "flip_fraction": flip_fraction, "fgsm_flip_fraction": fgsm_flip_fraction,
                "ratio_adv": num_adv.unsqueeze(1) / dist_b_adv,
                "ratio_control": num_control.unsqueeze(1) / dist_b_control,
                "dist_adv": dist_b_adv, "dist_control": dist_b_control,
            }
            c_model_result["eps"][eps] = {
                "flip_mask": flip_mask, "flip_fraction": flip_fraction, "fgsm_flip_fraction": fgsm_flip_fraction,
                "ratio_adv": num_adv.unsqueeze(1) / dist_c_adv,
                "ratio_control": num_control.unsqueeze(1) / dist_c_control,
                "dist_adv": dist_c_adv, "dist_control": dist_c_control,
            }

        results["method_b"]["models"][name] = b_model_result
        results["method_c"]["models"][name] = c_model_result

    return results


def pgd_fold_summary(results: dict) -> dict:
    """Per method (b/c): per model/epsilon mean fold-ratio and exception
    count (mean adversarial ratio <= mean control ratio, on the genuinely
    flipped subset), plus an overall aggregate across every model x epsilon
    x line/segment combination - computed exactly the way the FGSM numbers
    this is meant to sit alongside were reported (Method B: 13.53x/95-96
    exceptions; Method C: 9.88x/96-96), so the two are directly
    comparable."""
    summary = {}
    for method_key in ("method_b", "method_c"):
        n_units = results[method_key]["n_lines"] if method_key == "method_b" else results[method_key]["n_segments"]
        by_model_eps = {}
        overall_folds, overall_exceptions, overall_total = [], 0, 0

        for mname, mres in results[method_key]["models"].items():
            by_model_eps[mname] = {}
            for eps, e in mres["eps"].items():
                flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
                n_flipped = flip_idx.shape[0]
                if n_flipped == 0:
                    by_model_eps[mname][eps] = {"n_flipped": 0}
                    continue

                folds, exceptions = [], 0
                for i in range(n_units):
                    adv_mean = e["ratio_adv"][flip_idx, i].mean().item()
                    ctrl_mean = e["ratio_control"][flip_idx, i].mean().item()
                    overall_total += 1
                    if adv_mean <= ctrl_mean:
                        exceptions += 1
                        overall_exceptions += 1
                    else:
                        folds.append(adv_mean / ctrl_mean)
                        overall_folds.append(adv_mean / ctrl_mean)

                by_model_eps[mname][eps] = {
                    "n_flipped": n_flipped, "n_units": n_units, "exceptions": exceptions,
                    "mean_fold": sum(folds) / len(folds) if folds else float("nan"),
                    "flip_fraction": e["flip_fraction"], "fgsm_flip_fraction": e["fgsm_flip_fraction"],
                }

        summary[method_key] = {
            "by_model_eps": by_model_eps,
            "overall_mean_fold": sum(overall_folds) / len(overall_folds) if overall_folds else float("nan"),
            "overall_exceptions": overall_exceptions, "overall_total": overall_total,
        }

    return summary
