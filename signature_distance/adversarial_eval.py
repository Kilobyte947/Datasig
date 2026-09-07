"""Adversarial / Lipschitz-ratio evaluation, for every (method, attack,
analysis-angle) combination this project computes: Method A, Method B, and
Method C (Hilbert), FGSM and PGD, plus the per-path/border-line follow-up
analyses built on top of Method B's evaluation. Kept as one file, clearly
sectioned below, rather than split by method/attack - the underlying
protocol (load the shared SmallCNN/StrongCNN checkpoint once, perturb once
per model/epsilon, compute a margin-change numerator over a distance
denominator, never merge across methods) is identical across every
section; only the denominator and the sample it's evaluated on differ.

Numerator convention throughout: `margin(model, x, y_true)` (distances.py),
the same scalar this project's Lipschitz-ratio numerator is always built
from. Denominators are pixel-Euclidean or a per-method signature distance
(both distances.py). Attacks are `fgsm_attack`/`pgd_attack`/
`random_noise_perturbation` (attacks.py) - the random control is a
magnitude-matched, non-gradient-directed perturbation, used throughout to
check whether a distance measure separates genuinely *adversarial* shifts
from equally-large but undirected ones.

Models: every driver in this file loads the shared `mnist_example`
SmallCNN/StrongCNN checkpoint (`models.train_or_load_small_cnn`/
`train_or_load_strong_cnn` - no training here), so every number in this
file is computed on literally the same trained weights, directly
comparable across sections. (A pre-publication audit found several of
these drivers previously trained their own fresh, quick 3-epoch pair
instead - which gave a spurious FGSM > PGD fold-ratio reversal for Method
B; switched to the canonical checkpoint for exactly that reason. Seed/
width sweeps that need independently-trained models by design live
elsewhere - `mnist_example`'s adversarial seed/width sweeps, not in this
file.)
"""

import math
from pathlib import Path

import torch

from signature_distance.attacks import fgsm_attack, pgd_attack, random_noise_perturbation
from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    METHOD_A_R,
    METHOD_B_BORDER_LINE_INDICES,
    METHOD_B_INFORMATIVE_LINE_INDICES,
    METHOD_B_LINES,
    METHOD_B_R,
    METHOD_B_WINNER_DEPTH,
    METHOD_B_WINNER_LINES,
    METHOD_C_DEPTH,
    SIGNATURE_DEPTH,
    choose_rescale_factor,
    margin,
    method_a_signature_distance,
    method_b_signature_distance,
    per_line_distances,
    pixel_euclidean_distance,
    rescale_signature,
)
from signature_distance.models import train_or_load_small_cnn, train_or_load_strong_cnn
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import (
    NUM_SEGMENTS,
    POINTS_PER_SEGMENT,
    hilbert_stream,
    line_stream,
    make_hilbert_curve,
)

torch.set_default_dtype(torch.float64)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------


def run_method_a_adversarial_evaluation(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                         seed: int = 0, device: str = "cpu", verbose: bool = True) -> dict:
    """Loads the shared canonical SmallCNN/StrongCNN checkpoint (no training
    here - see `run_pgd_comparison`'s docstring for why), then for a sample
    of test images (n_per_class per digit, from data_pool.load_eval_pool)
    and each epsilon: generate an FGSM adversarial perturbation and a
    magnitude-matched random control perturbation, compute the margin-change
    numerator, both denominators (pixel-Euclidean and Method A signature
    distance), and the resulting ratios. Returns a nested dict, one entry
    per model.

    Field names use `denom_pixel_*`/`denom_sig_*` and `ratio_pixel_*`/
    `ratio_sig_*` (not Method B's `denom_a_*`/`denom_b_*` convention, which
    labels the two distances under test - (a) pixel, (b) Method B - not
    "Method A"/"Method B"; reusing that here would make "ratio_b" mean
    "Method A's ratio", exactly the confusion worth avoiding).
    """
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, train_acc, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "train_acc": train_acc, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)  # (N, 28, 28)
    images_c = images.unsqueeze(1)  # (N, 1, 28, 28) for model input
    generator = torch.Generator().manual_seed(seed)

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "models": {}}

    for name, info in models.items():
        model = info["model"]
        model_result = {
            "train_acc": info["train_acc"], "test_acc": info["test_acc"],
            "eps": {},
        }

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)  # (N,1,28,28)
            x_adv = x_adv_c.squeeze(1)  # (N,28,28), for Method A's stream functions

            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()

            denom_pixel_adv = pixel_euclidean_distance(images_c, x_adv_c)
            denom_pixel_control = pixel_euclidean_distance(images_c, x_control_c)
            denom_sig_adv = method_a_signature_distance(images, x_adv)
            denom_sig_control = method_a_signature_distance(images, x_control)

            with torch.no_grad():
                preds_adv = model(x_adv_c).argmax(dim=1)
            flip_fraction = (preds_adv != labels).float().mean().item()

            eps_result = {
                "flip_fraction": flip_fraction,
                "numerator_adv": num_adv, "numerator_control": num_control,
                "denom_pixel_adv": denom_pixel_adv, "denom_pixel_control": denom_pixel_control,
                "denom_sig_adv": denom_sig_adv, "denom_sig_control": denom_sig_control,
                "ratio_pixel_adv": num_adv / denom_pixel_adv,
                "ratio_sig_adv": num_adv / denom_sig_adv,
                "ratio_pixel_control": num_control / denom_pixel_control,
                "ratio_sig_control": num_control / denom_sig_control,
                "labels": labels, "x_adv": x_adv, "images": images,
            }

            top10 = eps_result["ratio_sig_adv"].topk(min(10, images.shape[0])).indices
            eps_result["top10_pairs"] = [
                (int(labels[i]), int(preds_adv[i]), float(eps_result["ratio_sig_adv"][i]))
                for i in top10
            ]

            model_result["eps"][eps] = eps_result

        results["models"][name] = model_result

    return results


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------


def run_method_b_adversarial_evaluation(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                         seed: int = 0, device: str = "cpu", verbose: bool = True) -> dict:
    """Loads the shared canonical SmallCNN/StrongCNN checkpoint (no training
    here - see `run_pgd_comparison`'s docstring for why), then for a sample
    of test images (n_per_class per digit, from data_pool.load_eval_pool)
    and each epsilon: generate an FGSM adversarial perturbation and a
    magnitude-matched random control perturbation, compute the margin-change
    numerator, both denominators (pixel-Euclidean and Method B signature
    distance), and the resulting ratios. Returns a nested dict, one entry
    per model.

    Field names: "a" = pixel-Euclidean, "b" = Method B signature distance
    (`denom_a_*`/`denom_b_*`/`ratio_a_*`/`ratio_b_*`).
    """
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, train_acc, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "train_acc": train_acc, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)  # (N, 28, 28)
    images_c = images.unsqueeze(1)  # (N, 1, 28, 28) for model input
    generator = torch.Generator().manual_seed(seed)

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "models": {}}

    for name, info in models.items():
        model = info["model"]
        model_result = {
            "train_acc": info["train_acc"], "test_acc": info["test_acc"],
            "eps": {},
        }

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)  # (N,1,28,28)
            x_adv = x_adv_c.squeeze(1)  # (N,28,28), for Method B's stream functions

            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()

            denom_a_adv = pixel_euclidean_distance(images_c, x_adv_c)
            denom_a_control = pixel_euclidean_distance(images_c, x_control_c)
            denom_b_adv = method_b_signature_distance(images, x_adv)
            denom_b_control = method_b_signature_distance(images, x_control)

            with torch.no_grad():
                preds_adv = model(x_adv_c).argmax(dim=1)
            flip_fraction = (preds_adv != labels).float().mean().item()

            eps_result = {
                "flip_fraction": flip_fraction,
                "numerator_adv": num_adv, "numerator_control": num_control,
                "denom_a_adv": denom_a_adv, "denom_a_control": denom_a_control,
                "denom_b_adv": denom_b_adv, "denom_b_control": denom_b_control,
                "ratio_a_adv": num_adv / denom_a_adv, "ratio_b_adv": num_adv / denom_b_adv,
                "ratio_a_control": num_control / denom_a_control,
                "ratio_b_control": num_control / denom_b_control,
                "labels": labels, "x_adv": x_adv, "images": images,
            }

            top10 = eps_result["ratio_b_adv"].topk(min(10, images.shape[0])).indices
            eps_result["top10_pairs"] = [
                (int(labels[i]), int(preds_adv[i]), float(eps_result["ratio_b_adv"][i]))
                for i in top10
            ]

            model_result["eps"][eps] = eps_result

        results["models"][name] = model_result

    return results


# ---------------------------------------------------------------------------
# Method C / Hilbert
# ---------------------------------------------------------------------------


def per_segment_rescaled_signatures(images: torch.Tensor, curve: torch.Tensor,
                                     depth: int, r: float) -> torch.Tensor:
    stream = hilbert_stream(images, curve)
    num_segments = stream.shape[1]
    sig = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_segments)], dim=1
    )
    return rescale_signature(sig, r=r, depth=depth)


def run_hilbert_adversarial_eval(depth: int, n_per_class: int = 20,
                                  epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                                  verbose: bool = True) -> dict:
    """Full per-path adversarial/control evaluation for Method C, same
    framework as `run_per_path_adversarial_eval` below (FGSM, matched
    random control, margin-difference numerator, per-segment distances),
    applied to the Hilbert-curve segments instead of Method B's reference
    lines. All 16 segments are used (no a priori "informative subset"
    exclusion - unlike Method B's structural border lines, there's no a
    priori reason any particular segment index would be chance-level for a
    space-filling curve; Stage A's own AUC screen, `distances.
    evaluate_hilbert_depths`, is the empirical check for that, not an
    assumption carried in here). Loads the shared canonical SmallCNN/
    StrongCNN checkpoint (no training - see `run_pgd_comparison`'s
    docstring for why)."""
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, _, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    curve = make_hilbert_curve()
    sig_orig_raw = torch.stack(
        [signature_of_stream(hilbert_stream(images, curve)[:, i], depth=depth) for i in range(NUM_SEGMENTS)], dim=1
    )
    r = choose_rescale_factor(sig_orig_raw, depth=depth)
    sig_orig = rescale_signature(sig_orig_raw, r=r, depth=depth)

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "depth": depth, "r": r, "models": {}}

    for name, info in models.items():
        model = info["model"]
        model_result = {"test_acc": info["test_acc"], "eps": {}}

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)
            x_adv = x_adv_c.squeeze(1)
            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)
                preds_adv = model(x_adv_c).argmax(dim=1)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()

            sig_adv = per_segment_rescaled_signatures(x_adv, curve, depth, r)
            sig_control = per_segment_rescaled_signatures(x_control, curve, depth, r)

            dist_adv = per_line_distances(sig_orig, sig_adv)
            dist_control = per_line_distances(sig_orig, sig_control)

            ratio_adv = num_adv.unsqueeze(1) / dist_adv
            ratio_control = num_control.unsqueeze(1) / dist_control

            model_result["eps"][eps] = {
                "flip_mask": preds_adv != labels,
                "flip_fraction": (preds_adv != labels).float().mean().item(),
                "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                "dist_adv": dist_adv, "dist_control": dist_control,
            }

        results["models"][name] = model_result

    return results


def summarize_hilbert_result(results: dict) -> dict:
    """Per model/epsilon, per segment: mean ratio on genuinely adversarial
    (flipped) pairs vs. mean ratio on the matched control pairs - same
    definition as Method B's per-path fold-ratio finding, over all 16
    segments (no exclusion)."""
    summary = {}
    for name, mres in results["models"].items():
        summary[name] = {}
        for eps, e in mres["eps"].items():
            flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
            n_flipped = flip_idx.shape[0]
            if n_flipped == 0:
                summary[name][eps] = {"n_flipped": 0}
                continue
            per_segment = {}
            for i in range(NUM_SEGMENTS):
                per_segment[i] = {
                    "mean_ratio_adv_flipped": e["ratio_adv"][flip_idx, i].mean().item(),
                    "mean_ratio_control_matched": e["ratio_control"][flip_idx, i].mean().item(),
                }
            summary[name][eps] = {"n_flipped": n_flipped, "per_segment": per_segment}
    return summary


def hilbert_robustness_check(results: dict, n_exclude: int = 2) -> dict:
    """Same spirit as Method B's fold_ratio_robustness: identifies the
    `n_exclude` segments with the smallest mean baseline distance (the
    same kind of scale confound flagged for Method B's lines 9/14 -
    checked here rather than assumed absent) and reports whether the
    aggregate fold-ratio survives their exclusion.
    """
    summary = summarize_hilbert_result(results)
    report = {}

    for name, mres in results["models"].items():
        report[name] = {}
        for eps, e in mres["eps"].items():
            s = summary[name][eps]
            n_flipped = s["n_flipped"]
            if n_flipped == 0:
                report[name][eps] = {"n_flipped": 0}
                continue

            baseline_dist = {
                i: ((e["dist_adv"][:, i].mean() + e["dist_control"][:, i].mean()) / 2).item()
                for i in range(NUM_SEGMENTS)
            }
            excluded = sorted(baseline_dist, key=baseline_dist.get)[:n_exclude]
            kept = [i for i in range(NUM_SEGMENTS) if i not in excluded]

            fold_all = {
                i: s["per_segment"][i]["mean_ratio_adv_flipped"] / s["per_segment"][i]["mean_ratio_control_matched"]
                for i in range(NUM_SEGMENTS)
            }
            fold_kept = {i: fold_all[i] for i in kept}

            report[name][eps] = {
                "n_flipped": n_flipped,
                "excluded_segments": excluded,
                "mean_fold_all": sum(fold_all.values()) / len(fold_all),
                "mean_fold_kept": sum(fold_kept.values()) / len(fold_kept),
                "all_kept_survive_adv_gt_control": all(v > 1.0 for v in fold_kept.values()),
            }

    return report


def run_hilbert_adversarial_eval_with_images(depth: int = 3, n_per_class: int = 20,
                                              epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                                              verbose: bool = True) -> dict:
    """Same checkpoint-loading/attack pipeline as `run_hilbert_adversarial_eval`
    above, additionally retaining `images`, `labels`, `x_adv`, `x_control`,
    and the fixed `curve` for gallery plotting (`plots.
    plot_hilbert_spike_gallery`/`plot_spike_comparison`). No new
    adversarial generation or metric - given seed=0, reproduces that
    function's own ratio/distance results exactly (both load the identical
    canonical checkpoint, so this holds even more strongly now than under
    independently-trained models)."""
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, _, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    curve = make_hilbert_curve()
    sig_orig_raw = torch.stack(
        [signature_of_stream(hilbert_stream(images, curve)[:, i], depth=depth) for i in range(NUM_SEGMENTS)],
        dim=1,
    )
    r = choose_rescale_factor(sig_orig_raw, depth=depth)
    sig_orig = rescale_signature(sig_orig_raw, r=r, depth=depth)

    results = {
        "n_images": images.shape[0], "epsilons": list(epsilons), "depth": depth, "r": r,
        "images": images, "labels": labels, "curve": curve, "models": {},
    }

    for name, info in models.items():
        model = info["model"]
        model_result = {"test_acc": info["test_acc"], "eps": {}}

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)
            x_adv = x_adv_c.squeeze(1)
            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)
                preds_adv = model(x_adv_c).argmax(dim=1)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()

            sig_adv = per_segment_rescaled_signatures(x_adv, curve, depth, r)
            sig_control = per_segment_rescaled_signatures(x_control, curve, depth, r)

            dist_adv = per_line_distances(sig_orig, sig_adv)
            dist_control = per_line_distances(sig_orig, sig_control)

            ratio_adv = num_adv.unsqueeze(1) / dist_adv
            ratio_control = num_control.unsqueeze(1) / dist_control

            model_result["eps"][eps] = {
                "flip_mask": preds_adv != labels,
                "flip_fraction": (preds_adv != labels).float().mean().item(),
                "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                "dist_adv": dist_adv, "dist_control": dist_control,
                "x_adv": x_adv, "x_control": x_control,
            }

        results["models"][name] = model_result

    return results


# ---------------------------------------------------------------------------
# PGD (Method B + Method C)
#
# SmallCNN/StrongCNN are trained ONCE and PGD/control/FGSM perturbations are
# generated ONCE per model/epsilon, then reused for BOTH Method B and Method
# C's evaluation - the two methods see literally the same perturbed images,
# not just a matched sample size, which is what makes the head-to-head
# comparison meaningful rather than coincidental (same discipline
# `run_stage_b_validation` (method_b_sweep.py) uses, sharing perturbations
# across finalists the same way). FGSM is also run here, on the same
# freshly-trained models, purely to report a same-run flip-rate comparison
# against PGD - it does not feed either method's PGD evaluation.
# ---------------------------------------------------------------------------


def run_pgd_comparison(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                        pgd_steps: int = 10, verbose: bool = True) -> dict:
    """Loads the shared canonical SmallCNN/StrongCNN checkpoint (the same
    trained weights `mnist_example` uses, via `models.train_or_load_*` -
    no training here), generates PGD (+ magnitude-matched random control,
    + FGSM for the flip-rate comparison only) perturbations once per
    model/epsilon, then evaluates BOTH Method B's winning configuration
    and Method C against the SAME PGD-perturbed images - per-line/per-
    segment ratios, never merged, same framework as
    `run_hilbert_adversarial_eval`/`method_b_sweep.run_stage_b_validation`,
    with `pgd_attack` in place of `fgsm_attack`.

    Previously trained a fresh 3-epoch SmallCNN/StrongCNN pair per call;
    switched to the canonical checkpoint after a pre-publication audit
    found that undertrained pair gave an FGSM > PGD fold-ratio reversal
    that doesn't reproduce on the properly-trained models (PGD, the
    stronger attack by flip-rate, correctly gives the higher fold-ratio
    on the canonical checkpoint - see `adversarial.md`)."""
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, _, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "test_acc": test_acc}

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
    this is meant to sit alongside were reported, so the two are directly
    comparable (see `adversarial_eval.ipynb`'s PGD section for the current
    headline figures - the specific numbers here shift with the checkpoint/
    winner-geometry choice, so aren't hardcoded in this docstring).

    A (model, eps, line/segment) triple whose adversarial or control
    perturbation leaves that unit's signature unchanged (dist == 0) makes
    its ratio non-finite and is excluded from every mean-fold below,
    counted separately as `degenerate` rather than silently averaged in -
    one such case is enough to send a naive mean to +inf. This hits Method
    B's `overall_mean_fold` systematically because its winner geometry
    (`distances.METHOD_B_WINNER_LINES`) includes 2 lines running exactly
    along MNIST's constant-zero border (`distances.METHOD_B_BORDER_LINE_INDICES`)
    - a gradient-based attack legitimately produces exactly zero movement
    there for some images. `method_b`'s summary therefore also reports an
    `informative_*` breakdown restricted to the 14 non-border lines
    (`distances.METHOD_B_INFORMATIVE_LINE_INDICES`), mirroring the FGSM
    all16-vs-informative split (`summarize_border_and_pixel_check`) -
    use `informative_overall_mean_fold` as Method B's headline PGD figure,
    not `overall_mean_fold`, for exactly the same reason. Method C's
    Hilbert segments aren't pinned to fixed border coordinates the way
    Method B's reference lines are, so no equivalent split is needed there."""
    summary = {}
    for method_key in ("method_b", "method_c"):
        n_units = results[method_key]["n_lines"] if method_key == "method_b" else results[method_key]["n_segments"]
        is_border = (
            (lambda i: i in METHOD_B_BORDER_LINE_INDICES) if method_key == "method_b" else (lambda i: False)
        )
        by_model_eps = {}
        overall_folds, overall_exceptions, overall_degenerate, overall_total = [], 0, 0, 0
        informative_folds, informative_exceptions, informative_degenerate, informative_total = [], 0, 0, 0

        for mname, mres in results[method_key]["models"].items():
            by_model_eps[mname] = {}
            for eps, e in mres["eps"].items():
                flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
                n_flipped = flip_idx.shape[0]
                if n_flipped == 0:
                    by_model_eps[mname][eps] = {"n_flipped": 0}
                    continue

                folds, exceptions, degenerate = [], 0, 0
                informative_folds_eps, informative_exceptions_eps = [], 0
                for i in range(n_units):
                    adv_mean = e["ratio_adv"][flip_idx, i].mean().item()
                    ctrl_mean = e["ratio_control"][flip_idx, i].mean().item()
                    fold = adv_mean / ctrl_mean if adv_mean > ctrl_mean else None
                    is_degenerate = fold is not None and not math.isfinite(fold)
                    overall_total += 1
                    if is_degenerate:
                        degenerate += 1
                        overall_degenerate += 1
                    elif adv_mean <= ctrl_mean:
                        exceptions += 1
                        overall_exceptions += 1
                    else:
                        folds.append(fold)
                        overall_folds.append(fold)

                    if not is_border(i):
                        informative_total += 1
                        if is_degenerate:
                            informative_degenerate += 1
                        elif adv_mean <= ctrl_mean:
                            informative_exceptions += 1
                            informative_exceptions_eps += 1
                        else:
                            informative_folds.append(fold)
                            informative_folds_eps.append(fold)

                by_model_eps[mname][eps] = {
                    "n_flipped": n_flipped, "n_units": n_units,
                    "exceptions": exceptions, "degenerate": degenerate,
                    "mean_fold": sum(folds) / len(folds) if folds else float("nan"),
                    "informative_mean_fold": (sum(informative_folds_eps) / len(informative_folds_eps)
                                               if informative_folds_eps else float("nan")),
                    "informative_exceptions": informative_exceptions_eps,
                    "flip_fraction": e["flip_fraction"], "fgsm_flip_fraction": e["fgsm_flip_fraction"],
                }

        summary[method_key] = {
            "by_model_eps": by_model_eps,
            "overall_mean_fold": sum(overall_folds) / len(overall_folds) if overall_folds else float("nan"),
            "overall_exceptions": overall_exceptions, "overall_degenerate": overall_degenerate,
            "overall_total": overall_total,
            "informative_overall_mean_fold": (sum(informative_folds) / len(informative_folds)
                                               if informative_folds else float("nan")),
            "informative_overall_exceptions": informative_exceptions,
            "informative_overall_degenerate": informative_degenerate,
            "informative_overall_total": informative_total,
        }

    return summary


# ---------------------------------------------------------------------------
# Per-path (Method B)
#
# Corrects the framing of the merged-distance evaluation above: don't merge
# the 16 reference-line paths into any single combined score (not the
# 496-dim concatenation, not max/top-k/weighted-sum). Instead, treat the 16
# paths the way the pixel-space Lipschitz work (toy_example/mnist_example)
# treats individual pixels - as separate coordinates, each producing its
# own local Lipschitz ratio, examined as a collection, never reduced to one
# number. Directly mirrors the gradient-norm estimator's per-pixel
# sensitivity approach, with paths standing in for pixels.
#
# Uses the ORIGINAL 8h+8v/depth=4 geometry (`distances.METHOD_B_LINES` -
# `make_reference_lines()`'s default args), predating the `method_b_sweep.py`
# hyperparameter sweep - NOT the swept 16h+0v/depth=2 winner geometry used
# by the headline/PGD/border-check sections above. A "line N" referenced
# anywhere in this section (or `plots.plot_spike_gallery`/
# `plot_spike_comparison`) therefore does NOT refer to the same physical
# line as "line N" elsewhere in this file - see README.md's explicit note
# that index numbers are not comparable across the two geometries.
# ---------------------------------------------------------------------------

# The 4 lines running exactly along the image border (row/col 0 or 27) -
# structural consequence of make_reference_lines()'s default
# angles_deg=(0, 90), counts=(8, 8): index 0/7 are the first/last of the 8
# horizontal lines (rows via linspace(0, 27, 8)), index 8/15 the first/last
# of the 8 vertical lines. Identified by distances.run_per_line_auc_diagnostic's
# AUC ranking as carrying zero same/different-digit signal (AUC == 0.5000
# exactly, every one of them) - MNIST digits essentially never touch the
# border, so these run through background regardless of the image.
BORDER_LINE_INDICES = (0, 7, 8, 15)
INFORMATIVE_LINE_INDICES = tuple(i for i in range(16) if i not in BORDER_LINE_INDICES)
# distances.run_per_line_auc_diagnostic's single highest-AUC individual line.
BEST_LINE_INDEX = 6


def per_line_rescaled_signatures(images: torch.Tensor, depth: int = SIGNATURE_DEPTH,
                                  r: float = METHOD_B_R) -> torch.Tensor:
    """(N, num_lines, sig_dim) rescaled per-line signatures for a batch of
    images, stopping one step before method_b_signature_distance's
    concatenation - reuses line_stream/signature_of_stream/rescale_signature
    unchanged."""
    num_lines = METHOD_B_LINES.shape[0]
    stream = line_stream(images, METHOD_B_LINES)
    sig = torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
    )
    return rescale_signature(sig, r=r, depth=depth)


def run_per_path_adversarial_eval(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                   seed: int = 0, verbose: bool = True) -> dict:
    """Same sample/models/attack setup as `run_method_b_adversarial_evaluation`
    above (loads the shared canonical SmallCNN/StrongCNN checkpoint, same
    FGSM epsilons and magnitude-matched random control), but computes 16
    SEPARATE per-line ratios per pair instead of one merged ratio - the
    per-line signatures and distances aren't retained by that function's
    own return value, so they're recomputed here via the same underlying
    calls, not duplicated logic.
    """
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, _, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    sig_orig = per_line_rescaled_signatures(images)  # (N, num_lines, sig_dim) - computed once

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "models": {},
               "labels": labels, "images": images}

    for name, info in models.items():
        model = info["model"]
        model_result = {"test_acc": info["test_acc"], "eps": {}}

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)
            x_adv = x_adv_c.squeeze(1)

            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)
                preds_adv = model(x_adv_c).argmax(dim=1)

            num_adv = (margin_orig - margin_adv).abs()      # (N,)
            num_control = (margin_orig - margin_control).abs()

            sig_adv = per_line_rescaled_signatures(x_adv)
            sig_control = per_line_rescaled_signatures(x_control)

            dist_adv = per_line_distances(sig_orig, sig_adv)          # (N, num_lines)
            dist_control = per_line_distances(sig_orig, sig_control)  # (N, num_lines)

            ratio_adv = num_adv.unsqueeze(1) / dist_adv          # (N, num_lines) - never merged
            ratio_control = num_control.unsqueeze(1) / dist_control

            model_result["eps"][eps] = {
                "flip_mask": preds_adv != labels,
                "flip_fraction": (preds_adv != labels).float().mean().item(),
                "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                "dist_adv": dist_adv, "dist_control": dist_control,
                "x_adv": x_adv, "x_control": x_control,
            }

        results["models"][name] = model_result

    return results


def summarize_informative_subset(results: dict) -> dict:
    """Per model/epsilon, per informative line (the 12 non-border lines,
    line_6 highlighted separately): mean ratio over the GENUINELY adversarial
    pairs (prediction actually flipped) vs. mean ratio over the SAME index
    subset's control pairs (same images, for a matched, apples-to-apples
    comparison - not the full 200-image control set, since the flipped
    subset is a specific, often small, harder-to-classify slice)."""
    summary = {}
    for name, mres in results["models"].items():
        summary[name] = {}
        for eps, e in mres["eps"].items():
            flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
            n_flipped = flip_idx.shape[0]
            per_line = {}
            for i in INFORMATIVE_LINE_INDICES:
                if n_flipped == 0:
                    per_line[i] = {"mean_ratio_adv_flipped": None, "mean_ratio_control_matched": None}
                else:
                    per_line[i] = {
                        "mean_ratio_adv_flipped": e["ratio_adv"][flip_idx, i].mean().item(),
                        "mean_ratio_control_matched": e["ratio_control"][flip_idx, i].mean().item(),
                    }
            summary[name][eps] = {"n_flipped": n_flipped, "per_line": per_line}
    return summary


def spike_analysis(results: dict) -> dict:
    """Per model/epsilon: for every pair, which of the 12 INFORMATIVE lines
    has the largest ratio (argmax over ratio_adv restricted to
    INFORMATIVE_LINE_INDICES - the border lines are excluded here
    specifically because their near-constant, near-zero-distance signatures
    make their ratio a numerically degenerate near-zero-denominator blowup,
    not a meaningful "spike"; verified/reported below). Reports the
    distribution of which line wins most often, for both adversarial and
    control pairs, plus each distribution's entropy (bits) - a more
    concentrated/peaked distribution (lower entropy) means the perturbation
    consistently spikes the same one or two lines; a flatter distribution
    (entropy close to log2(12) ~= 3.58 bits, uniform over 12 lines) means it
    spreads roughly evenly.
    """
    import math

    idx_tensor = torch.tensor(INFORMATIVE_LINE_INDICES)
    analysis = {}
    for name, mres in results["models"].items():
        analysis[name] = {}
        for eps, e in mres["eps"].items():
            ratio_adv_informative = e["ratio_adv"][:, idx_tensor]      # (N, 12)
            ratio_control_informative = e["ratio_control"][:, idx_tensor]

            argmax_adv = idx_tensor[ratio_adv_informative.argmax(dim=1)]
            argmax_control = idx_tensor[ratio_control_informative.argmax(dim=1)]

            def _distribution_and_entropy(argmax_indices):
                counts = {i: 0 for i in INFORMATIVE_LINE_INDICES}
                for v in argmax_indices.tolist():
                    counts[v] += 1
                n = argmax_indices.shape[0]
                probs = [c / n for c in counts.values() if c > 0]
                entropy = -sum(p * math.log2(p) for p in probs)
                return counts, entropy

            counts_adv, entropy_adv = _distribution_and_entropy(argmax_adv)
            counts_control, entropy_control = _distribution_and_entropy(argmax_control)

            # Sanity check on the degenerate-denominator concern: are border
            # lines' distances actually much smaller than informative lines'?
            border_dist_mean = e["dist_adv"][:, list(BORDER_LINE_INDICES)].mean().item()
            informative_dist_mean = e["dist_adv"][:, idx_tensor].mean().item()

            analysis[name][eps] = {
                "argmax_counts_adv": counts_adv, "entropy_adv_bits": entropy_adv,
                "argmax_counts_control": counts_control, "entropy_control_bits": entropy_control,
                "max_entropy_bits": math.log2(len(INFORMATIVE_LINE_INDICES)),
                "border_line_mean_distance": border_dist_mean,
                "informative_line_mean_distance": informative_dist_mean,
            }

    return analysis


# Robustness check: does the fold-ratio finding above survive excluding
# lines 9 and 14 (flagged, in the spike-count analysis, as having
# systematically smaller baseline distances than the other informative
# lines regardless of perturbation type)? Read-only over an already-computed
# `run_per_path_adversarial_eval` result - no new signature computation.

EXCLUDED_LINES = (9, 14)
ROBUST_LINE_INDICES = tuple(i for i in INFORMATIVE_LINE_INDICES if i not in EXCLUDED_LINES)


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def fold_ratio_robustness(results: dict) -> dict:
    """Per model/epsilon: per-line fold-ratio (mean adversarial ratio on
    genuinely flipped pairs / mean control ratio on the same pairs - same
    definition as the fold-ratio finding above), reported for both the
    original 12-line informative set and the 10-line set with lines 9 and
    14 excluded, side by side - plus each line's baseline distance (mean of
    dist_adv and dist_control, i.e. not perturbation-direction-dependent)
    and the Pearson correlation between baseline distance and fold-ratio
    across the 10-line set, to check whether the scale confound extends
    beyond lines 9/14.
    """
    summary = summarize_informative_subset(results)
    report = {}

    for name, mres in results["models"].items():
        report[name] = {}
        for eps, e in mres["eps"].items():
            s = summary[name][eps]
            n_flipped = s["n_flipped"]
            if n_flipped == 0:
                report[name][eps] = {"n_flipped": 0}
                continue

            fold_12 = {
                i: s["per_line"][i]["mean_ratio_adv_flipped"] / s["per_line"][i]["mean_ratio_control_matched"]
                for i in INFORMATIVE_LINE_INDICES
            }
            fold_10 = {i: fold_12[i] for i in ROBUST_LINE_INDICES}

            baseline_dist_10 = {
                i: ((e["dist_adv"][:, i].mean() + e["dist_control"][:, i].mean()) / 2).item()
                for i in ROBUST_LINE_INDICES
            }

            xs = [baseline_dist_10[i] for i in ROBUST_LINE_INDICES]
            ys = [fold_10[i] for i in ROBUST_LINE_INDICES]

            report[name][eps] = {
                "n_flipped": n_flipped,
                "fold_12": fold_12,
                "fold_10": fold_10,
                "mean_fold_12": sum(fold_12.values()) / len(fold_12),
                "mean_fold_10": sum(fold_10.values()) / len(fold_10),
                "min_fold_10": min(fold_10.values()),
                "max_fold_10": max(fold_10.values()),
                "all_10_survive_adv_gt_control": all(v > 1.0 for v in fold_10.values()),
                "baseline_dist_10": baseline_dist_10,
                "dist_fold_correlation_10": _pearson(xs, ys),
            }

    return report


def run_robustness_report(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                           verbose: bool = True) -> dict:
    """Reproducible entry point: regenerates results via
    run_per_path_adversarial_eval (deterministic given seed=0, verified
    bit-for-bit reproducible across independent prior runs) and runs
    fold_ratio_robustness on it. Prints a summary table.
    """
    results = run_per_path_adversarial_eval(
        n_per_class=n_per_class, epsilons=epsilons, seed=seed, verbose=verbose,
    )
    report = fold_ratio_robustness(results)

    for name, per_eps in report.items():
        for eps, r in per_eps.items():
            if r["n_flipped"] == 0:
                print(f"{name} eps={eps}: no flips, skipped")
                continue
            print(f"=== {name} eps={eps} (n_flipped={r['n_flipped']}) ===")
            print(f"  mean fold-ratio, 12 lines: {r['mean_fold_12']:.2f}x")
            print(f"  mean fold-ratio, 10 lines (excl. {EXCLUDED_LINES}): {r['mean_fold_10']:.2f}x")
            print(f"  min/max (10-line): {r['min_fold_10']:.2f}x / {r['max_fold_10']:.2f}x")
            print(f"  all 10 lines still adv > control: {r['all_10_survive_adv_gt_control']}")
            print(f"  dist-vs-fold correlation (10-line): {r['dist_fold_correlation_10']:.3f}")
            print()

    return {"results": results, "report": report}


# ---------------------------------------------------------------------------
# Border-line and pixel check (Method B, winner geometry)
#
# Two verification checks on Method B's all-16-line adversarial fold-ratio
# figures, requested and answered directly rather than assumed:
#
# 1. Is the ratio computation protected against near-zero border-line
#    distances? Checked by direct code inspection first:
#    `distances.per_line_distances` is a raw `(sig1-sig2).norm(dim=-1)`
#    with no floor, and every driver that produces this figure divides
#    `num_adv / dist_adv` directly, with no clamp anywhere. No epsilon
#    floor exists. `run_border_and_pixel_check`/`summarize_border_and_pixel_check`
#    below then check empirically whether that theoretical risk actually
#    manifests: are the border lines' (`distances.METHOD_B_BORDER_LINE_INDICES`)
#    adversarial distances/ratios disproportionate, and how much does the
#    all-16-line mean fold-ratio change if they're excluded?
# 2. Plain pixel-Euclidean's own adversarial fold-ratio, on the EXACT SAME
#    200-image pool, models, FGSM perturbations, and epsilons as Method B's
#    winning configuration - `pixel_euclidean_distance` is only ever used
#    elsewhere to size the magnitude-matched random control, never as its
#    own competing ratio denominator.
#
# Both checks share the same underlying run (same models, same FGSM
# perturbations, same flip mask), computed once and read two ways.
# ---------------------------------------------------------------------------


def run_border_and_pixel_check(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                                verbose: bool = True) -> dict:
    """Loads the shared canonical SmallCNN/StrongCNN checkpoint (no
    training here - see `run_pgd_comparison`'s docstring for why), generates
    FGSM + magnitude-matched random control once per model/epsilon, then
    computes (on the IDENTICAL perturbed images): Method B's all-16-line
    signature ratios (diagnostic only - see `summarize_border_and_pixel_check`),
    and pixel-Euclidean's own ratio - both from the same `num_adv`/
    `num_control` numerator, so they're directly comparable on identical
    pairs."""
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, _, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    num_lines = METHOD_B_WINNER_LINES.shape[0]
    stream_orig = line_stream(images, METHOD_B_WINNER_LINES)
    sig_orig_raw = torch.stack(
        [signature_of_stream(stream_orig[:, i], depth=METHOD_B_WINNER_DEPTH) for i in range(num_lines)], dim=1
    )
    r = choose_rescale_factor(sig_orig_raw, depth=METHOD_B_WINNER_DEPTH)
    sig_orig = rescale_signature(sig_orig_raw, r=r, depth=METHOD_B_WINNER_DEPTH)

    results = {"n_images": images.shape[0], "epsilons": list(epsilons), "r": r, "models": {}}

    for name, info in models.items():
        model = info["model"]
        model_result = {"test_acc": info["test_acc"], "eps": {}}

        for eps in epsilons:
            x_adv_c = fgsm_attack(model, images_c, labels, eps)
            x_adv = x_adv_c.squeeze(1)
            fgsm_l2 = pixel_euclidean_distance(images_c, x_adv_c)
            x_control_c = random_noise_perturbation(images_c, fgsm_l2, generator=generator)
            x_control = x_control_c.squeeze(1)

            with torch.no_grad():
                margin_orig = margin(model, images_c, labels)
                margin_adv = margin(model, x_adv_c, labels)
                margin_control = margin(model, x_control_c, labels)
                preds_adv = model(x_adv_c).argmax(dim=1)

            num_adv = (margin_orig - margin_adv).abs()
            num_control = (margin_orig - margin_control).abs()
            flip_mask = preds_adv != labels

            # Method B: all 16 lines, no exclusion - the diagnostic-only
            # all-16-line convention (see `summarize_border_and_pixel_check`).
            stream_adv = line_stream(x_adv, METHOD_B_WINNER_LINES)
            stream_control = line_stream(x_control, METHOD_B_WINNER_LINES)
            sig_adv = rescale_signature(
                torch.stack([signature_of_stream(stream_adv[:, i], depth=METHOD_B_WINNER_DEPTH)
                             for i in range(num_lines)], dim=1),
                r=r, depth=METHOD_B_WINNER_DEPTH,
            )
            sig_control = rescale_signature(
                torch.stack([signature_of_stream(stream_control[:, i], depth=METHOD_B_WINNER_DEPTH)
                             for i in range(num_lines)], dim=1),
                r=r, depth=METHOD_B_WINNER_DEPTH,
            )
            dist_b_adv = per_line_distances(sig_orig, sig_adv)
            dist_b_control = per_line_distances(sig_orig, sig_control)
            ratio_b_adv = num_adv.unsqueeze(1) / dist_b_adv
            ratio_b_control = num_control.unsqueeze(1) / dist_b_control

            # Pixel-Euclidean, same numerator, same x_adv/x_control.
            dist_pixel_adv = pixel_euclidean_distance(images_c, x_adv_c)
            dist_pixel_control = pixel_euclidean_distance(images_c, x_control_c)
            ratio_pixel_adv = num_adv / dist_pixel_adv
            ratio_pixel_control = num_control / dist_pixel_control

            model_result["eps"][eps] = {
                "flip_mask": flip_mask, "flip_fraction": flip_mask.float().mean().item(),
                "ratio_b_adv": ratio_b_adv, "ratio_b_control": ratio_b_control,
                "dist_b_adv": dist_b_adv, "dist_b_control": dist_b_control,
                "ratio_pixel_adv": ratio_pixel_adv, "ratio_pixel_control": ratio_pixel_control,
                "dist_pixel_adv": dist_pixel_adv, "dist_pixel_control": dist_pixel_control,
            }

        results["models"][name] = model_result

    return results


def _fold_and_exceptions(ratio_adv_1d: torch.Tensor, ratio_control_1d: torch.Tensor) -> dict:
    """`adv_mean`/`ctrl_mean` can be non-finite when a line's adversarial or
    control perturbation leaves that exact line unchanged (dist == 0) - this
    happens systematically for border-touching lines/segments (constant-zero
    MNIST border pixels: a gradient-based attack legitimately produces zero
    movement there for some images), and is flagged as `degenerate` rather
    than silently averaged into `fold` (which would let one exact-zero
    denominator blow the aggregate mean up to inf/absurdly large, as found
    in the pre-publication audit - see `distances.METHOD_B_BORDER_LINE_INDICES`
    and `summarize_border_and_pixel_check`'s border-vs-informative split)."""
    adv_mean = ratio_adv_1d.mean().item()
    ctrl_mean = ratio_control_1d.mean().item()
    fold = adv_mean / ctrl_mean if adv_mean > ctrl_mean else None
    degenerate = fold is not None and not math.isfinite(fold)
    return {"adv_mean": adv_mean, "ctrl_mean": ctrl_mean,
            "fold": fold if not degenerate else None,
            "exception": adv_mean <= ctrl_mean, "degenerate": degenerate}


def summarize_border_and_pixel_check(results: dict) -> dict:
    """Three summaries from the one run above:

    - `method_b_all16`: the all-16-line mean fold/exceptions - diagnostic
      only, unstable (see below), never the headline figure.
    - `border_vs_informative`: border lines' (`distances.METHOD_B_BORDER_LINE_INDICES`)
      mean adversarial/control distance and fold, reported separately from
      the informative lines', to check directly whether border lines
      have disproportionately small distances / inflated ratios - and what
      the all-16 mean fold becomes with them excluded, to quantify how
      much of the headline number (if any) they're responsible for.
    - `pixel`: pixel-Euclidean's own fold/exceptions, same flipped-pairs
      convention as everywhere else in this project, directly comparable
      to `method_b_all16` and to the informative-lines figure on IDENTICAL
      pairs.

    A (model, eps, line) triple whose adversarial or control perturbation
    leaves that exact line unchanged (dist == 0, `_fold_and_exceptions`'s
    `degenerate` flag) is excluded from every mean-fold below and counted
    separately (`degenerate`/`method_b_informative_degenerate`/`pixel_degenerate`)
    rather than being silently averaged in - one such case is enough to send
    a naive mean to +inf (found in the pre-publication audit on
    `method_b_all16`, which is exactly why it's diagnostic-only, never the
    headline figure - see `distances.METHOD_B_BORDER_LINE_INDICES`).
    """
    b_all16_folds, b_all16_exceptions, b_all16_degenerate, b_all16_total = [], 0, 0, 0
    b_border_dists_adv, b_border_dists_control = [], []
    b_informative_dists_adv, b_informative_dists_control = [], []
    b_informative_folds, b_informative_exceptions, b_informative_degenerate, b_informative_total = [], 0, 0, 0
    pixel_folds, pixel_exceptions, pixel_degenerate, pixel_total = [], 0, 0, 0
    per_line_detail = {i: {"folds": [], "dist_adv": [], "dist_control": []} for i in range(16)}

    for mname, mres in results["models"].items():
        for eps, e in mres["eps"].items():
            flip_idx = e["flip_mask"].nonzero(as_tuple=True)[0]
            if flip_idx.shape[0] == 0:
                continue

            for i in range(16):
                r = _fold_and_exceptions(e["ratio_b_adv"][flip_idx, i], e["ratio_b_control"][flip_idx, i])
                b_all16_total += 1
                per_line_detail[i]["dist_adv"].append(e["dist_b_adv"][flip_idx, i].mean().item())
                per_line_detail[i]["dist_control"].append(e["dist_b_control"][flip_idx, i].mean().item())
                if r["degenerate"]:
                    b_all16_degenerate += 1
                elif r["exception"]:
                    b_all16_exceptions += 1
                else:
                    b_all16_folds.append(r["fold"])
                    per_line_detail[i]["folds"].append(r["fold"])

                if i in METHOD_B_BORDER_LINE_INDICES:
                    b_border_dists_adv.append(e["dist_b_adv"][flip_idx, i].mean().item())
                    b_border_dists_control.append(e["dist_b_control"][flip_idx, i].mean().item())
                else:
                    b_informative_dists_adv.append(e["dist_b_adv"][flip_idx, i].mean().item())
                    b_informative_dists_control.append(e["dist_b_control"][flip_idx, i].mean().item())
                    b_informative_total += 1
                    if r["degenerate"]:
                        b_informative_degenerate += 1
                    elif r["exception"]:
                        b_informative_exceptions += 1
                    else:
                        b_informative_folds.append(r["fold"])

            rp = _fold_and_exceptions(e["ratio_pixel_adv"][flip_idx], e["ratio_pixel_control"][flip_idx])
            pixel_total += 1
            if rp["degenerate"]:
                pixel_degenerate += 1
            elif rp["exception"]:
                pixel_exceptions += 1
            else:
                pixel_folds.append(rp["fold"])

    return {
        "method_b_all16": {
            "mean_fold": sum(b_all16_folds) / len(b_all16_folds) if b_all16_folds else float("nan"),
            "exceptions": b_all16_exceptions, "degenerate": b_all16_degenerate, "total": b_all16_total,
        },
        "border_vs_informative": {
            "border_mean_dist_adv": sum(b_border_dists_adv) / len(b_border_dists_adv),
            "border_mean_dist_control": sum(b_border_dists_control) / len(b_border_dists_control),
            "informative_mean_dist_adv": sum(b_informative_dists_adv) / len(b_informative_dists_adv),
            "informative_mean_dist_control": sum(b_informative_dists_control) / len(b_informative_dists_control),
            "method_b_informative_mean_fold": sum(b_informative_folds) / len(b_informative_folds) if b_informative_folds else float("nan"),
            "method_b_informative_exceptions": b_informative_exceptions,
            "method_b_informative_degenerate": b_informative_degenerate, "method_b_informative_total": b_informative_total,
            "per_line_mean_fold": {i: (sum(v["folds"]) / len(v["folds"]) if v["folds"] else float("nan"))
                                    for i, v in per_line_detail.items()},
            "per_line_mean_dist_adv": {i: sum(v["dist_adv"]) / len(v["dist_adv"]) for i, v in per_line_detail.items()},
        },
        "pixel": {
            "mean_fold": sum(pixel_folds) / len(pixel_folds) if pixel_folds else float("nan"),
            "exceptions": pixel_exceptions, "degenerate": pixel_degenerate, "total": pixel_total,
        },
    }
