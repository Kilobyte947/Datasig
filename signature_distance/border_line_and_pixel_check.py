"""Two verification checks on Method B's all-16-line adversarial
fold-ratio figures (13.53x FGSM / 16.97x PGD), requested and answered
directly rather than assumed:

1. **Is the ratio computation behind 13.53x/95-96 protected against
   near-zero border-line distances?** Checked by direct code inspection
   first: `distances.per_line_distances` is a raw `(sig1-sig2).norm(dim=-1)`
   with no floor, and every driver that produces this figure
   (`method_b_sweep.run_stage_b_validation`, `pgd_adversarial_eval.
   run_pgd_comparison`) divides `num_adv / dist_adv` directly, with no
   clamp anywhere. No epsilon floor exists. This module then checks
   empirically whether that theoretical risk actually manifests: are the
   4 border lines' (`headline_plot.METHOD_B_BORDER_LINE_INDICES`)
   adversarial distances/ratios disproportionate, and how much does the
   all-16-line mean fold-ratio change if they're excluded?

2. **Plain pixel-Euclidean's own adversarial fold-ratio, on the EXACT
   SAME 200-image pool, models, FGSM perturbations, and epsilons as
   Method B's winning configuration** - the comparison that was
   established (repeatedly, in conversation) to never have been directly
   computed anywhere in this codebase. `pixel_euclidean_distance` is only
   ever used elsewhere to size the magnitude-matched random control, never
   as its own competing ratio denominator.

Both checks share the same underlying run (same models, same FGSM
perturbations, same flip mask), computed once and read two ways - not two
separate experiments.

Reuses, unmodified: `SmallCNN`/`StrongCNN`/`train_classifier`/
`load_mnist_train_test`/`margin`/`fgsm_attack`/`pixel_euclidean_distance`/
`random_noise_perturbation` (`method_b_adversarial_eval.py`); `line_stream`
(`streams.py`); `signature_of_stream` (`signatures.py`);
`choose_rescale_factor`/`rescale_signature`/`per_line_distances`
(`distances.py`); `METHOD_B_WINNER_LINES`/`METHOD_B_WINNER_DEPTH`
(`pgd_adversarial_eval.py`); `METHOD_B_BORDER_LINE_INDICES`
(`headline_plot.py`).
"""

from pathlib import Path

import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import choose_rescale_factor, per_line_distances, rescale_signature
from signature_distance.headline_plot import METHOD_B_BORDER_LINE_INDICES
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
from signature_distance.pgd_adversarial_eval import METHOD_B_WINNER_DEPTH, METHOD_B_WINNER_LINES
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream

RESULTS_DIR = Path(__file__).parent / "results"

INFORMATIVE_LINE_INDICES = tuple(i for i in range(16) if i not in METHOD_B_BORDER_LINE_INDICES)


def run_border_and_pixel_check(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                                cnn_epochs: int = 3, strong_epochs: int = 3, verbose: bool = True) -> dict:
    """Trains SmallCNN/StrongCNN once, generates FGSM + magnitude-matched
    random control once per model/epsilon, then computes (on the IDENTICAL
    perturbed images): Method B's all-16-line signature ratios (same
    convention as the 13.53x figure), and pixel-Euclidean's own ratio -
    both from the same `num_adv`/`num_control` numerator, so they're
    directly comparable on identical pairs."""
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

            # Method B: all 16 lines, no exclusion - same convention as
            # the 13.53x/95-96 figure.
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
    adv_mean = ratio_adv_1d.mean().item()
    ctrl_mean = ratio_control_1d.mean().item()
    return {"adv_mean": adv_mean, "ctrl_mean": ctrl_mean,
            "fold": adv_mean / ctrl_mean if adv_mean > ctrl_mean else None,
            "exception": adv_mean <= ctrl_mean}


def summarize_border_and_pixel_check(results: dict) -> dict:
    """Three summaries from the one run above:

    - `method_b_all16`: the all-16-line mean fold/exceptions (should
      reproduce ~13.53x/95-96 FGSM, confirming this is the same code path).
    - `border_vs_informative`: border lines' (`METHOD_B_BORDER_LINE_INDICES`)
      mean adversarial/control distance and fold, reported separately from
      the 12 informative lines', to check directly whether border lines
      have disproportionately small distances / inflated ratios - and what
      the all-16 mean fold becomes with them excluded, to quantify how
      much of the headline number (if any) they're responsible for.
    - `pixel`: pixel-Euclidean's own fold/exceptions, same flipped-pairs
      convention as everywhere else in this project, directly comparable
      to `method_b_all16` and to the 12-line-informative figure (~15.00x/
      1-72) on IDENTICAL pairs.
    """
    b_all16_folds, b_all16_exceptions, b_all16_total = [], 0, 0
    b_border_dists_adv, b_border_dists_control = [], []
    b_informative_dists_adv, b_informative_dists_control = [], []
    b_informative_folds, b_informative_exceptions, b_informative_total = [], 0, 0
    pixel_folds, pixel_exceptions, pixel_total = [], 0, 0
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
                if r["exception"]:
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
                    if r["exception"]:
                        b_informative_exceptions += 1
                    else:
                        b_informative_folds.append(r["fold"])

            rp = _fold_and_exceptions(e["ratio_pixel_adv"][flip_idx], e["ratio_pixel_control"][flip_idx])
            pixel_total += 1
            if rp["exception"]:
                pixel_exceptions += 1
            else:
                pixel_folds.append(rp["fold"])

    return {
        "method_b_all16": {
            "mean_fold": sum(b_all16_folds) / len(b_all16_folds) if b_all16_folds else float("nan"),
            "exceptions": b_all16_exceptions, "total": b_all16_total,
        },
        "border_vs_informative": {
            "border_mean_dist_adv": sum(b_border_dists_adv) / len(b_border_dists_adv),
            "border_mean_dist_control": sum(b_border_dists_control) / len(b_border_dists_control),
            "informative_mean_dist_adv": sum(b_informative_dists_adv) / len(b_informative_dists_adv),
            "informative_mean_dist_control": sum(b_informative_dists_control) / len(b_informative_dists_control),
            "method_b_12line_mean_fold": sum(b_informative_folds) / len(b_informative_folds) if b_informative_folds else float("nan"),
            "method_b_12line_exceptions": b_informative_exceptions, "method_b_12line_total": b_informative_total,
            "per_line_mean_fold": {i: (sum(v["folds"]) / len(v["folds"]) if v["folds"] else float("nan"))
                                    for i, v in per_line_detail.items()},
            "per_line_mean_dist_adv": {i: sum(v["dist_adv"]) / len(v["dist_adv"]) for i, v in per_line_detail.items()},
        },
        "pixel": {
            "mean_fold": sum(pixel_folds) / len(pixel_folds) if pixel_folds else float("nan"),
            "exceptions": pixel_exceptions, "total": pixel_total,
        },
    }
