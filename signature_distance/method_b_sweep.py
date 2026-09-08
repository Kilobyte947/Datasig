"""Method B hyperparameter optimisation (geometry, points per line, truncation
depth, interpolation). 
- Stage A is a cheap screen (no model training, per-line same/different-digit AUC); 
- Stage B is full validation (per-path adversarial/control evaluation on finalists).
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from signature_distance.attacks import fgsm_attack, random_noise_perturbation
from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    auc_for_distance,
    choose_rescale_factor,
    margin,
    per_line_distances,
    pixel_euclidean_distance,
    rescale_signature,
)
from signature_distance.models import train_or_load_small_cnn, train_or_load_strong_cnn
from signature_distance.signatures import signature_of_stream
from signature_distance.streams import line_stream, make_reference_lines

torch.set_default_dtype(torch.float64)

WIDTH = 2

def _level_sizes(depth: int) -> list:
    """Number of coefficients at each signature level 0..depth."""
    return [WIDTH ** n for n in range(depth + 1)]

def signature_dim(depth: int) -> int:
    """Total signature dimension at a given depth."""
    return sum(_level_sizes(depth))


def cubic_spline_refine(stream_one_line: torch.Tensor, upsample_factor: int = 8) -> torch.Tensor:
    """Fits a natural cubic spline through each image's line and resamples at a finer resolution,
    approximating the signature of a curved rather than piecewise-linear path. Does not change 
    signature dimension."""
    n, k, _ = stream_one_line.shape
    t = stream_one_line[:, :, 0].numpy()
    v = stream_one_line[:, :, 1].numpy()

    fine_t_frac = np.linspace(0.0, 1.0, k * upsample_factor).astype(t.dtype)
    out = np.empty((n, k * upsample_factor, 2), dtype=t.dtype)
    for i in range(n):
        # t is already arange(k)/(k-1) for every image (time_channel) -
        # spline is fit over that fixed grid, only v varies per image.
        cs = CubicSpline(t[i], v[i])
        out[i, :, 0] = fine_t_frac
        out[i, :, 1] = cs(fine_t_frac)
    return torch.from_numpy(out)


def build_stream(images: torch.Tensor, angles_deg: tuple, counts: tuple,
                  points_per_line: int, interpolation: str,
                  cubic_upsample: int = 8) -> torch.Tensor:
    """Builds the (N, num_lines, K, 2) stream for one Stage A configuration, with either linear or
    cubic-spline interpolation."""
    lines = make_reference_lines(angles_deg=angles_deg, counts=counts, points_per_line=points_per_line)
    stream = line_stream(images, lines)  # (N, num_lines, points_per_line, 2)
    if interpolation == "linear":
        return stream
    if interpolation == "cubic":
        num_lines = stream.shape[1]
        return torch.stack(
            [cubic_spline_refine(stream[:, i], cubic_upsample) for i in range(num_lines)], dim=1
        )
    raise ValueError(f"unknown interpolation: {interpolation!r}")


def per_line_aucs(sig: torch.Tensor, labels: torch.Tensor) -> list:
    """Same/different-digit AUC for each line in a batch of per-line signatures."""
    n = sig.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)
    same = (labels[iu] == labels[ju]).numpy().astype(int)
    aucs = []
    for i in range(sig.shape[1]):
        d = torch.cdist(sig[:, i], sig[:, i], p=2)[iu, ju].numpy()
        aucs.append(auc_for_distance(same, d)["auc"])
    return aucs


def evaluate_config(images: torch.Tensor, labels: torch.Tensor, angles_deg: tuple, counts: tuple,
                     points_per_line: int, depths: tuple, interpolation: str,
                     max_depth: int, cubic_upsample: int = 8) -> dict:
    """Builds the stream, computes the signature once at max_depth, then scores every depth in
    depths by slicing that result and computing per-line AUCs. Returns a dict keyed by depth."""
    stream = build_stream(images, angles_deg, counts, points_per_line, interpolation, cubic_upsample)
    num_lines = stream.shape[1]

    sig_max = torch.stack(
        [signature_of_stream(stream[:, i], depth=max_depth) for i in range(num_lines)], dim=1
    )  # (N, num_lines, signature_dim(max_depth)) - the one expensive JAX call for this config

    results_by_depth = {}
    for depth in depths:
        dim = signature_dim(depth)
        sig_raw = sig_max[:, :, :dim]
        r = choose_rescale_factor(sig_raw, depth=depth)
        sig = rescale_signature(sig_raw, r=r, depth=depth)
        aucs = per_line_aucs(sig, labels)
        n_chance = sum(1 for a in aucs if a <= 0.505)
        n_informative = num_lines - n_chance
        results_by_depth[depth] = {
            "r": r, "line_aucs": aucs, "best_auc": max(aucs), "mean_auc": sum(aucs) / len(aucs),
            "n_lines": num_lines, "n_informative": n_informative, "n_chance": n_chance,
        }

    return results_by_depth


GEOMETRY_VARIANTS = {
    "8h+8v (baseline)": ((0, 90), (8, 8)),
    "12h+4v": ((0, 90), (12, 4)),
    "16h+0v": ((0,), (16,)),
    "0h+16v": ((90,), (16,)),
}
POINTS_VARIANTS = (16, 32, 48, 64)
DEPTH_VARIANTS = (2, 3, 4, 5, 6)
INTERPOLATION_VARIANTS = ("linear", "cubic")


def run_stage_a_sweep(n_per_class: int = 15, seed: int = 0, verbose: bool = True) -> list:
    """Full joint sweep over geometry, points per line, depth, and interpolation. 
    Returns the scored configurations, sorted by best per-line AUC."""
    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    max_depth = max(DEPTH_VARIANTS)

    rows = []
    total = len(GEOMETRY_VARIANTS) * len(POINTS_VARIANTS) * len(INTERPOLATION_VARIANTS)
    done = 0
    for geom_name, (angles_deg, counts) in GEOMETRY_VARIANTS.items():
        for points_per_line in POINTS_VARIANTS:
            for interpolation in INTERPOLATION_VARIANTS:
                by_depth = evaluate_config(
                    images, labels, angles_deg, counts, points_per_line,
                    DEPTH_VARIANTS, interpolation, max_depth,
                )
                for depth, r in by_depth.items():
                    rows.append({
                        "geometry": geom_name, "points_per_line": points_per_line,
                        "depth": depth, "interpolation": interpolation,
                        "r": r["r"], "best_auc": r["best_auc"], "mean_auc": r["mean_auc"],
                        "n_lines": r["n_lines"], "n_informative": r["n_informative"],
                        "n_chance": r["n_chance"],
                    })
                done += 1
                if verbose:
                    print(f"  [{done}/{total}] {geom_name}, points={points_per_line}, interp={interpolation} done")

    rows.sort(key=lambda row: row["best_auc"], reverse=True)
    return rows

# ---------------------------------------------------------------------------
# Stage B: full validation on finalists
# ---------------------------------------------------------------------------

def _config_signatures(images: torch.Tensor, angles_deg: tuple, counts: tuple,
                        points_per_line: int, depth: int, interpolation: str,
                        cubic_upsample: int = 8) -> torch.Tensor:
    """Raw (unrescaled) per-line signatures for one finalist configuration."""
    stream = build_stream(images, angles_deg, counts, points_per_line, interpolation, cubic_upsample)
    num_lines = stream.shape[1]
    return torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
    )


def run_stage_b_validation(finalists: list, n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                            seed: int = 0, verbose: bool = True) -> dict:
    """Full per-path adversarial/control evaluation for each finalist configuration: FGSM, matched
    random control, margin-difference numerator, per-line distances. Loads the shared canonical
    checkpoint once and reuses the same perturbations across every finalist. finalists is a list of
    dicts with keys name, angles_deg, counts, points_per_line, depth, interpolation."""
    torch.manual_seed(seed)
    models = {}
    for name, loader in [("SmallCNN", train_or_load_small_cnn), ("StrongCNN", train_or_load_strong_cnn)]:
        trained, _, test_acc = loader(verbose=verbose)
        models[name] = {"model": trained, "test_acc": test_acc}

    images, labels = load_eval_pool(n_per_class=n_per_class, seed=seed)
    images_c = images.unsqueeze(1)
    generator = torch.Generator().manual_seed(seed)

    # Perturbations depend only on (model, epsilon), not on Method B's
    # config - computed once, reused for every finalist below.
    perturbations = {}
    for name, info in models.items():
        model = info["model"]
        perturbations[name] = {}
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
            perturbations[name][eps] = {
                "x_adv": x_adv, "x_control": x_control,
                "num_adv": (margin_orig - margin_adv).abs(),
                "num_control": (margin_orig - margin_control).abs(),
                "flip_mask": preds_adv != labels,
            }

    results = {}
    for finalist in finalists:
        fname = finalist["name"]
        if verbose:
            print(f"Evaluating finalist: {fname}")
        angles_deg, counts = finalist["angles_deg"], finalist["counts"]
        points_per_line, depth, interpolation = finalist["points_per_line"], finalist["depth"], finalist["interpolation"]

        sig_orig_raw = _config_signatures(images, angles_deg, counts, points_per_line, depth, interpolation)
        r = choose_rescale_factor(sig_orig_raw, depth=depth)
        sig_orig = rescale_signature(sig_orig_raw, r=r, depth=depth)

        finalist_result = {"config": finalist, "r": r, "models": {}}
        for mname in models:
            model_result = {"test_acc": models[mname]["test_acc"], "eps": {}}
            for eps in epsilons:
                p = perturbations[mname][eps]

                sig_adv_raw = _config_signatures(p["x_adv"], angles_deg, counts, points_per_line, depth, interpolation)
                sig_control_raw = _config_signatures(p["x_control"], angles_deg, counts, points_per_line, depth, interpolation)
                sig_adv = rescale_signature(sig_adv_raw, r=r, depth=depth)
                sig_control = rescale_signature(sig_control_raw, r=r, depth=depth)

                dist_adv = per_line_distances(sig_orig, sig_adv)
                dist_control = per_line_distances(sig_orig, sig_control)

                ratio_adv = p["num_adv"].unsqueeze(1) / dist_adv
                ratio_control = p["num_control"].unsqueeze(1) / dist_control

                model_result["eps"][eps] = {
                    "flip_mask": p["flip_mask"],
                    "flip_fraction": p["flip_mask"].float().mean().item(),
                    "ratio_adv": ratio_adv, "ratio_control": ratio_control,
                    "dist_adv": dist_adv, "dist_control": dist_control,
                }
            finalist_result["models"][mname] = model_result
        results[fname] = finalist_result

    return {"n_images": images.shape[0], "epsilons": list(epsilons), "labels": labels, "results": results}
