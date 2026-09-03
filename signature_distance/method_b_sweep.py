"""Stage 8 sweep: Method B hyperparameter optimisation (geometry,
points/line, truncation depth, interpolation) - Stage A cheap screen (no
model training, per-line same/different-digit AUC) and Stage B full
validation (per-path adversarial/control evaluation on finalists).

Reuses existing signature/distance/adversarial infrastructure:
`make_reference_lines`/`line_stream` (streams.py), `signature_of_stream`
(signatures.py), `choose_rescale_factor`/`rescale_signature`/
`per_line_distances`/`auc_for_distance` (distances.py), and
`SmallCNN`/`StrongCNN`/`train_classifier`/`fgsm_attack`/`margin`/
`random_noise_perturbation` (method_b_adversarial_eval.py). This module
adds new stream-construction variants (a cubic-spline refinement on top of
the existing linear stream) and new sweep/scoring orchestration.
`per_line_aucs` below and `per_line_diagnostics.py` both need the same
same/different-digit AUC computation (one hardcoded to the default
geometry, one generalized over the sweep grid) - factored into
`distances.auc_for_distance`, a small shared addition, rather than each
keeping its own copy.

Key efficiency fact this sweep relies on, verified directly rather than
assumed: a depth-D truncated signature's first `1+2+...+2**d` coefficients
(d <= D) are bit-identical to the depth-d signature computed directly -
truncating a tensor-algebra signature to a lower level never changes the
lower-level terms. So each (geometry, points, interpolation) stream only
needs the expensive signature step run ONCE, at the maximum depth swept;
all lower depths are prefix-sliced from that one result, not recomputed.
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import (
    auc_for_distance,
    choose_rescale_factor,
    per_line_distances,
    rescale_signature,
)
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

WIDTH = 2


def _level_sizes(depth: int) -> list:
    return [WIDTH ** n for n in range(depth + 1)]


def signature_dim(depth: int) -> int:
    return sum(_level_sizes(depth))


def cubic_spline_refine(stream_one_line: torch.Tensor, upsample_factor: int = 8) -> torch.Tensor:
    """Given one line's linear stream (N, K, 2) [t, value] (from the
    existing, unmodified `line_stream`), fit a natural cubic spline through
    each image's K points and resample at `upsample_factor * K` points.
    `signature_of_stream` (unmodified) still only ever computes a
    piecewise-LINEAR path signature - feeding it this finely-resampled,
    smooth curve is a standard way to approximate the signature of a
    genuinely curved (cubic-spline) path, since the piecewise-linear
    signature of a sufficiently fine discretization converges to the true
    curve's signature. Does not change signature dimension (that depends
    only on width/depth, not point count).
    """
    n, k, _ = stream_one_line.shape
    t = stream_one_line[:, :, 0].numpy()
    v = stream_one_line[:, :, 1].numpy()

    fine_t_frac = np.linspace(0.0, 1.0, k * upsample_factor)
    out = np.empty((n, k * upsample_factor, 2), dtype=np.float32)
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
    """(N, num_lines, K', 2) stream for one Stage-A config - reuses
    make_reference_lines/line_stream unmodified for the base (linear)
    construction; cubic_spline_refine (new, above) on top for the "cubic"
    variant."""
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
    """Same/different-digit AUC per line - uses the same shared
    `distances.auc_for_distance` helper `per_line_diagnostics.py` does
    (that module is hardcoded to the default geometry/depth=4; this
    version works for any geometry/depth, hence the separate call site)."""
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
    """Builds the stream once, computes the signature ONCE at max_depth,
    then for every depth in `depths` (all <= max_depth) prefix-slices that
    single computation (verified exact, see module docstring) and reports
    per-line AUCs, rescaled the same way the existing pipeline does
    (r re-derived per depth via choose_rescale_factor, unmodified).
    """
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
    """Full joint sweep (per README.md's Stage 8 note: "sweep together, not
    staged one-at-a-time") over geometry x points x depth x interpolation.
    Depth is handled cheaply via the max-depth-then-slice shortcut above,
    so the actual expensive-computation grid is geometry x points x
    interpolation (4 x 4 x 2 = 32 stream/signature builds), each scored at
    all 5 depths (160 total scored configs).
    """
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
    """Raw (unrescaled) per-line signatures for one finalist config."""
    stream = build_stream(images, angles_deg, counts, points_per_line, interpolation, cubic_upsample)
    num_lines = stream.shape[1]
    return torch.stack(
        [signature_of_stream(stream[:, i], depth=depth) for i in range(num_lines)], dim=1
    )


def run_stage_b_validation(finalists: list, n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                            seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                            verbose: bool = True) -> dict:
    """Full per-path adversarial/control evaluation for each finalist
    config, same framework as per_path_adversarial_eval.py (FGSM, matched
    random control, margin-difference numerator, per-line distances -
    reused unmodified via method_b_adversarial_eval.py/distances.py
    imports), applied to each finalist's own stream construction instead
    of the fixed baseline geometry.

    SmallCNN/StrongCNN are trained ONCE (they don't depend on Method B's
    geometry) and reused across every finalist - and the FGSM/control
    perturbations (also model/epsilon-dependent only, not Method-B-config-
    dependent) are likewise computed once per model/epsilon and reused,
    not regenerated per finalist.

    finalists: list of dicts with keys name, angles_deg, counts,
    points_per_line, depth, interpolation.
    """
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
