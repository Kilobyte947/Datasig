"""Method A adversarial / Lipschitz-ratio evaluation - standalone.

Direct counterpart to `method_b_adversarial_eval.py`: same evaluation
(train SmallCNN/StrongCNN fresh, FGSM adversarial perturbations + a
magnitude-matched random-noise control, compare the resulting Lipschitz
ratio - margin change / distance - under plain pixel-Euclidean distance vs.
the method's own signature distance), but the signature distance is now
Method A's (patch singular-value stream over 64 fixed pixel locations)
instead of Method B's (16 fixed reference lines). README.md flagged this as
missing ("Method A has no adversarial/Lipschitz-ratio evaluation counterpart
to Method B's") - this module fills that gap.

Reuses `method_b_adversarial_eval.py`'s model/training/attack infrastructure
UNMODIFIED rather than re-deriving it - same pattern already used by
`per_path_adversarial_eval.py` (SmallCNN, StrongCNN, train_classifier,
margin, fgsm_attack, random_noise_perturbation, pixel_euclidean_distance,
load_mnist_train_test are all imported, not redefined). That infrastructure
is method-agnostic (classifier architecture + attack, independent of which
distance denominator is evaluated against it), so duplicating it here would
just be drift risk with no benefit. This keeps the same transitive isolation
from `toy_lipschitz`/`mnist_lipschitz` that module documents - nothing here
imports from either.

Field names deliberately do NOT reuse method_b_adversarial_eval's
`denom_a_*`/`denom_b_*`/`ratio_a_*`/`ratio_b_*` convention (there, "a"/"b"
label the two distances under test - (a) pixel, (b) Method B - not "Method
A"/"Method B"). Reusing that convention here would make "ratio_b" mean
"Method A's ratio" in this file, which is exactly the confusion worth
avoiding - so this module uses `denom_pixel_*`/`denom_sig_*` and
`ratio_pixel_*`/`ratio_sig_*` instead, unambiguous regardless of which
sibling module the reader has open.
"""

from pathlib import Path

import torch

from signature_distance.data_pool import load_eval_pool
from signature_distance.distances import method_a_feature_vector, rescale_signature
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
from signature_distance.streams import make_pixel_order, patch_sv_stream

RESULTS_DIR = Path(__file__).parent / "results"
SIGNATURE_DEPTH = 4
# Fixed from the Phase 4 sanity check (run_experiment.sanity_check_demo,
# seed=0), not re-derived here - Method A's distance function isn't being
# changed by this evaluation, only tested. Matches README.md's documented
# r_A ~= 1.656.
METHOD_A_R = 1.6562803550838685
METHOD_A_PIXEL_ORDER = make_pixel_order(k=64, seed=0)  # fixed geometry, shared across every call


def method_a_signature_distance(images1: torch.Tensor, images2: torch.Tensor,
                                 depth: int = SIGNATURE_DEPTH, r: float = METHOD_A_R) -> torch.Tensor:
    """Method A's own pipeline - make_pixel_order (once, module-level) ->
    patch_sv_stream -> signature_of_stream -> rescale_signature ->
    method_a_feature_vector (identity) -> Euclidean. Reuses Method A's own
    code unmodified; nothing here is reimplemented.

    images1, images2: (N, 28, 28) float32 in [0, 1].
    """
    def _feature_vector(images):
        stream = patch_sv_stream(images, METHOD_A_PIXEL_ORDER)  # (N, 64, 2)
        sig = signature_of_stream(stream, depth=depth)  # (N, sig_dim)
        sig = rescale_signature(sig, r=r, depth=depth)
        return method_a_feature_vector(sig)  # (N, sig_dim) - identity

    vec1 = _feature_vector(images1)
    vec2 = _feature_vector(images2)
    return (vec1 - vec2).norm(dim=1)


def run_adversarial_evaluation(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05),
                                seed: int = 0, cnn_epochs: int = 3, strong_epochs: int = 3,
                                device: str = "cpu", verbose: bool = True) -> dict:
    """Train SmallCNN and StrongCNN fresh, then for a sample of test images
    (n_per_class per digit, from data_pool.load_eval_pool) and each epsilon:
    generate an FGSM adversarial perturbation and a magnitude-matched random
    control perturbation, compute the margin-change numerator, both
    denominators (pixel-Euclidean and Method A signature distance), and the
    resulting ratios. Returns a nested dict, one entry per model - same
    sample/model/attack protocol as
    method_b_adversarial_eval.run_adversarial_evaluation (retrained fresh
    here rather than reusing that run's models/checkpoints, matching
    per_path_adversarial_eval.py's precedent for a self-contained module).
    """
    torch.manual_seed(seed)
    train_loader, test_loader = load_mnist_train_test()

    models = {}
    for name, model, epochs in [("SmallCNN", SmallCNN(), cnn_epochs), ("StrongCNN", StrongCNN(), strong_epochs)]:
        if verbose:
            print(f"Training {name} ({epochs} epochs)...")
        trained, train_acc, test_acc = train_classifier(
            model, train_loader, test_loader, epochs=epochs, device=device, verbose=verbose
        )
        trained.eval()
        models[name] = {"model": trained, "train_acc": train_acc, "test_acc": test_acc}
        if verbose:
            print(f"  {name}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

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
