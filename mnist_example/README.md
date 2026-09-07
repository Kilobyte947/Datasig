# mnist_example

Experiment 2 of the Lipschitz-diagnostics project: scales the three
estimators built for a toy regression problem (`toy_example`, sibling
folder) up to real classifiers trained on MNIST. There's no closed-form
ground-truth Lipschitz constant here — validity comes from checking that
the three sub-methods roughly agree with each other and that estimates
are stable under resampling, not from comparing against a known answer.

This folder covers two related but separately-documented topics:

- **[`distance_measures.md`](distance_measures.md)** — the core question:
  does switching from plain Euclidean distance to a Mahalanobis distance
  fit from MNIST's own pixel covariance change what the Lipschitz
  estimators recover? Covers the three main models (logistic regression,
  MLP, CNN), the layer-decomposition, UMAP, smoothing, and
  truncated-Mahalanobis sub-experiments, and every validated finding
  along the way (the near/all ratio reversal and its confirmed
  mechanism, a label-error cross-reference, the smoothing fix for a
  numerically unstable embedding).
- **[`adversarial.md`](adversarial.md)** — does the tight-vs-loose
  Lipschitz bound gap the layer-decomposition sub-experiment finds
  actually matter in practice? Generates real FGSM/PGD adversarial
  examples and checks how close their achieved sensitivity comes to
  each bound, under both distance metrics, for both `SmallCNN` (with a
  width sweep and a multi-seed confirmation) and `StrongCNN` (with its
  own multi-seed sweep and a transferability study).

## Shared model checkpoint

`SmallCNN` and `StrongCNN` each have one canonical trained checkpoint
(`models.py::train_or_load_small_cnn`/`train_or_load_strong_cnn`,
cache-aside against `checkpoints/`, gitignored) — every experiment in
this folder that wants "the" `SmallCNN`/`StrongCNN` loads this same
checkpoint rather than independently retraining a copy. This matters
most for `StrongCNN`: its BatchNorm/Dropout layers make independent
training runs not bit-reproducible even at a fixed seed (CPU
multi-threaded floating-point non-associativity), so a shared checkpoint
is the only way two experiments see literally the same model, not just
the same architecture. `signature_distance` is intended to load this
same checkpoint too — not yet wired up; tracked as a separate follow-up.
Sweeps that vary width or train multiple seeds by design (the CNN-width
sweep, both multi-seed sub-experiments) still train their own models, for
the obvious reason that they need more than one.

## File reference

| File | Contents |
|---|---|
| `data.py` | MNIST loading (`load_mnist`, `get_dev_subset`, `stratified_subset_idx`, `make_loader`). |
| `models.py` | `LogisticRegressionModel`, `SmallMLP`, `SmallCNN`, `StrongCNN`, `FlattenedInputWrapper`, `train_classifier`, `margin_fn`, and the shared-checkpoint helpers above. |
| `estimators.py` | The three Lipschitz sub-methods (pairwise, local-perturbation, gradient-norm) plus their ratio-distribution support functions. |
| `distance.py` | Euclidean and Mahalanobis distance functions, the SVD-based precision-matrix machinery (ridge and truncated-eigenvalue), `class_separation_ratio`, `knn_label_purity`. |
| `embeddings.py` | `elementwise_embedding`, `local_patch_cross_terms` — fixed (non-learned) feature maps the Mahalanobis metric can be fit over instead of raw pixels. |
| `smoothing.py` | Gaussian-blur smoothing (`gaussian_blur_embedding`, `smoothed_cross_terms_embedding`) — fixes a numerical instability found in `local_patch_cross_terms`, see `distance_measures.md`. |
| `umap_embedding.py` | A learned UMAP embedding tried as an alternative distance metric — see `distance_measures.md`'s UMAP sub-experiment for why it's not adopted (a compression artifact, not a real signal). |
| `layer_decomposition.py` | The layer-decomposition sub-experiment — see `distance_measures.md`. |
| `augmentation.py` | `random_affine_augment` — light rotation/translation augmentation, used by `StrongCNN`'s training recipe. |
| `run_experiment.py` | The main distance-measures driver — see `distance_measures.md` for what each function does. |
| `plots.py` | All plotting logic for the distance-measures side. |
| `attacks.py` | FGSM/PGD, from scratch, no dependency on distance metric. |
| `adversarial_run_experiment.py`, `adversarial_plots.py`, `adversarial_seed_sweep.py`, `adversarial_strong_cnn.py`, `adversarial_strong_cnn_seed_sweep.py` | The adversarial sub-experiment's drivers and plots — see `adversarial.md`. |
| `notebook_distance_measures.ipynb` | The distance-measures side, one notebook: the flagship LR/MLP/CNN comparison and embedding-degree sweep, then the UMAP sub-experiment, then smoothing/`radius_multiplier`/truncated-Mahalanobis as later sections. |
| `notebook_adversarial.ipynb` | The adversarial side, one notebook: layer decomposition, then SmallCNN adversarial-vs-bound (baseline, width sweep, Mahalanobis, seed pilot), then StrongCNN adversarial-vs-bound, then the StrongCNN multi-seed sweep, as sequential sections. |
| `tests/` | One test file per source module above (`test_adversarial_*.py` for the adversarial side). |
| `checkpoints/` | The shared `SmallCNN`/`StrongCNN` checkpoints (gitignored). |
| `results/` | Generated outputs (gitignored except `.gitkeep`). |
| `data/` | Downloaded MNIST files (gitignored, ~63MB) — recreated automatically by `load_mnist` on first run. |

## How to run it

```bash
# from the repo root
.venv/bin/python -m pytest mnist_example/tests/ -v

.venv/bin/python -c "from mnist_example.run_experiment import main; main()"

.venv/bin/python -c "from mnist_example.adversarial_run_experiment import main; main()"
```

See `distance_measures.md`/`adversarial.md` for the full list of opt-in sweeps and their own run
commands.
