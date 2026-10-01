# Lipschitz-Constant Estimability

This repository studies a single question: **when can a trained model's worst-case sensitivity
(its Lipschitz constant) be recovered reliably from data alone, and how much does the choice of
distance metric or feature representation change the answer?**

A Lipschitz constant bounds how much a function's output can change for a given change in its
input. For a trained model, that bound is a safety guarantee - and it's exactly what determines
how vulnerable the model is to a small, deliberately crafted input change (an adversarial
attack). This repo asks whether that guarantee can actually be trusted when it's estimated from a
model and its training data, rather than known in closed form, and tracks down several concrete
ways in which it can't be, absent a careful choice of metric, representation, and validation.

**New to this project? Start with [`notebook_everything.ipynb`](notebook_everything.ipynb)** - a
single notebook that walks through the whole investigation end to end, in plain language, with
real numbers re-computed live. It assumes no prior familiarity with the codebase or with
Lipschitz constants.

## Repository structure

The investigation runs in three stages, each its own top-level package, each with its own
detailed README:

| Package | Question | Ground truth available? |
|---|---|---|
| [`toy_example/`](toy_example/README.md) | Validate the measurement methodology itself on a synthetic regression problem | Yes - closed-form `L*` |
| [`mnist_example/`](mnist_example/README.md) | Scale the same estimators to real classifiers (logistic regression, MLP, CNN) trained on MNIST, and test the adversarial-attack connection directly | No - validity comes from cross-method agreement and resampling stability |
| [`signature_distance/`](signature_distance/README.md) | Ask whether a fundamentally different feature representation (path signatures instead of raw pixels) changes the conclusions | No - reuses `mnist_example`'s checkpoints and framework |

Other things at the repo root:
- **`notebook_everything.ipynb`** - a curated, newcomer-accessible tour across all three packages
  above, re-running a representative subset of each package's own driver functions live. The best
  starting point if you're new to the project.
- **`CLAUDE.md`** - conventions, architecture notes, and non-obvious design rationale for anyone
  (human or AI) editing this codebase. Read it before making non-trivial changes.
- **`.venv/`** - a committed virtual environment with all dependencies (torch, torchvision,
  numpy, scikit-learn, matplotlib, pandas, pytest, jupyter) already installed. There's no
  `pyproject.toml`/`requirements.txt` - always invoke tools through `.venv/bin/python`/
  `.venv/bin/pytest` rather than a bare `python`/`pytest`.

Each package also follows the same internal module split (`estimators.py`, `data.py`,
`models.py`, `plots.py`, `run_experiment.py`, `tests/`) - see `CLAUDE.md`'s "Shared architecture
across experiments" section for the conventions that are common across all three rather than
repeated per package.

## Running things

```bash
# run all tests for one package
.venv/bin/python -m pytest toy_example/tests/ -v
.venv/bin/python -m pytest mnist_example/tests/ -v
.venv/bin/python -m pytest signature_distance/tests/ -v

# run a package's full experiment driver
.venv/bin/python -c "from toy_example.run_experiment import main; main()"
.venv/bin/python -c "from mnist_example.run_experiment import main; main()"

# execute a notebook end-to-end (regenerates results/ and plots in place)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook_everything.ipynb
```

See each package's own README for its specific sub-experiments and opt-in (slower) driver
functions - not everything is wired into `main()`.

## What's been found so far

Full derivations and every numerical table live in each package's README and in
`notebook_everything.ipynb`'s closing synthesis; this is just the headline.

- **Undersampling causes trained models to undershoot their true worst-case sensitivity, and this
  doesn't resolve with more data or more model capacity.** Shown directly against a known ground
  truth in `toy_example`; the same qualitative pattern recurs in `mnist_example` and
  `signature_distance`, neither of which has ground truth to check it against directly.
- **A better distance metric is not a uniform fix.** In `toy_example`, switching from Euclidean to
  a well-chosen Mahalanobis metric cuts error against the true Lipschitz constant from 19% to
  under 1% - but on real classifiers, the same switch helps three independent estimators by wildly
  different amounts, flips the sign of at least one real effect, and even reverses which direction
  it helps in depending on model architecture (`mnist_example`'s adversarial-bound comparison).
- **A different feature representation doesn't straightforwardly do better either.** Two of the
  three path-signature constructions tried in `signature_distance` trail the plain
  pixel-Euclidean baseline they were built to beat; the one that wins needed two rounds of tuning,
  and its cheap screening metric and its expensive validation metric disagreed about which
  configuration was actually best.
- **The throughline isn't any single winning metric or representation - it's that every one of
  them needed independent validation before being trusted.** See "Test methodology -
  checkpoint-gating" in `CLAUDE.md` for how that discipline is enforced throughout this codebase.

## Contributing / editing this codebase

Read `CLAUDE.md` first. It documents the shared architecture, the checkpoint-gating test
methodology (an estimator or numerical convention is never wired into a package's default
pipeline until it's been checked against an independent closed-form or analytic identity), and
conventions that aren't obvious from the code alone (e.g. why `toy_example` uses
`torch.float64` everywhere, why `mnist_example/distance.py` never forms a `(784, 784)` covariance
matrix directly).
