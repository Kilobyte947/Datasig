# Method A Adversarial / Lipschitz-Ratio Evaluation — Results

`method_a_adversarial_eval.py` (standalone, no imports from `toy_lipschitz`/
`mnist_lipschitz`; reuses `method_b_adversarial_eval.py`'s model/training/
attack infrastructure unmodified). FGSM, two epsilon ranges tested in
`notebook_method_a_adversarial_eval.ipynb`: the original imperceptible range
{0.02, 0.03, 0.05} (Findings 1-5 below, directly comparable to Method B's
own evaluation), and a much larger sweep {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}
run afterward to see how the picture changes well past the imperceptible
regime (Findings 6-9). 200 test images (20/class) in both cases, two
freshly-trained classifiers per run (SmallCNN test_acc 98.26%, StrongCNN
test_acc 99.34% — both retrained fresh here with the same seed/recipe as
`method_b_adversarial_eval.py`'s run, not reused from it, but statistically
indistinguishable from its documented 98.24%/99.36%: training is independent
of which distance denominator is evaluated downstream; the larger-epsilon
sweep retrains fresh again with the same recipe, so its own reported
accuracies match too). Method A distance uses the fixed r≈1.656 from the
Phase 4 sanity check, unmodified in both runs.

## Headline numbers

| | eps | flip_frac | pixel ratio (adv) | pixel ratio (ctrl) | pixel adv/ctrl | Method A ratio (adv) | Method A ratio (ctrl) | Method A adv/ctrl |
|---|---|---|---|---|---|---|---|---|
| SmallCNN  | 0.02 | 0.035 | 2.274 | 0.249 | **9.14** | 1.782 | 0.230 | **7.75** |
| SmallCNN  | 0.03 | 0.040 | 2.257 | 0.266 | **8.47** | 1.744 | 0.248 | **7.04** |
| SmallCNN  | 0.05 | 0.065 | 2.294 | 0.273 | **8.41** | 1.775 | 0.239 | **7.42** |
| StrongCNN | 0.02 | 0.020 | 2.759 | 0.551 | **5.00** | 1.996 | 0.471 | **4.24** |
| StrongCNN | 0.03 | 0.035 | 2.808 | 0.654 | **4.29** | 1.995 | 0.526 | **3.80** |
| StrongCNN | 0.05 | 0.045 | 2.953 | 0.812 | **3.64** | 2.114 | 0.726 | **2.91** |

"ctrl" = random (non-gradient-directed) noise, matched in L2 pixel-norm to
the FGSM perturbation for the same image — same convention as
`adversarial_eval_summary.md`.

## Finding 1: both distances separate adversarial from random-noise control, clearly

Both the pixel and Method A ratios are bimodal, well-separated distributions
across adversarial vs. control pairs, for both models (see the histogram
cell in `notebook_method_a_adversarial_eval.ipynb` — minimal overlap in
every case). Method A is not just noise: it distinguishes a
gradient-directed perturbation from an equally-sized undirected one, same as
Method B.

## Finding 2: Method A's separation is real but consistently slightly weaker than plain pixel distance

Look at the "adv/ctrl" columns: **pixel-Euclidean's separation ratio is
larger than Method A's in every one of the 6 rows above** (SmallCNN:
8.4–9.1x vs. 7.0–7.8x; StrongCNN: 3.6–5.0x vs. 2.9–4.2x). Same direction,
same consistency, as Method B's finding and the Phase 4 sanity check — a
third independent test now pointing the same way for Method A specifically.

## Finding 3: Method A vs. Method B, directly (not just architecture-level)

Because both modules train `SmallCNN`/`StrongCNN` with the identical
seed/recipe, and training is independent of which distance is evaluated
against the trained model afterward, the two runs' models end up
statistically indistinguishable (test_acc within 0.02pp, flip fractions
matching almost exactly — e.g. SmallCNN eps=0.03: `flip_frac=0.040` in both
runs). That makes a direct, same-numerator comparison meaningful, not just
an architecture-level one:

| model | eps | pixel a/c | Method A a/c | Method B a/c |
|---|---|---|---|---|
| SmallCNN | 0.02 | 9.14 | **7.75** | 7.42 |
| SmallCNN | 0.03 | 8.47 | **7.04** | 7.04 |
| SmallCNN | 0.05 | 8.41 | **7.42** | 7.20 |
| StrongCNN | 0.02 | 5.00 | **4.24** | 4.55 |
| StrongCNN | 0.03 | 4.29 | **3.80** | 3.76 |
| StrongCNN | 0.05 | 3.64 | **2.91** | 3.25 |

Method A's adv/ctrl ratio is higher than Method B's on all 3 SmallCNN
epsilons and 1 of 3 StrongCNN epsilons (4/6 rows) — a narrow, inconsistent
edge, not a clean win for either method. Both sit in the same ballpark,
both consistently below the pixel-Euclidean baseline. Nothing here suggests
one method's signature distance is a categorically better adversarial-shift
denominator than the other's.

## Finding 4: capacity contrast is visible in both distances

StrongCNN is harder to fool at these epsilons (flip_frac 0.020–0.045 vs.
SmallCNN's 0.035–0.065) — expected, the higher-capacity, higher-accuracy
model. Both distances register the capacity difference: StrongCNN's ratios
(pixel and Method A) are consistently higher in absolute terms than
SmallCNN's at matched epsilon. Same pattern documented for Method B.

## Finding 5: top-10 highest-Method-A-ratio pairs — same shape as Method B, different specific pair

At eps=0.03: SmallCNN's top 10 has **one** actual flip (`2 -> 7`); StrongCNN's
top 10 has **zero** — matching Method B's documented shape (1 flip for
SmallCNN, 0 for StrongCNN) exactly, even though the *specific* flipped pair
differs (Method B's was `6 -> 4`) — expected, since the two runs' distance
rankings differ even when the underlying trained models and adversarial
images are nearly identical. Neither this run's flip (`2 -> 7`) nor Method
B's (`6 -> 4`) matches Experiment 2's documented CNN confusable pairs (6/5,
8/2, 8/0) — same honest non-overlap as Method B's run, not glossed over.

## Larger-epsilon sweep (0.05-0.3): beyond the imperceptible regime

Same protocol, same models retrained fresh with the identical seed/recipe,
now run at epsilons {0.05, 0.1, 0.15, 0.2, 0.25, 0.3} — at eps=0.3 a
perturbation can shift any pixel by up to 30% of the full intensity range,
no longer imperceptible (confirmed directly in the notebook's example
gallery at eps=0.3). No Method B comparison here — Method B has only ever
been evaluated at 0.02/0.03/0.05, so there is nothing documented to compare
against at these epsilons.

### Headline numbers: larger-epsilon sweep

| | eps | flip_frac | pixel ratio (adv) | pixel ratio (ctrl) | pixel adv/ctrl | Method A ratio (adv) | Method A ratio (ctrl) | Method A adv/ctrl |
|---|---|---|---|---|---|---|---|---|
| SmallCNN  | 0.05 | 0.065 | 2.294 | 0.266 | **8.63** | 1.775 | 0.219 | **8.11** |
| SmallCNN  | 0.10 | 0.190 | 2.319 | 0.307 | **7.56** | 1.850 | 0.268 | **6.89** |
| SmallCNN  | 0.15 | 0.495 | 2.349 | 0.362 | **6.48** | 1.954 | 0.311 | **6.29** |
| SmallCNN  | 0.20 | 0.790 | 2.308 | 0.412 | **5.60** | 2.022 | 0.366 | **5.53** |
| SmallCNN  | 0.25 | 0.950 | 2.233 | 0.451 | **4.95** | 2.067 | 0.436 | **4.74** |
| SmallCNN  | 0.30 | 0.995 | 2.147 | 0.515 | **4.17** | 2.104 | 0.511 | **4.12** |
| StrongCNN | 0.05 | 0.045 | 2.953 | 0.795 | **3.71** | 2.114 | 0.699 | **3.02** |
| StrongCNN | 0.10 | 0.170 | 3.033 | 1.107 | **2.74** | 2.313 | 1.002 | **2.31** |
| StrongCNN | 0.15 | 0.355 | 2.718 | 1.277 | **2.13** | 2.192 | 1.165 | **1.88** |
| StrongCNN | 0.20 | 0.585 | 2.383 | 1.273 | **1.87** | 2.039 | 1.162 | **1.75** |
| StrongCNN | 0.25 | 0.740 | 2.090 | 1.312 | **1.59** | 1.901 | 1.271 | **1.50** |
| StrongCNN | 0.30 | 0.850 | 1.838 | 1.254 | **1.47** | 1.781 | 1.202 | **1.48** |

(Note the eps=0.05 row here differs slightly from Finding 1-5's table above
in the `ctrl` columns and adv/ctrl ratio — `flip_fraction` and `ratio(adv)`
match exactly (both depend only on the model/FGSM, unaffected by epsilon
ordering), but `random_noise_perturbation`'s generator is seeded once per
`run_adversarial_evaluation` call and drawn from sequentially across the
epsilon loop, so eps=0.05 being the 3rd epsilon in Finding 1-5's tuple
`(0.02, 0.03, 0.05)` vs. the 1st in this sweep's `(0.05, 0.1, ...)` draws a
different random control each time even with the same seed; the effect on
the adv/ctrl ratio is small, 8.41 vs. 8.63 for SmallCNN pixel.)

### Finding 6: flip fraction saturates fast

SmallCNN goes from 6.5% flipped at eps=0.05 to 99.5% at eps=0.3; StrongCNN
goes from 4.5% to 85.0%. StrongCNN stays consistently harder to fool at
every epsilon in the sweep — the same capacity contrast as Finding 4, now
visible across a much wider range.

### Finding 7: adv/control separation collapses monotonically as epsilon grows, for both distances, on both models

Every ratio falls monotonically as epsilon increases: SmallCNN pixel a/c
8.63→7.56→6.48→5.60→4.95→4.17; Method A a/c 8.11→6.89→6.29→5.53→4.74→4.12.
Same monotonic collapse for StrongCNN (pixel 3.71→...→1.47; Method A
3.02→...→1.48). This tracks the flip fraction directly: once the
perturbation is large enough that an equally-sized *random* shift starts
flipping predictions on its own, the control stops being a clean negative,
and the adversarial-vs-random distinction both distances are measuring
erodes.

### Finding 8: the pixel-vs-Method-A gap narrows sharply as epsilon grows — and briefly flips

At eps=0.05 pixel still leads clearly (SmallCNN +0.52, StrongCNN +0.69). By
eps=0.3 the SmallCNN gap has shrunk to near-zero (4.17 vs. 4.12), and for
StrongCNN specifically **Method A's ratio (1.48) narrowly exceeds pixel's
(1.47)** — the first time in this entire project (Phase 4, Method B's
evaluation, or Method A's own small-epsilon section above) that a signature
distance has matched or beaten the pixel-Euclidean baseline on this
comparison. Flagged plainly, not overclaimed: a 0.01 gap on a ~1.5 scale
from 200 images is well within sampling noise, and it occurs at the one
condition (85% flip rate) where the comparison is least informative to
begin with — both ratios are being compressed toward each other by the same
saturation effect documented in Finding 7, not obviously a genuine
capability edge for Method A.

### Finding 9: the top-10 highest-ratio pairs flip from mostly-correct to almost-all-flips

At eps=0.3, the top-10 highest-Method-A-ratio pairs are 9/10 genuine flips
for SmallCNN and 9/10 for StrongCNN — a complete reversal from Finding 5's
small-epsilon result (at most 1/10 flips there). Expected given Finding 6's
flip fractions (85-99.5% at this epsilon): the highest-ratio examples are no
longer "large margin swing, still correct" but genuinely misclassified
images.

## Verdict

Method A behaves sensibly as a Lipschitz-ratio denominator — it separates
genuinely adversarial shifts from equally-sized random ones, and it's
sensitive to model capacity — but like Method B, it sits slightly *below*
plain pixel-Euclidean distance at imperceptible epsilons, not above it.
Directly compared against Method B (same trained-model behaviour, same
numerator), Method A is neither clearly better nor clearly worse — a narrow
4/6 edge in adv/ctrl ratio that doesn't survive as a consistent pattern
across both models. Three independent tests now (Phase 4, Method B's
adversarial eval, this one) all land in the same place for both methods at
small epsilons: sensible, capacity-aware, but not yet beating the
pixel-space baseline either is meant to improve on.

The larger-epsilon sweep adds one more honest data point rather than
overturning that picture: pixel-Euclidean's advantage over Method A erodes
steadily and nearly vanishes by eps=0.3, but that convergence tracks the
control condition becoming uninformative as flip fraction saturates
(Finding 7), not clear evidence Method A becomes the *better* denominator at
large epsilon. The one StrongCNN crossover (Finding 8) is noted as an honest
observation, not a result to build on without a larger sample. Per the
project's "move forward regardless, but carry the finding" convention, both
the small- and large-epsilon results are carried forward as-is, not treated
as a stop condition and not smoothed into a single tidier narrative.
