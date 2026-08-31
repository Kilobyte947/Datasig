# Method B Adversarial / Lipschitz-Ratio Evaluation — Results

`method_b_adversarial_eval.py` (standalone, no imports from `toy_lipschitz`/
`mnist_lipschitz`). FGSM, epsilons {0.02, 0.03, 0.05}, 200 test images
(20/class), two freshly-trained classifiers (SmallCNN: architecture-matched
reimplementation of `mnist_lipschitz.models.SmallCNN`, test_acc 98.24%;
StrongCNN: reimplementation of `mnist_lipschitz.models.StrongCNN` with a
simplified training recipe — no augmentation, 3 epochs vs. the original's 25
— test_acc 99.36%, both reasonably close to Experiment 2's originals
98.65%/99.66% despite the simplification). Method B distance uses the fixed
r≈2.860 from the Phase 4 sanity check, unmodified.

## Headline numbers

| | eps | flip_frac | pixel ratio (adv) | pixel ratio (ctrl) | pixel adv/ctrl | Method B ratio (adv) | Method B ratio (ctrl) | Method B adv/ctrl |
|---|---|---|---|---|---|---|---|---|
| SmallCNN  | 0.02 | 0.035 | 2.282 | 0.251 | **9.11** | 0.477 | 0.064 | **7.42** |
| SmallCNN  | 0.03 | 0.040 | 2.269 | 0.268 | **8.47** | 0.472 | 0.067 | **7.04** |
| SmallCNN  | 0.05 | 0.070 | 2.314 | 0.273 | **8.46** | 0.485 | 0.067 | **7.20** |
| StrongCNN | 0.02 | 0.020 | 2.902 | 0.576 | **5.04** | 0.668 | 0.147 | **4.55** |
| StrongCNN | 0.03 | 0.025 | 2.987 | 0.680 | **4.39** | 0.681 | 0.181 | **3.76** |
| StrongCNN | 0.05 | 0.045 | 3.214 | 0.887 | **3.62** | 0.737 | 0.227 | **3.25** |

"ctrl" = random (non-gradient-directed) noise, matched in L2 pixel-norm to
the FGSM perturbation for the same image — the point of comparison is
whether a distance separates *genuinely adversarial* shifts from *equally
large but undirected* ones, not just any shift.

## Finding 1: both distances separate adversarial from random-noise control, clearly

Both `ratio_a` (pixel) and `ratio_b` (Method B) are bimodal, well-separated
distributions across adversarial vs. control pairs, for both models (see
`results/adv_ratio_distribution_SmallCNN.png` / `results/adv_ratio_distribution_StrongCNN.png`
— minimal overlap between the two histograms in every case). So Method B is
not just noise: it does distinguish a gradient-directed perturbation from an
equally-sized undirected one.

## Finding 2: Method B's separation is real but consistently slightly weaker than plain pixel distance

Look at the "adv/ctrl" columns: **pixel-Euclidean's separation ratio is
larger than Method B's in every one of the 6 rows above** (SmallCNN:
8.5–9.1x vs. 7.0–7.4x; StrongCNN: 3.6–5.0x vs. 3.3–4.6x). Consistent
direction, consistent gap, small sample-to-sample variation across epsilons.
Honest read: on this specific test, Method B is not (yet) beating the
pixel-space baseline it's meant to improve on — it's in the same ballpark,
a bit behind. This is the same pattern as the Phase 4 sanity check (weak,
not-yet-better-than-baseline signal), now showing up in a second,
independent test.

## Finding 3: capacity contrast is visible in both, mildly stronger in the raw ratio for Method B

StrongCNN is harder to fool at these epsilons (flip_frac 0.02–0.045 vs.
SmallCNN's 0.035–0.07) — expected, it's the higher-capacity, higher-accuracy
model. Both distances register a capacity difference: StrongCNN's ratios
(both a and b) are consistently higher in absolute terms than SmallCNN's at
matched epsilon, meaning StrongCNN's margin shifts more per unit of
perturbation-distance under either metric. Neither distance shows the other
being flatly uninformative about capacity — this isn't a place where Method B
reveals something pixel-distance completely misses, at this sample size.

## Finding 4: top-10 highest-Method-B-ratio pairs — mostly high-margin-shift, not misclassifications

At eps=0.03: SmallCNN's top 10 has **one** actual flip (`6 -> 4`); StrongCNN's
top 10 has **zero** — every other entry is a pair where the prediction didn't
change (true == adv_pred) but the margin moved a lot relative to distance.
So the highest-ratio pairs under Method B are mostly "large margin swing,
still correctly classified," not misclassifications. The one flip found
(SmallCNN, `6 -> 4`) doesn't match Experiment 2's documented CNN confusable
pairs (6/5, 8/2, 8/0) — expected per the task's own caveat (fresh training
run, different seed, not an exact reproduction), but worth being explicit
that no overlap was found, not glossing over it.

## Verdict

Method B behaves sensibly as a Lipschitz-ratio denominator — it separates
genuinely adversarial shifts from equally-sized random ones, and it's
sensitive to model capacity — but on both tests run so far (Phase 4's
within/cross-digit check, and this adversarial/control separation check) it
sits slightly *below* plain pixel-Euclidean distance, not above it.
Per the "move forward regardless" instruction, this isn't a stop condition —
but it's two independent, consistent data points now, both pointing the same
direction, worth carrying into whatever comes next (hyperparameter tuning,
PGD, or a design change) rather than treating as noise.

## Files

- `results/adv_ratio_distribution_SmallCNN.png`, `results/adv_ratio_distribution_StrongCNN.png`
  — ratio histograms, adversarial vs. control, per model.
- `results/adv_example_pairs.png` — original/perturbed image pairs with both distance
  values annotated.
