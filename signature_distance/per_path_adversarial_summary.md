# Method B: Per-Path Lipschitz Ratios Under Adversarial Perturbation

Corrects the framing of the last two tasks: the 16 reference-line paths are
never merged into any single score here (no 496-dim concatenation, no
max/top-k/weighted-sum) — each produces its own local ratio
`margin_difference / line_i_distance`, examined as a collection of 16
numbers per pair, the same way the pixel-space Lipschitz work (Experiment
1/2) treats individual pixels. Same 200-image sample, same two freshly
trained models (SmallCNN 98.24% test acc, StrongCNN 99.36%), same three
FGSM epsilons and magnitude-matched random control as the existing merged
adversarial evaluation — nothing in `distances.py`, `method_b_adversarial_eval.py`,
or `per_line_diagnostics.py` was changed to build this.

Primary reporting is restricted to the 12 **informative lines** (excludes
the 4 border lines — row/col exactly 0 or 27 — already shown by
`per_line_diagnostics.py` to sit at exactly AUC=0.500, chance level). Full
16-line data is in the saved result object; the write-up below covers the
12-line informative subset, with `line_6` (the single best individual
performer from that diagnostic) highlighted throughout.

## Finding 1: per-line ratios separate adversarial from control far more sharply than the merged distance did

For every one of the 12 informative lines, in every one of 6
model×epsilon combinations (72 comparisons total, no exceptions): mean
ratio on genuinely adversarial (flipped-prediction) pairs is higher than
on magnitude-matched random-control pairs from the *same* images.

| model | eps | n_flipped/200 | mean fold-ratio (adv/control) across 12 lines | line_6 fold-ratio |
|---|---|---|---|---|
| SmallCNN | 0.02 | 7 | **10.81x** | 19.98x |
| SmallCNN | 0.03 | 8 | **9.36x** | 10.53x |
| SmallCNN | 0.05 | 14 | **10.54x** | 15.71x |
| StrongCNN | 0.02 | 4 | **11.27x** | 7.25x |
| StrongCNN | 0.03 | 5 | **4.18x** | 5.70x |
| StrongCNN | 0.05 | 9 | **4.18x** | 4.27x |

Compare this to the merged 496-dim distance's adv/control separation from
the earlier evaluation (adv/ctrl ratio of raw means, all pairs, not
flip-restricted): 7.0–7.4x (SmallCNN), 3.3–4.6x (StrongCNN). The per-line,
flipped-only numbers here (4–11x mean, individual lines up to ~20x) are
comparable to noticeably larger than that — a real improvement in
separation clarity when the paths are kept separate, though see the
sample-size caveat below before treating this as conclusively better.

**Caveat, stated plainly**: `n_flipped` is small (4–14 out of 200) — FGSM
at these conservative epsilons doesn't fool either model often. These
fold-ratios come from few examples and are correspondingly noisy. What's
trustworthy is the *direction* (always adv > control, 72/72, no
exceptions) — the exact magnitudes should be treated as indicative, not
precise, until run on a larger flipped sample (e.g. a higher epsilon
sweep to get more flips, out of scope here).

## Finding 2: adversarial perturbations concentrate on fewer paths than random noise — modestly, but with zero exceptions

For every pair, the informative line with the largest ratio was
identified (argmax over the 12), then the *distribution* of which line
wins was compared, adversarial vs. control, via entropy (max = log2(12) =
3.585 bits; lower = more concentrated on fewer lines):

| model | eps | entropy, adversarial (bits) | entropy, control (bits) |
|---|---|---|---|
| SmallCNN | 0.02 | 3.042 | 3.307 |
| SmallCNN | 0.03 | 2.993 | 3.243 |
| SmallCNN | 0.05 | 2.972 | 3.293 |
| StrongCNN | 0.02 | 3.009 | 3.274 |
| StrongCNN | 0.03 | 3.021 | 3.173 |
| StrongCNN | 0.05 | 3.033 | 3.254 |

**Adversarial entropy is lower than control entropy in all 6 of 6
combinations** — a small but completely consistent effect (~0.2–0.3 bits
out of a 3.585-bit max). This is real, if modest, evidence for the
"concentrated, not diffuse" hypothesis: FGSM's perturbation does spike a
smaller subset of paths more consistently than an equally-large random
shift does, even though it perturbs every pixel.

## An important confound, checked directly rather than assumed away

Two lines (`line_9`, `line_14`) win the argmax far more often than the
rest — in almost every model/epsilon combination, roughly 2–3x more often
than the next-most-frequent line, under **both** adversarial *and*
control conditions. Checking why: these two lines have systematically
smaller mean signature distances than the other 10 informative lines,
regardless of perturbation type (e.g. SmallCNN eps=0.03, control:
line_9=0.276, line_14=0.279, vs. 0.31–0.69 for the rest). A smaller
denominator makes a line more likely to win the argmax by construction,
independent of whether anything adversarially meaningful happened there.

So: **raw "which line spikes" counts are confounded by each line's own
baseline distance scale** and shouldn't be read as "these are the
adversarially special lines" on their own. The entropy comparison above
is the more trustworthy summary, since it's a within-condition
(adv-vs-its-own-control) comparison rather than a raw popularity count —
though it isn't fully immune to the same confound either, since a
generically low-distance line could depress entropy in both conditions
similarly (partially, not entirely, why the adv/control gap remains the
cleaner signal to read).

## Gallery: does the spike look "stroke-relevant"?

Two example flipped pairs (`results/spike_gallery_smallcnn_0.png`,
`results/spike_gallery_smallcnn_1.png`), image, perturbed image, and the spiking
line highlighted in red over all 16 lines in gray:

- Pair 45 (digit "7", eps=0.03): `line_9` (vertical, col≈3.9) spikes,
  ratio 6.94 — this line does cross through part of the stroke (the
  digit's horizontal bar).
- Pair 131 (digit "6", eps=0.03): `line_6` (horizontal, row≈23) spikes,
  ratio 15.21 — this line runs mostly through **background** below the
  digit's loop, not directly through visible ink.

**Mixed, stated plainly rather than oversold**: the first example is
consistent with "the spike lands on stroke-relevant structure," the
second isn't obviously so from a purely visual read. Two examples aren't
enough to conclude a clean stroke-relevance pattern either way — worth
checking on more examples before drawing a firm conclusion here.

## Verdict

Keeping the 16 paths separate (never merging them) surfaces a
consistently stronger, more legible adversarial-vs-control separation
than the merged 496-dim distance showed, and a small but zero-exception
"more concentrated under attack" entropy effect. Both are genuine,
positive findings for the per-path framing — but two honest caveats sit
alongside them: the flipped-pair sample sizes are small, and the raw
"which specific line spikes" question is confounded by generic per-line
distance-scale differences that show up under random control too, not
just under attack. Per this task's scope: no new merged-score
construction, no hyperparameter sweep, and nothing in the existing
merged-distance pipeline, adversarial eval, or AUC diagnostic was touched
— this is additive, sitting alongside those results, not replacing them.

## Files

- `results/spike_gallery_smallcnn_0.png`, `results/spike_gallery_smallcnn_1.png` — example
  galleries.
- `per_path_adversarial_eval.py` — the new module (per-line signature
  helper, evaluation driver, summary/spike-analysis functions, gallery
  plot).
