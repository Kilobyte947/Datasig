# Method B: Per-Line Distance Breakdown + Same/Different-Digit AUC

Diagnostic testing the method's original "path by path" framing (compare
reference line *i* to reference line *i* directly, ask whether images are
close on **each** path) against the merged 496-dim concatenated distance
already built and documented (Phase 3/4, adversarial eval). Additive only —
the merged pipeline is unchanged.

Same 300-image sample as Phase 4 (30/class, seed 0), all 44,850 unique
pairs, r ≈ 2.860 (same derivation as before — `choose_rescale_factor` on
this sample). For each of 17 distance measures (16 individual lines +
merged), −distance is used as a same/different-digit classifier score and
scored with ROC AUC.

## Ranking (full)

| rank | measure | AUC | FPR @ 90% TPR | orientation |
|---|---|---|---|---|
| 1 | **line_6** | **0.6394** | 0.8392 | horizontal (row=23.1) |
| 2 | merged | 0.6315 | 0.8257 | — |
| 3 | line_5 | 0.6069 | 0.8161 | horizontal (row=19.3) |
| 4 | line_1 | 0.5969 | 0.8253 | horizontal (row=3.9) |
| 5 | line_10 | 0.5906 | 0.8545 | vertical (col=7.7) |
| 6 | line_4 | 0.5900 | 0.8492 | horizontal (row=15.4) |
| 7 | line_13 | 0.5705 | 0.8823 | vertical (col=19.3) |
| 8 | line_3 | 0.5685 | 0.8781 | horizontal (row=11.6) |
| 9 | line_2 | 0.5670 | 0.8682 | horizontal (row=7.7) |
| 10 | line_12 | 0.5602 | 0.8762 | vertical (col=15.4) |
| 11 | line_11 | 0.5578 | 0.8752 | vertical (col=11.6) |
| 12 | line_14 | 0.5310 | 0.8865 | vertical (col=23.1) |
| 13 | line_9 | 0.5092 | 0.8973 | vertical (col=3.9) |
| 14 | line_0 | 0.5000 | 1.0000 | horizontal (row=0.0) |
| 15 | line_7 | 0.5000 | 0.9933 | horizontal (row=27.0) |
| 16 | line_8 | 0.5000 | 1.0000 | vertical (col=0.0) |
| 17 | line_15 | 0.5000 | 0.9933 | vertical (col=27.0) |

See `results/per_line_auc_ranking.png`.

## Direct question: does any individual line beat the merged distance?

**Yes, but only one of 16, and narrowly.** `line_6` (a horizontal line
roughly 4/5 of the way down the image, row≈23) reaches AUC 0.6394 vs.
merged's 0.6315 — a real but small margin (+0.008). `line_5` is close
behind merged (0.607 vs 0.632) but doesn't beat it. **13 of the 16
individual lines score below the merged distance.** So this is not evidence
of widespread dilution across most lines — it's evidence that concatenation
is *close to* as good as the single best line, with one line edging it out
narrowly, not swamped by noise from the other 15 the way the "signal
averaged into invisibility" hypothesis would predict if it applied broadly.

**One place that hypothesis clearly does apply**: the 4 edge lines
(`line_0`, `line_7`, `line_8`, `line_15` — rows/cols exactly 0 or 27, the
image border) sit at **exactly AUC = 0.5000**, chance level, carrying zero
same/different-digit signal. This makes sense structurally: MNIST digits
essentially never touch the image border, so these 4 lines run almost
entirely through background regardless of the image, and their signatures
are close to constant across the whole dataset. These 4 *are* being merged
into the 496-dim vector alongside 12 informative lines, contributing pure
noise to that vector for no benefit — a concrete, fixable inefficiency in
the current line placement (not the aggregation method itself).

## A discrepancy worth stating plainly, not smoothing over

AUC (threshold-independent) ranks `line_6` above `merged`. But at the
specific 90%-TPR operating point, `line_6`'s FPR (0.8392) is *slightly
worse* (higher) than merged's (0.8257) — the opposite ordering. Both numbers
are honestly reported above rather than picking whichever supports one
reading. Given both FPRs are extremely high (>0.8, meaning at 90% TPR you're
still accepting >80% of different-digit pairs as "same"), the more accurate
overall characterization is: **every measure here — merged and every
individual line — is a weak discriminator in absolute terms**, consistent
with everything already found in Phase 4 and the adversarial evaluation.
This diagnostic doesn't change that; it explains one concrete piece of
*why* (4 of 16 lines contribute nothing, sitting on the image border).

## Secondary finding: which lines rank highest

Horizontal lines dominate the top of the ranking (5 of the top 6 spots:
`line_6, line_5, line_1, line_4`, plus `line_10` vertical breaking in at
rank 5). The best-performing rows are around row 15–23 (roughly the lower-
middle of the image) and row 3.9 — not a single obvious band, but the worst
performers (chance-level) are unambiguously the 4 border lines. Not
sufficient sample to draw firm line-placement conclusions from alone, but
consistent with "avoid placing lines exactly at the image border" being a
cheap, likely-beneficial change for a future hyperparameter pass.

## Verdict

Terry's "path by path" framing does surface something real: one line beats
the merged distance, and the reason 15 others don't is now visible and
explainable (4 carry zero signal by construction — border placement — the
rest are individually weaker than the pool combined). This is not strong
evidence that per-line comparison as a *strategy* beats concatenation
outright — only one line wins, narrowly, and the operating-point numbers
don't unambiguously favor it either. It *is* good evidence that current line
placement wastes a quarter of the budget (4/16 lines) on structurally
uninformative positions, independent of whatever aggregation method is used
downstream. Per this task's scope, no further action taken here (no
per-line adversarial eval, no line-placement change) — this is the
foundational diagnostic the next decision should be based on.
