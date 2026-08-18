# Cross-reference: flagged near-neighbor pairs vs. known MNIST test-set label errors

Validity check, not an accuracy-improvement step: are any of the near-neighbor pairs this
project's ratio-distribution analysis has flagged as high-ratio (large margin swing for a small
distance) secretly just a mislabeled example, rather than genuine model behavior? No retraining,
no dataset modification — this only cross-references already-saved index/label data.

**Revision note**: this file originally checked a different set of pairs for the original CNN
("8/2", "8/0") and logistic regression ("0/7") — those turned out to be stale, generated under an
earlier default of `run_mnist_experiment`'s `n_lipschitz_points` parameter. See "Root cause of the
prior mismatch" below. This version reflects the current, reproducible top pairs.

A structured version of this table (for programmatic use, e.g. the notebook's
label-error-cross-reference section) is saved alongside this file as `label_error_crossref.json`.

**Source of known label errors**: `mnist_lipschitz/data/known_label_errors_mnist_test.json`,
extracted from [cleanlab/label-errors](https://github.com/cleanlab/label-errors) (Apache-2.0) — 15
confirmed errors out of 100 cleanlab-flagged candidates, each confirmed by majority vote of 5
Amazon Mechanical Turk workers rejecting the original label. See that file for full methodology
and provenance.

**Method**: for each model, its top-6 highest-ratio Euclidean near-neighbor pairs' global MNIST
test-set indices were recovered from the already-saved near-neighbor search results
(`ii`/`jj`/`subset_idx` arrays in the relevant `results/*.npz` file) — no search was recomputed.

## Results

| Group | Rank | Pair | Test indices | Known label error? |
|---|---|---|---|---|
| Original CNN (weaker) | 1 | 6/5 | 1014, 4529 | No |
| Original CNN (weaker) | 2 | 4/9 | 8812, 882 | No |
| Original CNN (weaker) | 3 | 2/6 | 2200, 9679 | **Yes** |
| Original CNN (weaker) | 4 | 8/5 | 3559, 2413 | No |
| Original CNN (weaker) | 5 | 2/0 | 2098, 4527 | No |
| Original CNN (weaker) | 6 | 6/6 | 6847, 9529 | No |
| Original logistic regression | 1 | 1/1 | 5642, 2359 | No |
| Original logistic regression | 2 | 1/1 | 5642, 749 | No |
| Original logistic regression | 3 | 1/1 | 5642, 3253 | No |
| Original logistic regression | 4 | 5/1 | 4577, 5642 | No |
| Original logistic regression | 5 | 1/1 | 5642, 4308 | No |
| Original logistic regression | 6 | 3/3 | 5955, 1069 | No |
| Original MLP | 1 | 2/3 | 5381, 7821 | No |
| Original MLP | 2 | 6/5 | 1014, 4529 | No |
| Original MLP | 3 | 3/3 | 4430, 3943 | No |
| Original MLP | 4 | 8/0 | 2272, 1333 | No |
| Original MLP | 5 | 3/8 | 5955, 1613 | No |
| Original MLP | 6 | 1/1 | 5642, 749 | No |
| Stronger CNN, raw MNIST | 1 (top pair) | 9/4 | 882, 8812 | No |
| Stronger CNN, raw MNIST | 2 | 1/1 | 9368, 2803 | No |
| Stronger CNN, raw MNIST | 4 | 1/1 | 2803, 949 | No |
| Stronger CNN, raw MNIST | 5 | 1/1 | 2803, 1603 | No |

Notable pattern, not part of the original request but visible directly in the table above:
logistic regression's top-6 pairs are almost entirely the *same* digit (5 of 6 rows: four "1" vs.
"1" and one "3" vs. "3") rather than cross-digit confusions — closer to the "same-digit,
surprisingly-high-ratio" pattern documented for the stronger CNN's "1/1" pairs elsewhere in this
project than to the cross-digit-confusion story this section originally told for all three models.
Rank 4 ("5/1") is the one exception — a genuine cross-digit pair, not same-digit at all. **One
recurring point, not five or six independent findings**: test index 5642 is a member of 5 of these
6 pairs (ranks 1-5, i.e. every row except rank 6) — the same boundary-sitting image compared
against five different neighbors, not five separate discoveries. See the visual-check writeup in
`README.md`'s "Validity check" section for the full per-pair assessment
(`pair_diagnostic_lr_top6.png`).

## Conclusion

**Of the 22 pairs checked, exactly one involves a known label error**: the original CNN's current
rank-3 pair, test index 9679. Every other pair — including logistic regression's near-exclusively
same-digit top-6 — is clean. This still supports genuine model behavior as the general pattern,
with one confirmed, interesting exception (see below).

## Positive finding: the near-neighbor diagnostic caught a real labeling problem

The original CNN's actual rank-3 highest-ratio near-neighbor pair (Euclidean) is **6 vs. 2** (test
indices 9679 and 2200, ratio 2.41). Index **9679 is itself one of the 15 independently
mTurk-verified known label errors** in `known_label_errors_mnist_test.json` — its original label
"6" was rejected by a majority of mTurk workers (though without a single agreed-on replacement
label; see that file's `mturk_votes` for index 9679). This is a positive result, not a caveat: it's
direct, independent evidence that this project's near-neighbor diagnostic can surface genuine
*labeling* problems in MNIST's own test set, not just model confusions — the same kind of real,
interpretable failure the near-neighbor checkpoint's whole design is meant to catch, just from an
unexpected source.

## Root cause of the prior mismatch

An earlier version of this cross-reference checked pairs named in this notebook's own
"Highest-ratio near-neighbour pairs" commentary: "8/2" and "8/0" for the original CNN, "0/7" for
logistic regression. Those don't appear in the table above because they don't reproduce under the
current codebase — investigated via `git log -S` on `run_experiment.py` and confirmed by rerunning
`run_mnist_experiment()` with current code and the documented `seed=0`:

- The notebook's "8/2, 8/0, 7/2" (CNN) / "9/3, 2/4, 0/7" (logistic regression) text was written in
  commit `4db5ae9` (2026-08-12, "Extend ratio-distribution checkpoint to MLP/CNN and Mahalanobis
  distance"), when `run_mnist_experiment`'s `n_lipschitz_points` parameter defaulted to **300**.
- The very next day, commit `48bd088` ("Add covariance eigenvalue spectrum plot to
  mnist_lipschitz", 2026-08-13) changed that default to **1000** — apparently as an unrelated
  cleanup, with no update to the near-neighbor-pairs commentary to match.
- `n_lipschitz_points` sizes `query_idx`, the Lipschitz sub-method estimators' query set — which is
  then passed as `exclude_idx` to the ratio-distribution analysis's stratified near-neighbor subset
  draw (`data.py::stratified_subset_idx`). A different `exclude_idx` changes which candidates are
  available per class for that subset, so even with the identical seed, a different
  `n_lipschitz_points` produces a different subset and therefore different near-neighbor pairs.
- Model training itself is **not** affected by this parameter at all, and remains bit-for-bit
  reproducible: rerunning `run_mnist_experiment()` today produced train/test accuracies for all
  three models matching `README.md`'s documented table exactly to 4 decimal places. Only the
  near-neighbor subset composition shifted.

This is genuine stale documentation from a superseded parameter default — not file corruption, and
not non-determinism in training or the near-neighbor search itself. The old "8/2"/"8/0"/"0/7" pairs
are retired from this file and the notebook's commentary; the table above reflects the current,
reproducible pairs. `8/2` and `8/0`, for reference, still exist somewhere in the current CNN's full
near-neighbor pair list — just much further down (rank 47 and rank 26 of ~3590 unique pairs) than
"top pair" would suggest — and `0/7` doesn't appear anywhere in logistic regression's current list
under either distance metric.
