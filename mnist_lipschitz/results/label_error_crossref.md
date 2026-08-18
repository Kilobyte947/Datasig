# Cross-reference: flagged near-neighbor pairs vs. known MNIST test-set label errors

Validity check, not an accuracy-improvement step: are any of the near-neighbor pairs this
project's ratio-distribution analysis has flagged as high-ratio (large margin swing for a small
distance) secretly just a mislabeled example, rather than genuine model behavior? No retraining,
no dataset modification — this only cross-references already-saved index/label data.

**Source of known label errors**: `mnist_lipschitz/data/known_label_errors_mnist_test.json`,
extracted from [cleanlab/label-errors](https://github.com/cleanlab/label-errors) (Apache-2.0) — 15
confirmed errors out of 100 cleanlab-flagged candidates, each confirmed by majority vote of 5
Amazon Mechanical Turk workers rejecting the original label. See that file for full methodology
and provenance.

**Method**: for each named digit pair below, its two images' global MNIST test-set indices were
recovered from the already-saved near-neighbor search results (`ii`/`jj`/`subset_idx` arrays in
the relevant `results/*.npz` file) — no search was recomputed. Where a digit pair appears more
than once among a model's near-neighbor pairs, the highest-ratio (Euclidean) occurrence is used.

## Results

| Group | Pair | Test indices | Rank (Euclidean, of ~3590) | Known label error? |
|---|---|---|---|---|
| Original (weaker) CNN | 6/5 | 1014, 4529 | 1 | No |
| Original CNN | 8/2 | 4117, 7205 | 47 | No |
| Original CNN | 8/0 | 2272, 1333 | 26 | No |
| Original logistic regression | 9/3 | 9905, 8319 | 65 | No |
| Original logistic regression | 2/4 | 1374, 3188 | 731 | No |
| Original logistic regression | 0/7 | — | not present | not checkable (see note) |
| Original MLP | 2/7 | 8295, 1581 | 40 | No |
| Original MLP | 7/1 | 1039, 3039 | 179 | No |
| Original MLP | 1/6 | 1688, 2473 | 431 | No |
| Stronger CNN, raw MNIST | 9/4 (top pair) | 882, 8812 | 1 | No |
| Stronger CNN, raw MNIST | 1/1 (gallery rank 2) | 9368, 2803 | 2 | No |
| Stronger CNN, raw MNIST | 1/1 (gallery rank 4) | 2803, 949 | 4 | No |
| Stronger CNN, raw MNIST | 1/1 (gallery rank 5) | 2803, 1603 | 5 | No |

**Note on 0/7**: this digit pair doesn't appear anywhere among the original logistic regression's
~3590 unique near-neighbor pairs, under either distance metric, in the currently-saved
`results/mnist_experiment_arrays.npz` — there's nothing to check. See the data-provenance note
below.

## Conclusion

**None of the 12 checkable pairs above involve an image from the known-label-errors list.** This
is evidence the flagged pairs reflect genuine model behavior — real, if sometimes surprising,
sensitivity in the trained models' decision surfaces — rather than being artifacts of mislabeled
test data.

## Data-provenance caveat

The "6/5, 8/2, 8/0" pairs checked for the original CNN above come from this project's existing
notebook commentary describing that model's near-neighbor gallery. Cross-referencing against the
*currently-saved* `results/mnist_experiment_arrays.npz`, `6/5` matches exactly (rank 1, as
described), but `8/2` and `8/0` are real pairs that do exist in the data — just much further down
the ranked list (rank 47 and rank 26) than "top pair" would suggest, and `0/7` (one of the
original logistic-regression pairs) isn't present at all. This points to the notebook's
descriptive text reflecting a different run or subset draw than what's currently saved in
`results/`, not an error in this cross-reference. Flagged here rather than silently resolved, per
this project's convention of checking rather than assuming.

One direct consequence worth surfacing: the pair that *currently* occupies the original CNN's
actual rank-3 slot under Euclidean distance — where the notebook's text placed "8/2" — is a
different pair, **6 vs. 2** (indices 9679 and 2200, ratio 2.41). Index **9679 is itself a confirmed
known label error** (original label "6"; majority of mTurk workers rejected it, though without a
single-label replacement — see `known_label_errors_mnist_test.json`). This is not a match for any
pair named in the original request (it's a different pair than "8/2"), but it does mean that
*if* the original CNN's near-neighbor results were regenerated (not done here, per this task's
scope), its current actual top-3 highest-ratio pair would be confounded by a known-bad label —
worth keeping in mind if this ratio-distribution analysis is ever rerun on the original CNN.
