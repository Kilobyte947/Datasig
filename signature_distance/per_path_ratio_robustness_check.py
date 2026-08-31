"""Robustness check: does the per-path adversarial/control fold-ratio
finding (per_path_adversarial_eval.py's Finding 1) survive excluding lines
9 and 14?

Context: per_path_adversarial_eval.py found, for all 12 informative lines
in all 6 model x epsilon combinations, adversarial ratio > control ratio
(mean fold-ratios 4-11x). Separately (Finding 2/3 in that module), lines 9
and 14 were flagged as having systematically smaller baseline distances
than the other 10 lines, regardless of perturbation type - a confound
noted for the spike-counting analysis, but never checked against the
fold-ratio finding itself. This module checks that, read-only, using
already-computed results - no new signature computation, no changes to
distances.py / method_b_adversarial_eval.py / per_line_diagnostics.py /
per_path_adversarial_eval.py.
"""

from signature_distance.distances import per_line_distances  # noqa: F401 (kept for readers tracing provenance)
from signature_distance.per_path_adversarial_eval import (
    BEST_LINE_INDEX,
    INFORMATIVE_LINE_INDICES,
    run_per_path_adversarial_eval,
    summarize_informative_subset,
)

EXCLUDED_LINES = (9, 14)
ROBUST_LINE_INDICES = tuple(i for i in INFORMATIVE_LINE_INDICES if i not in EXCLUDED_LINES)


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def fold_ratio_robustness(results: dict) -> dict:
    """Per model/epsilon: per-line fold-ratio (mean adversarial ratio on
    genuinely flipped pairs / mean control ratio on the same pairs - same
    definition as per_path_adversarial_eval's Finding 1), reported for
    both the original 12-line informative set and the 10-line set with
    lines 9 and 14 excluded, side by side - plus each line's baseline
    distance (mean of dist_adv and dist_control, i.e. not
    perturbation-direction-dependent) and the Pearson correlation between
    baseline distance and fold-ratio across the 10-line set, to check
    whether the scale confound extends beyond lines 9/14.
    """
    summary = summarize_informative_subset(results)
    report = {}

    for name, mres in results["models"].items():
        report[name] = {}
        for eps, e in mres["eps"].items():
            s = summary[name][eps]
            n_flipped = s["n_flipped"]
            if n_flipped == 0:
                report[name][eps] = {"n_flipped": 0}
                continue

            fold_12 = {
                i: s["per_line"][i]["mean_ratio_adv_flipped"] / s["per_line"][i]["mean_ratio_control_matched"]
                for i in INFORMATIVE_LINE_INDICES
            }
            fold_10 = {i: fold_12[i] for i in ROBUST_LINE_INDICES}

            baseline_dist_10 = {
                i: ((e["dist_adv"][:, i].mean() + e["dist_control"][:, i].mean()) / 2).item()
                for i in ROBUST_LINE_INDICES
            }

            xs = [baseline_dist_10[i] for i in ROBUST_LINE_INDICES]
            ys = [fold_10[i] for i in ROBUST_LINE_INDICES]

            report[name][eps] = {
                "n_flipped": n_flipped,
                "fold_12": fold_12,
                "fold_10": fold_10,
                "mean_fold_12": sum(fold_12.values()) / len(fold_12),
                "mean_fold_10": sum(fold_10.values()) / len(fold_10),
                "min_fold_10": min(fold_10.values()),
                "max_fold_10": max(fold_10.values()),
                "all_10_survive_adv_gt_control": all(v > 1.0 for v in fold_10.values()),
                "baseline_dist_10": baseline_dist_10,
                "dist_fold_correlation_10": _pearson(xs, ys),
            }

    return report


def run_and_report(n_per_class: int = 20, epsilons=(0.02, 0.03, 0.05), seed: int = 0,
                    cnn_epochs: int = 3, strong_epochs: int = 3, verbose: bool = True) -> dict:
    """Reproducible entry point: regenerates results via
    per_path_adversarial_eval.run_per_path_adversarial_eval (same
    parameters used throughout this task chain - deterministic given
    seed=0, verified bit-for-bit reproducible across 3 independent prior
    runs) and runs fold_ratio_robustness on it. Prints a summary table.
    """
    results = run_per_path_adversarial_eval(
        n_per_class=n_per_class, epsilons=epsilons, seed=seed,
        cnn_epochs=cnn_epochs, strong_epochs=strong_epochs, verbose=verbose,
    )
    report = fold_ratio_robustness(results)

    for name, per_eps in report.items():
        for eps, r in per_eps.items():
            if r["n_flipped"] == 0:
                print(f"{name} eps={eps}: no flips, skipped")
                continue
            print(f"=== {name} eps={eps} (n_flipped={r['n_flipped']}) ===")
            print(f"  mean fold-ratio, 12 lines: {r['mean_fold_12']:.2f}x")
            print(f"  mean fold-ratio, 10 lines (excl. {EXCLUDED_LINES}): {r['mean_fold_10']:.2f}x")
            print(f"  min/max (10-line): {r['min_fold_10']:.2f}x / {r['max_fold_10']:.2f}x")
            print(f"  all 10 lines still adv > control: {r['all_10_survive_adv_gt_control']}")
            print(f"  dist-vs-fold correlation (10-line): {r['dist_fold_correlation_10']:.3f}")
            print()

    return {"results": results, "report": report}


if __name__ == "__main__":
    run_and_report()
