from signature_distance.per_line_diagnostics import run_per_line_auc_diagnostic


def test_run_per_line_auc_diagnostic_structure():
    # Small sample for speed - correctness of the underlying distance math
    # is already covered by test_distances.py; this just checks the
    # diagnostic's own orchestration/reporting shape.
    result = run_per_line_auc_diagnostic(n_per_class=3, seed=0)

    assert result["n_images"] == 30
    assert "merged" in result["measures"]
    line_keys = [k for k in result["measures"] if k.startswith("line_")]
    assert len(line_keys) == 16

    for key, entry in result["measures"].items():
        assert 0.0 <= entry["auc"] <= 1.0
        assert 0.0 <= entry["fpr_at_tpr90"] <= 1.0

    # ranked is sorted descending by AUC
    aucs = [entry["auc"] for _, entry in result["ranked"]]
    assert aucs == sorted(aucs, reverse=True)
    assert len(result["ranked"]) == 17

    assert isinstance(result["best_individual_line_beats_merged"], bool)
